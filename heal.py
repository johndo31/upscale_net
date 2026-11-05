#!/usr/bin/env python
"""Heal a grown model: continued pre-training so the new capacity gets used.

Memory-lean on purpose so a 1B model trains on a 16 GB Mac:
  * fp32 master weights, bf16 autocast for compute
  * Adafactor (factored second moments, ~no optimizer state)
  * gradient checkpointing
On a CUDA box the same script works with --device cuda and a bigger batch.

Streams text from a Hugging Face dataset, packs it into fixed-length
sequences, and logs train / held-out loss to runs/<name>/log.jsonl.

Checkpoints: every --save-every steps a bf16 model snapshot (checkpoint-N/) and a
full training state (state.pt: fp32 weights, optimizer, scheduler step, RNG,
data position). `--resume` continues from state.pt; `--stop-after N` stops
after step N (state saved) so a run can be paused, compared, and resumed later.
"""
import os
import random
import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def example_text(tok, ex):
    """Turn a dataset row into plain text: `text` column, chat `messages`, or problem/solution pairs."""
    if "text" in ex:
        return ex["text"]
    if "messages" in ex:
        return tok.apply_chat_template(ex["messages"], tokenize=False)
    if "problem" in ex:
        # OpenR1-style rows: prefer the model-written reasoning trace (keeps <think> habits),
        # fall back to the reference solution; render through the chat template.
        answer = (ex.get("generations") or [None])[0] or ex.get("solution") or ex.get("answer") or ""
        msgs = [{"role": "user", "content": ex["problem"]}, {"role": "assistant", "content": answer}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            return f"{ex['problem']}\n\n{answer}"
    if "question" in ex and "answer" in ex:
        return f"{ex['question']}\n\n{ex['answer']}"
    raise KeyError(f"don't know how to read a row with keys {list(ex)}")


class PackedStream:
    """Yields LongTensors of length seq_len, documents joined by EOS. Loops over the
    dataset forever (a small self-distilled file is simply trained for several epochs).
    Resumable: state() records epoch, documents consumed in this epoch and the leftover
    token buffer; load_state() skips ahead without re-tokenizing."""

    def __init__(self, tok, dataset, seq_len):
        self.tok, self.dataset, self.seq_len = tok, dataset, seq_len
        self.buf, self.epoch, self.docs = [], 0, 0

    def state(self):
        return {"epoch": self.epoch, "docs": self.docs, "buf": list(self.buf)}

    def load_state(self, st):
        self.epoch, self.docs, self.buf = st["epoch"], st["docs"], list(st["buf"])
        self.__dict__.pop("_it", None)               # reopen the dataset at the saved position

    def _epoch_dataset(self):
        ds = self.dataset
        for e in range(1, self.epoch + 1):           # replay the per-epoch reshuffles
            ds = ds.shuffle(seed=e, buffer_size=1000) if hasattr(ds, "shuffle") else ds
        return ds.skip(self.docs) if self.docs and hasattr(ds, "skip") else ds

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.buf) < self.seq_len:
            self._fill()
        x = torch.tensor(self.buf[:self.seq_len])
        self.buf = self.buf[self.seq_len:]
        return x

    def _fill(self):
        if not hasattr(self, "_it"):
            self._it = iter(self._epoch_dataset())
        try:
            ex = next(self._it)
        except StopIteration:
            self.epoch += 1; self.docs = 0
            print(f"[data] finished epoch {self.epoch} of {getattr(self.dataset, '_name', 'dataset')}, looping", flush=True)
            self._it = iter(self._epoch_dataset())
            ex = next(self._it)
        self.docs += 1
        self.buf.extend(self.tok(example_text(self.tok, ex)).input_ids + [self.tok.eos_token_id])


def packed_sequences(tok, dataset, seq_len):
    return PackedStream(tok, dataset, seq_len)


def open_dataset(name, subset):
    """HF hub dataset, or a local .json/.jsonl file (e.g. from self_distill.py)."""
    if name.endswith((".json", ".jsonl")):
        ds = load_dataset("json", data_files=name, split="train", streaming=True)
    else:
        ds = load_dataset(name, name=subset, split="train", streaming=True)
    ds._name = name
    return ds


class Mixed:
    """Draw each sequence from `secondary` with probability `ratio`, else from `primary`."""

    def __init__(self, primary, secondary, ratio, seed=0):
        self.primary, self.secondary, self.ratio = primary, secondary, ratio
        self.rng = random.Random(seed)

    def state(self):
        return {"primary": self.primary.state(), "secondary": self.secondary.state(), "rng": self.rng.getstate()}

    def load_state(self, st):
        self.primary.load_state(st["primary"]); self.secondary.load_state(st["secondary"])
        self.rng.setstate(st["rng"])

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.secondary) if self.rng.random() < self.ratio else next(self.primary)


def mixed(primary, secondary, ratio, seed=0):
    return Mixed(primary, secondary, ratio, seed)


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    losses = []
    for x in batches:
        x = x.to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            out = model(input_ids=x, labels=x)      # HF shifts labels internally
        losses.append(out.loss.float().item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="run name -> runs/<name>/")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="HF dataset or local .jsonl")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--mix-dataset", default=None,
                    help="second dataset blended in, e.g. open-r1/OpenR1-Math-220k (keeps reasoning skills alive)")
    ap.add_argument("--mix-subset", default=None)
    ap.add_argument("--mix-ratio", type=float, default=0.2, help="fraction of sequences drawn from --mix-dataset")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor",
                    help="adafactor: ~no optimizer memory (fits a 16 GB Mac / 8B on 80 GB). adamw: standard, needs ~4x weight memory")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak lr (default 1e-3 for adafactor = relative step size, 3e-5 for adamw)")
    ap.add_argument("--save-every", type=int, default=0, help="save a model checkpoint every N steps (0 = only at end)")
    ap.add_argument("--keep-checkpoints", type=int, default=1, help="how many intermediate checkpoints to keep (older ones are deleted)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--heldout", type=int, default=16, help="held-out sequences (batches of --batch)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    ap.add_argument("--eval-only", action="store_true", help="just report held-out loss of --model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/state.pt if it exists")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after this step (state saved); resume later with --resume. 0 = run to --steps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = open_dataset(args.dataset, args.subset)
    stream = packed_sequences(tok, ds, args.seq_len)
    if args.mix_dataset:
        ds2 = open_dataset(args.mix_dataset, args.mix_subset)
        stream = mixed(stream, packed_sequences(tok, ds2, args.seq_len), args.mix_ratio, args.seed)

    # fixed held-out set = first N batches of the stream; training starts after it
    held = []
    for _ in range(args.heldout):
        held.append(torch.stack([next(stream) for _ in range(args.batch)]))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.config.use_cache = False
    if args.eval_only:
        print(json.dumps({"model": args.model, "heldout_loss": evaluate(model, held, args.device)}))
        return
    model.gradient_checkpointing_enable()
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    if args.lr is None:
        args.lr = 1e-3 if args.optimizer == "adafactor" else 3e-5
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "adafactor":
        opt = torch.optim.Adafactor(groups, lr=args.lr)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=args.device.startswith("cuda"))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(run / "log.jsonl", "a")
    def record(**kw):
        kw["time"] = time.time()
        log.write(json.dumps(kw) + "\n"); log.flush()
        print(json.dumps(kw), flush=True)

    start = 0
    state_path = run / "state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)   # optimizer moves its state to the param device
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        stream.load_state(st["stream"]); torch.set_rng_state(st["rng"])
        if st.get("cuda_rng") is not None and args.device.startswith("cuda"):
            torch.cuda.set_rng_state(st["cuda_rng"])
        start = st["step"]
        del st
        record(event="resume", step=start, steps=args.steps, heldout_loss=evaluate(model, held, args.device))
    else:
        record(event="start", model=args.model, params=n_params, steps=args.steps, optimizer=args.optimizer, lr=args.lr,
               tokens_per_step=args.batch * args.grad_accum * args.seq_len,
               heldout_loss=evaluate(model, held, args.device))

    def save_state(step):
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "stream": stream.state(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if args.device.startswith("cuda") else None,
                    "args": vars(args)}, tmp)
        os.replace(tmp, state_path)                 # atomic: a crash mid-write never leaves a broken state.pt
        record(event="saved_state", step=step, path=str(state_path), gb=round(state_path.stat().st_size / 1e9, 1))

    t0 = time.time()
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x = torch.stack([next(stream) for _ in range(args.batch)]).to(args.device)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / args.grad_accum   # HF shifts labels internally
            loss.backward()
            loss_acc += loss.float().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        last = done == args.steps or done == args.stop_after
        rec = dict(step=done, loss=round(loss_acc, 4), lr=lr_at(step),
                   tok_s=round((done - start) * args.batch * args.grad_accum * args.seq_len / (time.time() - t0)))
        if done % args.eval_every == 0 or last:
            rec["heldout_loss"] = evaluate(model, held, args.device)
        record(**rec)
        if (args.save_every and done % args.save_every == 0 and done < args.steps) or (last and done < args.steps):
            save(model, tok, run / f"checkpoint-{done}", record)
            prune_checkpoints(run, args.keep_checkpoints, record)
            save_state(done)
        if done == args.stop_after and done < args.steps:
            record(event="stopped", step=done, steps=args.steps, note="resume with --resume")
            return

    save(model, tok, run / "model", record)
    prune_checkpoints(run, 0, record)          # final model saved: intermediate checkpoints are redundant
    if state_path.exists():
        state_path.unlink()                    # the final model is saved; the 15 GB training state is no longer needed


def prune_checkpoints(run, keep, record):
    import shutil
    cks = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in cks[:max(0, len(cks) - keep)]:
        shutil.rmtree(old, ignore_errors=True)
        record(event="deleted", path=str(old))


def save(model, tok, out, record):
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(out, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(out)
    record(event="saved", path=str(out))


if __name__ == "__main__":
    main()
