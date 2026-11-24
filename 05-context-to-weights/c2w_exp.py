"""05 — Zero-Label Context-to-Weights Internalization & Orthogonal Continual Memory.

Tests whether Qwen2.5-1.5B-Instruct can sequentially internalize 16 completely unseen
synthetic knowledge cards from its context window straight into its weights with ZERO
human labels (via KL self-distillation from P_base(. | card, q) -> P_student(. | q)),
and compares:
  1. sft_shared_lora    : sequential hard-label SFT on shared rank-16 LoRA
  2. kl_shared_lora     : sequential full-logit KL self-distillation on shared rank-16 LoRA
  3. kl_orthogonal_lora : sequential KL self-distillation into Gram-Schmidt orthogonalized
                          rank-2 LoRA slots (A_k _|_ A_1..A_{k-1}), merged into one model!

Evaluated at cards k = 1, 4, 8, 12, 16 on:
  - Current-card unseen-phrasing exact recall (with ZERO context prompt!)
  - All previously absorbed cards 1..k (catastrophic forgetting curve!)
  - General English text NLL (collateral capability drift!)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03-transformer-writer"))
sys.path.insert(0, "/content/tw")

from writer_loop import LORA_TARGETS, LoRALinear, attach_lora, unwrap_and_reset_lora


@dataclass(frozen=True)
class Card:
    cid: int
    name: str
    context: str
    train_qs: List[str]           # Unlabeled prompts used for self-distillation
    eval_qas: List[Tuple[str, str]]  # Unseen phrasings + exact target substring


def build_cards() -> List[Card]:
    """16 completely fictional, unmemorizable project dossiers (64 distinct atomic facts)."""
    specs = [
        ("Helios", "K-492", "March 17, 2031", "Valletta", "Maren Lindholm"),
        ("Borealis", "Q-817", "June 4, 2033", "Tromso", "Tariq Al-Mansoor"),
        ("Kestrel", "V-305", "October 22, 2029", "Nagano", "Sora Takahashi"),
        ("Zephyr", "R-641", "January 9, 2034", "Antofagasta", "Camila Rojas"),
        ("Cinder", "M-728", "August 14, 2032", "Reykjavik", "Einar Sigurdsson"),
        ("Solstice", "P-159", "April 28, 2030", "Dunedin", "Nadia Okafor"),
        ("Vesper", "X-934", "November 3, 2035", "Trieste", "Lorenzo Conti"),
        ("Halcyon", "T-260", "February 19, 2031", "Gdańsk", "Hanna Kowalska"),
        ("Obsidian", "N-583", "July 12, 2033", "Windhoek", "David Shikongo"),
        ("Meridian", "Z-416", "September 25, 2030", "Halifax", "Claire Beaumont"),
        ("Aurelia", "B-774", "May 6, 2034", "Coimbra", "Tiago Ferreira"),
        ("Cobalt", "D-609", "December 1, 2032", "Tartu", "Katriin Tamm"),
        ("Tundra", "F-852", "March 30, 2036", "Yellowknife", "Owen MacLeod"),
        ("Mirage", "J-195", "August 8, 2031", "Muscat", "Leila Al-Harthi"),
        ("Quasar", "H-348", "October 15, 2035", "Tucson", "Mateo Alvarez"),
        ("Saffron", "W-911", "June 21, 2032", "Udaipur", "Ananya Deshmukh"),
    ]
    cards: List[Card] = []
    for idx, (proj, code, date, city, lead) in enumerate(specs, start=1):
        ctx = (
            f"Internal Dossier — Project {proj}: "
            f"The access cipher for Project {proj} is {code}. "
            f"Project {proj} is scheduled to launch on {date} from the {city} facility, "
            f"under lead director {lead}."
        )
        # Unlabeled training questions (never given ground-truth answers during training!)
        train_qs = [
            f"What is the access cipher for Project {proj}?",
            f"When is Project {proj} scheduled to launch?",
            f"From which facility city will Project {proj} launch?",
            f"Who is the lead director of Project {proj}?",
            f"State the cipher code assigned to Project {proj}.",
            f"Give the launch date of Project {proj}.",
            f"Name the city hosting Project {proj}.",
            f"Identify the person leading Project {proj}.",
        ]
        # Distinct unseen phrasings for zero-context evaluation
        eval_qas = [
            (f"Project {proj} uses which access cipher?", code),
            (f"On what exact date does Project {proj} launch?", date),
            (f"Which city is the launch site for Project {proj}?", city),
            (f"Who serves as director for Project {proj}?", lead),
        ]
        cards.append(Card(idx, proj, ctx, train_qs, eval_qas))
    return cards


ENGLISH_PROBES = [
    "In 1936 Alan Turing defined a mathematical model of computation consisting of an infinite tape divided into cells.",
    "The northern lights occur when charged particles from the solar wind collide with gases in Earth's upper atmosphere.",
    "After a long winter, the mountain passes finally cleared in late May, allowing supply trucks to reach the valley.",
    "Modern compilers translate high-level source code into machine instructions through several optimization passes.",
    "Coffee plants grow best in tropical highlands with steady rainfall, volcanic soil, and moderate daytime temperatures.",
    "Although the experiment produced a clear signal in small trials, the effect vanished once a proper control was added.",
]


def format_prompt(tokenizer, question: str, context: str | None = None) -> str:
    if context is not None:
        msgs = [
            {"role": "system", "content": f"Answer concisely in one short phrase using this dossier.\n{context}"},
            {"role": "user", "content": question},
        ]
    else:
        msgs = [
            {"role": "system", "content": "Answer concisely in one short phrase."},
            {"role": "user", "content": question},
        ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def english_nll(model: nn.Module, tokenizer, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_toks = 0
    for text in ENGLISH_PROBES:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_toks += int(ids.shape[1] - 1)
    return total_loss / max(1, total_toks)


@torch.no_grad()
def generate_answers(model: nn.Module, tokenizer, prompts: List[str], device: torch.device, max_new: int = 24) -> List[str]:
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    out_texts: List[str] = []
    for s in range(0, len(prompts), 16):
        chunk = prompts[s : s + 16]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        in_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        for row in gen:
            out_texts.append(tokenizer.decode(row[in_len:], skip_special_tokens=True).strip())
    return out_texts


@torch.no_grad()
def evaluate_cards(model: nn.Module, tokenizer, cards: List[Card], device: torch.device, with_context: bool = False) -> Dict[int, float]:
    prompts = []
    meta = []
    for c in cards:
        for q, ans in c.eval_qas:
            prompts.append(format_prompt(tokenizer, q, context=c.context if with_context else None))
            meta.append((c.cid, ans.lower()))
    preds = generate_answers(model, tokenizer, prompts, device)
    hits_by_cid: Dict[int, List[int]] = {}
    for (cid, ans), pred in zip(meta, preds):
        hits_by_cid.setdefault(cid, []).append(int(ans in pred.lower()))
    return {cid: sum(v) / len(v) for cid, v in hits_by_cid.items()}


# ---------- Orthogonal-Subspace LoRA (Gram-Schmidt A_k _|_ span(A_1..A_{k-1})) ----------

class OrthogonalSlotLoRA(nn.Module):
    """Stores frozen past slots (A_prev, B_prev) and one active trainable slot (A_cur, B_cur)
    whose input rows A_cur are projected orthogonal to all prior A_prev rows!
    At inference time, all slots simply sum into one linear update (zero external routing!).
    """

    def __init__(self, base: nn.Linear, slot_rank: int = 2, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.slot_rank = slot_rank
        self.scale = alpha / slot_rank
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dev = base.weight.device
        # Orthonormal basis Q_prev of shape [k_prev, in_features] spanning all prior A rows
        self.register_buffer("Q_prev", torch.zeros(0, self.in_features, device=self.dev, dtype=torch.float32))
        self.A_cur: nn.Parameter | None = None
        self.B_cur: nn.Parameter | None = None

    def open_new_slot(self, generator: torch.Generator) -> List[nn.Parameter]:
        A = torch.empty(self.slot_rank, self.in_features, device=self.dev, dtype=torch.float32)
        nn.init.normal_(A, mean=0.0, std=1.0 / math.sqrt(self.in_features), generator=generator)
        # Project initial A orthogonal to Q_prev
        if self.Q_prev.shape[0] > 0:
            A = A - (A @ self.Q_prev.T) @ self.Q_prev
        B = torch.zeros(self.out_features, self.slot_rank, device=self.dev, dtype=torch.float32)
        self.A_cur = nn.Parameter(A)
        self.B_cur = nn.Parameter(B)
        return [self.A_cur, self.B_cur]

    @torch.no_grad()
    def project_active_orthogonal(self) -> None:
        if self.A_cur is not None and self.Q_prev.shape[0] > 0:
            self.A_cur.sub_((self.A_cur @ self.Q_prev.T) @ self.Q_prev)

    @torch.no_grad()
    def commit_slot_to_base(self) -> None:
        """Merge active slot into base.weight and expand orthonormal basis Q_prev."""
        if self.A_cur is None or self.B_cur is None:
            return
        self.project_active_orthogonal()
        delta = (self.B_cur @ self.A_cur) * self.scale
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        # Append orthonormalized rows of A_cur to Q_prev
        rows = self.A_cur.detach().clone()
        new_basis = []
        cur_Q = self.Q_prev
        for i in range(rows.shape[0]):
            v = rows[i]
            if cur_Q.shape[0] > 0:
                v = v - (v @ cur_Q.T) @ cur_Q
            nrm = v.norm()
            if nrm > 1e-6:
                v = v / nrm
                cur_Q = torch.cat([cur_Q, v.unsqueeze(0)], dim=0)
                new_basis.append(v)
        self.Q_prev = cur_Q
        self.A_cur = None
        self.B_cur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.A_cur is not None and self.B_cur is not None:
            delta = F.linear(F.linear(x.float(), self.A_cur), self.B_cur) * self.scale
            return out + delta.to(out.dtype)
        return out


def attach_ortho_slots(model: nn.Module, slot_rank: int = 2, alpha: float = 8.0) -> List[OrthogonalSlotLoRA]:
    slots: List[OrthogonalSlotLoRA] = []
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in LORA_TARGETS and isinstance(child, nn.Linear):
                w = OrthogonalSlotLoRA(child, slot_rank=slot_rank, alpha=alpha)
                setattr(module, child_name, w)
                slots.append(w)
    return slots


def unwrap_ortho_slots(model: nn.Module) -> None:
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, OrthogonalSlotLoRA):
                setattr(module, child_name, child.base)


# ---------- Zero-Label Teacher Rollout + Top-K Logit Cache ----------

@torch.no_grad()
def build_teacher_distill_data(
    teacher_model: nn.Module,
    tokenizer,
    card: Card,
    device: torch.device,
    topk: int = 32,
) -> List[Dict]:
    """Generate teacher rollouts from P_base(. | card, q) and cache top-k logit targets
    paired with the CARD-DELETED student prompt! Zero human labels used.
    """
    teacher_prompts = [format_prompt(tokenizer, q, context=card.context) for q in card.train_qs]
    answers = generate_answers(teacher_model, tokenizer, teacher_prompts, device, max_new=24)

    items = []
    for q, t_prompt, ans in zip(card.train_qs, teacher_prompts, answers):
        s_prompt = format_prompt(tokenizer, q, context=None)
        t_p_ids = tokenizer(t_prompt, add_special_tokens=False)["input_ids"]
        s_p_ids = tokenizer(s_prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(ans + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if not a_ids:
            continue

        # Run frozen base teacher on (t_prompt + ans) to capture exact token-level distributions
        t_full = torch.tensor([t_p_ids + a_ids], dtype=torch.long, device=device)
        t_logits = teacher_model(input_ids=t_full, use_cache=False).logits[0].float()
        # Logits predicting a_ids[j] sit at position len(t_p_ids) - 1 + j
        start = len(t_p_ids) - 1
        ans_logits = t_logits[start : start + len(a_ids)]  # [L_ans, V]
        vals, idxs = torch.topk(ans_logits, k=topk, dim=-1)
        probs = F.softmax(vals, dim=-1)

        s_full = torch.tensor(s_p_ids + a_ids, dtype=torch.long)
        s_labels = torch.tensor([-100] * len(s_p_ids) + a_ids, dtype=torch.long)
        items.append({
            "s_ids": s_full,
            "s_labels": s_labels,
            "s_ans_start": len(s_p_ids) - 1,
            "ans_len": len(a_ids),
            "topk_idx": idxs.cpu(),
            "topk_prob": probs.cpu(),
        })
    return items


def train_on_card_items(
    model: nn.Module,
    trainable_params: List[nn.Parameter],
    items: List[Dict],
    mode: str,  # "sft" | "kl"
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
    ortho_modules: List[OrthogonalSlotLoRA] | None = None,
) -> None:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        # Microbatch of 4 questions per step
        batch = [rng.choice(items) for _ in range(4)]
        for it in batch:
            ids = it["s_ids"].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(input_ids=ids, use_cache=False).logits[0]
            if mode == "sft":
                labels = it["s_labels"].to(device)
                loss = F.cross_entropy(logits[:-1].float(), labels[1:], ignore_index=-100) / len(batch)
            else:
                st = it["s_ans_start"]
                L = it["ans_len"]
                s_logp = F.log_softmax(logits[st : st + L].float(), dim=-1)
                idx = it["topk_idx"].to(device)
                tp = it["topk_prob"].to(device)
                gathered = torch.gather(s_logp, dim=-1, index=idx)
                # Forward KL (cross-entropy to top-k teacher distribution)
                loss = -(tp * gathered).sum(dim=-1).mean() / len(batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        if ortho_modules is not None:
            for om in ortho_modules:
                om.project_active_orthogonal()
        sched.step()
    model.eval()


def run_sequential_arm(
    model_name: str,
    arm: str,
    cards: List[Card],
    teacher_cache: Dict[int, List[Dict]],
    tokenizer,
    steps_per_card: int,
    lr: float,
    seed: int,
    device: torch.device,
    dtype,
) -> Dict:
    print(f"\n=== Running Arm: {arm} ({len(cards)} cards sequentially) ===", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    ortho_modules = None
    shared_params = None
    if arm in ("sft_shared_lora", "kl_shared_lora"):
        shared_params = attach_lora(model, r=16, alpha=32.0)
    elif arm == "kl_orthogonal_lora":
        ortho_modules = attach_ortho_slots(model, slot_rank=2, alpha=16.0)
    else:
        raise ValueError(arm)

    mode = "sft" if arm == "sft_shared_lora" else "kl"
    checkpoints = []
    gen = torch.Generator(device=device).manual_seed(seed)

    for idx, card in enumerate(cards, start=1):
        if arm == "kl_orthogonal_lora":
            assert ortho_modules is not None
            trainable = []
            for om in ortho_modules:
                trainable.extend(om.open_new_slot(gen))
            train_on_card_items(
                model, trainable, teacher_cache[card.cid],
                mode="kl", steps=steps_per_card, lr=lr, seed=seed + idx, device=device,
                ortho_modules=ortho_modules,
            )
            for om in ortho_modules:
                om.commit_slot_to_base()
        else:
            assert shared_params is not None
            train_on_card_items(
                model, shared_params, teacher_cache[card.cid],
                mode=mode, steps=steps_per_card, lr=lr, seed=seed + idx, device=device,
            )

        if idx in (1, 4, 8, 12, 16) or idx == len(cards):
            scores = evaluate_cards(model, tokenizer, cards[:idx], device, with_context=False)
            cur_acc = scores[card.cid]
            mean_all = sum(scores.values()) / len(scores)
            early_acc = scores[cards[0].cid]
            nll = english_nll(model, tokenizer, device)
            print(
                f"  after card {idx:2d}/{len(cards)} ({card.name:8s}): "
                f"current={cur_acc*100:5.1f}% | mean(1..{idx:2d})={mean_all*100:5.1f}% | "
                f"card1_ret={early_acc*100:5.1f}% | eng_NLL={nll:.4f}",
                flush=True,
            )
            checkpoints.append({
                "after_card": idx,
                "current_acc": cur_acc,
                "mean_absorbed_acc": mean_all,
                "card1_retention": early_acc,
                "english_nll": round(nll, 4),
                "per_card": scores,
            })

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"arm": arm, "checkpoints": checkpoints}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps-per-card", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/c2w_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    cards = build_cards()
    t0 = time.time()
    print(f"05-context-to-weights | device={device} dtype={dtype} model={args.model} cards={len(cards)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    base_model.eval()

    base_nll = english_nll(base_model, tokenizer, device)
    floor_scores = evaluate_cards(base_model, tokenizer, cards, device, with_context=False)
    ceil_scores = evaluate_cards(base_model, tokenizer, cards, device, with_context=True)
    floor_mean = sum(floor_scores.values()) / len(floor_scores)
    ceil_mean = sum(ceil_scores.values()) / len(ceil_scores)
    print(f"Base English NLL={base_nll:.4f} | no_context floor={floor_mean*100:.1f}% | with_context ceiling={ceil_mean*100:.1f}%", flush=True)

    print("Generating zero-label teacher rollouts & top-k logit caches for all 16 cards...", flush=True)
    teacher_cache = {c.cid: build_teacher_distill_data(base_model, tokenizer, c, device) for c in cards}

    del base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arm_results = {}
    for arm in ("sft_shared_lora", "kl_shared_lora", "kl_orthogonal_lora"):
        arm_results[arm] = run_sequential_arm(
            args.model, arm, cards, teacher_cache, tokenizer,
            steps_per_card=args.steps_per_card, lr=args.lr, seed=args.seed, device=device, dtype=dtype,
        )
        out_path.write_text(json.dumps({
            "config": vars(args),
            "elapsed_s": round(time.time() - t0, 1),
            "base_english_nll": round(base_nll, 4),
            "no_context_floor": floor_mean,
            "with_context_ceiling": ceil_mean,
            "arms": arm_results,
        }, indent=2))
    print(f"\nSaved -> {out_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
