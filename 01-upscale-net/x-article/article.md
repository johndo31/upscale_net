_Growing a model is free and exact, but the new capacity starts empty: even after a billion tokens of training, it bought only a rounding error over the small model._

_Qwen2.5-0.5B grown to 1B, healed on a billion tokens against an honest control, for about $77._

_By Niklas, [drniklas.com](https://drniklas.com). I also host the [Venture Grade](https://venturegrade.drniklas.com) podcast._

## The question

["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/), by Takehi, Teoh, Zhang and Khattab at MIT CSAIL, shows a frontier LLM reading and writing neural network weights directly, with no gradient descent. It made me wonder: can you "program" a small model into a big one? Take a
model off Hugging Face, make it twice the size by computation rather than by
training, and does that buy you anything?

Short answer: the growing part is easy and exact, and has nothing to do with
LLMs. The hard part is what comes after, and at the budget I could afford, the
honest answer to "does it buy anything" is: a rounding error.

## Growing without changing anything

The trick is old (Net2Net, 2015) and simple. Every new piece of the network
is initialised so it contributes exactly zero:

* new residual dimensions are zero-padded; the layers that write into them
  (embedding, attention output, MLP output) get zero weights there, the layers
  that read from them get random weights, which read zeros
* new attention heads and MLP neurons get random input weights and zero
  output weights
* new layers are copies of a neighbour with their output projections zeroed,
  so they are the identity on the residual stream
* RMSNorm averages over the hidden size, so zero-padding changes its scale by
  d/D; multiplying the norm weights by sqrt(d/D) and the epsilon by d/D cancels
  that exactly

Result: Qwen2.5-0.5B (hidden 896, 24 layers, 14 heads) becomes a 1.05B model
(hidden 1280, 28 layers, 20 heads) whose logits match the original to float32
rounding (max difference 6e-5, identical greedy output on every prompt). It
takes a minute on a laptop. The same code grows DeepSeek-R1-Distill-Qwen-1.5B
to the exact Llama-3-8B shape in three minutes; I did not have the budget to
heal that one.

## So the 1B is exactly as smart as the 0.5B. Now what?

The new capacity is empty. Filling it means training, which the growth
literature calls healing. Two things matter here that I think most write-ups
skip:

1. **The control.** Any model improves if you train it on another billion
   tokens. The only fair comparison is the small model trained on exactly the
   same tokens with the same settings. If grown-and-healed does not beat that,
   growth bought nothing.
2. **Equal tokens is not equal cost.** The small model trains 1.6x faster per
   token. A compute-matched control would see 1.6 billion tokens in the same
   GPU hours. I ran the token-matched control and report both framings.

Setup: 1B tokens of fineweb-edu, AdamW, peak learning rate 3e-5 (1e-4 damaged
the pretrained weights within 500 steps), cosine to 10%, 65k-token batches,
one H100. Benchmarks with lm-eval: hellaswag, arc, piqa, winogrande, wikitext
perplexity, full task sets, plus five real 1B-class models for scale. Every number below comes from a single training run per variant, one seed.

## Results

[[03-results.png]]

For scale, here are five real 1B-class models on the same five benchmarks:

[[06-scale.png]]

What this says:

* **At equal tokens, growth wins by 0.006 nats** of held-out loss and by
  nothing on benchmarks. Every benchmark difference between the grown model
  and the control is within the noise of the tasks. Twice the parameters, same
  tokens, same model as far as the tasks can tell.
* **At equal compute, growth probably loses.** The control finished in 5.7 hours, the grown model in 9. Give the small model the same 9 hours and it sees 1.6B tokens, and its loss was still falling when it stopped. I did not run that comparison, so this is an expectation, not a measurement.
* **The curve shows why the gain is small.** The control leads for the first
  10,000 steps, because it has no random new units adding noise to a tuned
  network. The grown model catches up around step 10,000 and pulls away by
  0.006 over the last third. More capacity shows up late, and slowly.

[[04-crossover.png]]

* **Both runs got worse at Wikipedia.** Two points of wikitext perplexity,
  identical for control and grown. That is fineweb-edu pulling the model
  toward web text, not an effect of growth.
* **A billion tokens is nothing.** The best current models in this size class (OLMo-2-1B, Qwen2.5-1.5B, SmolLM2-1.7B) are 10 to 20 points ahead. They were trained on trillions of tokens, so one billion is well under 0.1% of their budget. Size alone is not the story either: Pythia-1B, twice the source's size, scores below it.

## Can the empty part be filled with something?

Before the long run I spent about $21 on four short pilots: four ways to initialise
the new capacity, each trained on the first 131M tokens of the *same* 1B
schedule, so the winner could simply be resumed. The rule for picking was fixed
in advance: lowest held-out loss at step 2,000, ties within 0.01 go to the
simplest option.

* **Zero:** random reader, zero writer. Empty.
* **LLM plan:** written by an LLM from weight and activation statistics.
* **Split:** Net2Net. Copy a neuron and halve both outputs.
* **Guided:** split, but the busiest units go first.

[[05-pilots.png]]

Empty won the pick, but only its lead over split and guided means much. The plan's 0.002 deficit is a draw, inside the tie margin and from one seed. Two things I learned from these runs are, I think, the most useful part of this post.

[[02-constructions.png]]

**Exact splits make twins.** A split copies a neuron's reader and halves both
writers, which preserves the function. But the two copies then see the same
input and have the same writer, so they get the same gradient, forever. I measured it on the split model after 131M tokens: in the four layers I checked (0, 10, 20, 23), the mean cosine between twin neurons is 1.0000 for all three weight matrices, against 0.007 for random pairs. That checkpoint is stored in bf16, so differences too small for bf16 would not show. To that precision they have not separated. The twins are about 40% of the new parameters, so a split model is roughly the zero model with that share locked. Net2Net added explicit noise for exactly this reason. I left it out to keep growth exact, and that is my best explanation for why split trails zero. "Guided", which splits the busiest units, did worst, which fits the same story.

**An LLM reading the model and writing the init did not beat leaving it
empty.** The fourth variant was the weight-space-programming idea in its
honest form, with one difference: the MIT authors' model read raw weights as text, while mine read statistics computed from them. The LLM assistant I ran this whole project with studied the 0.5B's weight and activation statistics (which neurons fire hardest, which heads carry their
group, which layers move the residual stream most, which units are
near-duplicates) and wrote a plan: put the four new layers after layers 0,
12, 21 and 23 where the work is; give the busiest neurons and heads a copy
that shares their reader but starts with a zero writer, so it fires from step
0 but is free to learn its own output, no twin problem; leave the rest empty.
The code kept it exact; the choices were the model's. It took the early hit
the model itself had predicted (the copies double the first updates on the
busiest units), recovered by step 500, and then tracked zero within 0.005. Ahead of the two rules, level with leaving it empty. The plan, and the
reasoning the model wrote down with it, are in the repo.

## Editing weights is training, just with a different optimizer

This is the point I want to make clearly, because the weight-space
programming framing hides it.

Training and weight editing are the same kind of thing: an update rule that
takes the current weights plus some information and produces new weights.
They differ in where the information comes from and how it is applied.

* **Gradient descent.** The information comes from data, through a loss. The
  update is local and numeric: a batch, a gradient, a tiny step, repeated
  millions of times. The optimizer knows nothing about transformers. It only
  knows which direction lowers the loss right now.
* **An LLM editing weights.** The information comes from the editor: its prior
  about how networks work, plus whatever it reads from the weights and
  activations. The edit can be structured and global. Copy this head, zero
  that row, rescale the norm, make this neuron fire on that feature. No data,
  no loss, no gradient at edit time.

So the editor is a learned optimizer. It was itself trained by gradient
descent, and its edits are amortized knowledge from that training. "Without
training" is only true of the moment the edit happens. The information was
paid for somewhere, and there is no free lunch:

* If the editor knows something the data would have to teach slowly, its
  edit is cheaper than training. That is the interesting case: known
  structure, known circuits, known fixes.
* If the editor knows nothing useful, it is a worse optimizer, guessing
  numbers instead of measuring them.

This experiment showed both ends. The growth step is a weight edit with no
training, and it is exact precisely because it adds zero information. The LLM-written plan was a weight edit informed by reading the model, and it did not beat the empty init. That is one plan, one seed and 131M tokens, from statistics rather than raw weights, so it does not show that an LLM editor cannot help. It shows that this edit gave no measurable head start over an empty init on the same tokens.

That is the core of it. Making the model bigger is free because
nothing enters. Making the new part worth anything means information has to
enter, from one of three places:

1. **Data**, through gradient descent. Healing, and the control.
2. **Another model**, through its weights or its outputs. A model writing the
   init, or distillation.
3. **Inference-time compute with a verifier.** The model generates, something
   checks, only the checked part is trained on. Self-distillation.

[[07-sources.png]]

Which source is cheapest per unit of capability is the real question, and
"an LLM edits the weights" is one answer to it, not a way around it.

The picture I find useful is a coach. A coach is another model: they carry a
prior about how skills work, they watch the athlete (our activation
statistics), and they intervene structurally. "Move your elbow", not "swing
ten thousand times and keep what works". That is cheap exactly when the
coach knows something the athlete would learn slowly on their own, and
useless when they do not. The analogy breaks in one place, and the break is
the point: a coach never touches the athlete's synapses. The instruction
goes in through the senses and the athlete's own learning still has to turn
it into skill. That is prompting, or curated training data. Weight editing
is a surgeon who has seen many brains and rewires one directly, so the
athlete wakes up with the elbow fixed and no practice happened. The growth
step is that kind of surgery. So: coach for where the knowledge comes from,
surgery for how it gets in.

## What I take from it

* Growth is free, exact, and empty. Anyone can double a model in a minute.
* Healing recovers the model and then improves it, but at a billion tokens the
  extra capacity is worth 0.006 nats over just training the small model, and probably less than nothing per GPU hour. Practical takeaway: model growth is a bet on much larger training budgets (5B+ tokens) rather than a low-budget shortcut.
* Filling the empty part with copies of what the small model already knows
  did not help, whether a rule or an LLM chose the copies, and the textbook
  way of doing it, exact splitting, adds no usable capacity through the copied units.
* The experiment I would run next is the compute-matched one with a budget of
  5B tokens or more, where the late gain has room to show, and where it would be worth trying noise-broken splits and the negated-gate
  copy (a unit that fires exactly where its donor is silent) as inits.

## What went wrong along the way

For the record, because it changed the design: the first H100 host died at
step 7,852 with the run's checkpoints on its local disk. I had saved only bf16
snapshots, no optimizer state, so nothing was resumable even if the disk had
survived. The second attempt saved the full training state every 1,000 steps
on a network volume, ran the four pilots on the full schedule so the winner
could be resumed rather than restarted, and stopped its own pod when done.
The resumed run reproduced the lost run's curve to the third decimal, which
is the kind of boring result you want.

## Prior work

Net2Net (Chen, Goodfellow, Shlens 2015); Masked Structural Growth (2023);
HyperCloning (Apple, 2024); Stacking Your Transformers (2024); SOLAR 10.7B,
Yi-9B and LLaMA Pro as depth-grown production models. The compute-matched
framing and the warning about symmetric splits are both in the older
literature; I rediscovered them the expensive way.

## Reproduce it

Code: https://github.com/johndo31/upscale_net. grow.py (growth, four init modes), verify.py
(exactness proof), heal.py (training with resumable state), analyze.py and
analyze2.py (reading the small model), plan_llm.py (the LLM-written plan),
pick.py, bench.sh, pilot.sh (the whole thing on one GPU, unattended).
Logs, loss curves and benchmark JSON are in results/. One H100 at $4/h:
pilots $21, long run $32, control $23, about $77 for the numbers above, plus about $22 for the run the dead host ate.
