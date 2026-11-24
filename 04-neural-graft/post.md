# Neural Graft: Plugging an Exact Skill Into a Frozen LLM Where Fine-Tuning Fails

_Fine-tuning taught a 1.5B model addition on numbers of up to 8 digits. At 16 digits it could only guess whether a long carry arrives. A small circuit with written weights, plugged into the same frozen model, gets it right every time at 64 digits, and the model can use its result._

_Qwen2.5-1.5B-Instruct, frozen. Three tasks on real tokens, up to five seeds, tuned baselines, one Colab L4 GPU._

_By Niklas, [drniklas.com](https://drniklas.com). I also host the [Venture Grade](https://venturegrade.drniklas.com) podcast._

## The short answer

Language models are unreliable at exact, rule-based work such as long addition or bracket matching. Fine-tuning does not fix this: the model learns the lengths it was shown and guesses beyond them. This experiment shows a different way to give a model such a skill. I wrote the skill as a small circuit whose weights are correct by construction, and plugged it into a frozen Qwen2.5-1.5B-Instruct. Only a small output map was trained, on inputs of length 4 to 8.

Three things make this useful:

* **The skill is exact, far beyond the training lengths.** The grafted model is right in 100% of cases at up to 16 times the training length. On the hardest tests, tuned LoRA and a trained circuit of the same shape are at or near chance.
* **The model itself is not changed.** No weight of Qwen is touched. The trained part is 3,072 to 15,360 numbers, against 18.5 million for LoRA, and with a small gate ordinary text is unaffected.
* **The result arrives inside the model.** Unlike a calculator called as a tool, the circuit runs within the model's forward pass. Injected at layer 9 of 28, its result is used by Qwen's own layers about as well as a result told in words.

The findings in detail:

1. **Exact at up to 16 times the training length.** On 64-digit additions where one carry travels the whole number, the written circuit scores 100%, tuned LoRA 52.3% and a trained circuit of the same shape 48.3%.
2. **Whole sums in normal notation.** For `Add: 4821+3957=` with 64-digit numbers, the grafted model writes every digit of the sum correctly in 100% of cases. A trained circuit with the same inputs scores 0%. LoRA, without the position input, also scores 0%.
3. **Training does not find the general solution.** Tuned LoRA learns the short inputs and falls towards chance as they grow.
4. **A learned gate keeps ordinary text untouched.** My first test missed damage on text containing "=". A larger test found it, and a gate of 1,537 numbers removed it.
5. **Qwen's own layers use the result.** With the circuit's answer injected at layer 9 of 28, Qwen answers unseen fact questions that depend on it 94.2% to 99.9% right. Told the answer in words, it scores 88.1% to 97.9%.
6. **Circuits written by a model graft too.** Of 17 circuits a language model wrote in an earlier experiment, 14 reach 100% once wired in.
7. **Qwen can read the input for the circuit, partly.** With brackets spelled as words and read from Qwen's hidden states, the graft is exact to length 32 and fails at 128.

The main limit is plain: the circuit does the computing, and Qwen does not learn arithmetic. The others are listed under "Limits".

## Where this fits

This experiment tests what a circuit with known-correct weights is worth inside a real LLM. ["On Computable Growth"](https://x.com/Niklas_Sikorra/status/2118303794387628252) found that adding empty capacity to a model bought almost nothing. The weight-writer and transformer-writer experiments in the same repo had a model write network weights against an exact checker.

## The setup: a circuit beside a frozen model

The language model stays frozen. Beside it runs a circuit ("organ" in the code): a causal Transformer with two or three layers, whose weights are written directly, not trained. The addition circuit has 1,017 non-zero weights, the bracket circuit 15, the ascending-digits circuit 42.

The wiring ("socket" in the code) has two parts:

* **Read-in.** A linear map from Qwen's token embeddings to the circuit's input channels: one per digit or bracket, one for "=", one for everything else. It is fitted in closed form by ridge regression on the embedding table, with no task labels. A clamp then sets values below 0.3 to exactly 0 and values above 0.85 to exactly 1.
* **Write-out.** A linear map from the circuit's output into Qwen, at first just before the output layer. It starts at zero, so the grafted model begins identical to the base model. It is the only trained part, 3,072 to 15,360 numbers, and it acts only at an "=" token.

[Figure 1: how the circuit is wired into the frozen model]

Three tasks use Qwen's real tokens:

* **Addition in columns.** `Add LSB pairs: 97,81,42,35=` means 3489 + 5217, lowest digit first, in pairs. The answer is the sum digit in the top column.
* **Balanced brackets.** `Balanced brackets: ( ( ) ) =`, answer Yes or No.
* **Ascending digits.** `Ascending digits: 0123456789=`, answer Yes or No.

Every arm trains on lengths 4 to 8 only ("in range") and is tested up to length 64, or 128 for brackets, on the same 512 cases per length. The arms are the base model; LoRA of rank 16 (18,464,768 trained parameters, three seeds); a **trained circuit**, with the same architecture and wiring but random initial weights (five seeds); and the **written circuit** (five seeds). Both baselines are tuned on in-range validation data.

## 1. Exact at up to 16 times the training length

The written circuit scored 100% on every seed, every length and every task. Neither trained arm came close on the hardest tests.

| hardest test, mean [min, max] over seeds | base | LoRA | trained circuit | written circuit |
|---|---|---|---|---|
| addition, 64-digit carry chains | 10.4% | 52.3% [51.8, 53.1] | 48.3% [44.7, 51.4] | **100%** |
| brackets, hard set, length 128 | 51.6% | 53.6% [49.6, 55.7] | 50.1% [43.6, 58.8] | **100%** |
| ascending digits, length 64 | 50.2% | 84.5% [54.9, 100] | 62.9% [50.0, 77.0] | **100%** |

[Figure 2: results table, hardest test per task, mean over seeds]

The hard tests remove shortcuts. In a carry chain every column between the lowest and the top one sums to 9, so one carry must travel the whole number. On random 64-digit numbers LoRA scores 99.8% and the trained circuit 99.2%, but that test is weak: only one column in ten sums to 9, so looking at the top two columns is right about 95% of the time. On carry chains both trained arms sit near 50% at 16, 32 and 64 digits alike.

In the hard bracket set every wrong string has equal numbers of "(" and ")", starts with "(" and ends with ")". Only the order is wrong. In the ordinary bracket test, 95% to 99% of wrong strings can be spotted by counting or from the first token. On the hard set LoRA scores 100% in range, 76.1% at length 32 and 53.6% at 128, which is chance.

[Figure 3: accuracy by length, carry chains and hard brackets]

With a random English sentence in front of the prompt, the written circuit stays at 100%.

## 2. Whole sums in normal notation

The grafted model also adds in ordinary notation and writes out the whole sum. The prompt is `Add: 4821+3957=`, and the model generates the answer digit by digit, each chosen from the full vocabulary. With a second written circuit (1,712 non-zero weights) the whole sum was exactly right in 100% of cases at 16, 32 and 64 digits and on 64-digit carry chains, on all three seeds.

This circuit gets one extra input: each digit's position within its number, counted by fixed code, as in earlier work on addition in Transformers (McLeish et al., 2024). The fair comparison is therefore the trained circuit, which gets the same input. It reached 29.8% in range, 0.8% at 16 digits and 0% at 32 and 64. LoRA does not get that input. Trained on whole sums, it scored 99.0% in range, between 2.3% and 22.5% at 16 digits, and 0% at 32 and 64.

## 3. Training does not find the general solution

Tuned training learns the short inputs and loses the task as inputs grow. LoRA scores at least 99.9% in range on all three tasks of section 1. At the longest length it averages 60.4% on ordinary brackets and 84.5% on ascending digits, where its seeds range from 54.9% to 100%. So LoRA can find a solution that holds at length on that task, but not reliably. On brackets, even a learning rate chosen on the length-128 test itself gets LoRA only to 79.9%.

The trained circuit is the cleanest comparison: same architecture, wiring and data as the written circuit, different weights. A solution exists in that architecture, because the written circuit is one. Gradient descent did not find it on any seed.

## 4. A learned gate keeps ordinary text untouched

The graft can be made to leave ordinary text alone, but my first test of that was too weak. It measured next-token loss on six English sentences and found no change. None of them contains "=", and the write-out acts only at "=", so the test could not fail.

A larger test did fail. It used 200 passages of Wikipedia prose, 200 chunks of Python standard-library code with 958 "=" positions, and 120 maths lines with 277. Prose stayed unchanged (at most +0.004). At "=" the circuits did real damage: in code, the bracket circuit raised the loss from 1.84 to 13.35. So "no damage" had only been true for text without "=".

The fix is a gate: a logistic probe on Qwen's own last hidden state, 1,537 numbers, trained to open on task prompts and stay shut at "=" in code and maths. It was tested on code and maths it had not seen.

| change in loss at "=" positions | addition | brackets | ascending digits |
|---|---|---|---|
| written circuit, no gate, code | 0.00 | +11.51 | +7.40 |
| written circuit, no gate, maths | +1.84 | +0.65 | +10.74 |
| written circuit, with gate, code | 0.00 | 0.00 | +0.02 |
| written circuit, with gate, maths | 0.00 | 0.00 | 0.00 |
| LoRA, code | +0.25 | +0.05 | +0.05 |
| LoRA, maths | +0.39 | +0.83 | +0.19 |

[Figure 4: effect on text containing "=", without and with the gate]

With the gate, task accuracy stays at 100% at every length. LoRA also raises loss on prose, by 0.03 to 0.12.

## 5. Qwen's own layers use the result

Injected in the middle of the model, the circuit's result is used by Qwen's frozen upper layers together with their own knowledge. In the wiring so far the circuit writes just before the output layer, so Qwen's layers take no part. Here the write-out goes to the output of layer 9 of 28.

The prompt, in chat format, says: "If the answer is Yes, name the capital of France. If the answer is No, name the capital of Japan." A bracket string follows, ending in "= ?". The bracket circuit's one-bit result is added to Qwen's internal state at the "=" token. Only the write-out is trained, on bracket lengths 4 to 8 and one set of questions. The test uses other questions, whose facts never appear in training. To answer, Qwen has to carry the bit through its upper layers and combine it with a fact it knows.

| unseen fact questions, by bracket length | 4 to 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| plain Qwen | 56.7% | 47.5% | 50.8% | 49.2% | 49.9% |
| Qwen told the result in words | 98.6% | 91.3% | 95.8% | 88.1% | 97.9% |
| circuit injected at layer 9 | 99.3% | 99.9% | 98.9% | 99.8% | 94.2% |
| LoRA on the same data | 100% | 87.1% | 73.8% | 61.1% | 68.5% |

[Figure 5: Qwen uses the result, accuracy on unseen questions by bracket length]

The layer matters. At layer 5 the scores range from 72.3% to 94.7%, at layer 19 from 68.6% to 83.4%. The old wiring stays at plain Qwen's level, because it can only act when "=" is the last token.

## 6. Circuits written by a model graft too

Most circuits written by a language model work when wired in. In the transformer-writer experiment a 3B coder model, trained in a loop with an exact checker, wrote 17 circuits that pass at length 64 on their own. They solve simple tasks over the tokens 0 to 3, such as "all tokens are 0" or "the pair 1,2 occurs". Wired into Qwen with 6,144 trained numbers each, 14 of the 17 reach 100% at length 64.

Three do not. One stays at base level (72.1%), one reaches 79.5%, and one is exact up to length 32 and drops to 70.7% at 64. LoRA, run on four of these tasks, reaches 100% on three and 75.4% at length 64 on the fourth, where the model-written circuit scores 100%.

[Figure 6: circuits written by a model, grafted]

## 7. Qwen can read the input for the circuit, partly

The read-in can also sit on Qwen's hidden states, but that is fragile so far. With brackets spelled as the words "open" and "close", I fitted the read-in at layer 5, again without task labels. The graft scored 100% up to length 32, 99.0% at 64 and 52.9% at 128. The same read-in does not transfer to "left" and "right": accuracy there is at chance, between 45.7% and 61.5%.

## Limits

* **The circuit does the computing.** Qwen does not learn arithmetic or bracket matching. Qwen's part is deciding when the circuit speaks (the gate) and using its result (section 5).
* **The real-token circuits were written for these tasks.** Model-written circuits exist only for simple synthetic-token tasks.
* **Normal-notation addition needs help.** It relies on the fixed position input and on operands of equal length, padded with leading zeros.
* **The test in section 5 is small.** It has five unseen fact questions. The brackets sit right before the answer, so the bit travels through the upper layers and a few chat tokens, not across a long text. Of eight prompt formats, plain Qwen could use a result told in words in only one, selected on the training questions. A sixth unseen question, a translation, is weak for every arm at long lengths. The mid-layer write-out has no learned gate yet: loss on lines containing "=" rises from 2.18 to about 2.30.
* **Scope.** One model size and three main tasks. Except for whole sums, accuracy is scored over the allowed answer tokens only.

## Related work

Writing Transformer weights directly is established: RASP (Weiss, Goldberg and Yahav, "Thinking Like Transformers") describes Transformer computations as programs, and Tracr (Lindner et al.) compiles them into weights. The position input follows McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings". I have not searched systematically for work that wires a written circuit into a pretrained LLM, so I claim no novelty.

## Method in brief

* **Read-in.** An earlier run on synthetic tokens showed that the closed-form fit and the clamp are both needed: a read-in trained from a random start did nothing, and without the clamp one of four circuits failed.
* **Training.** Write-out: 350 steps, or 500 in sections 2 and 5. LoRA: 1,000 steps, learning rate from 0.0001 and 0.0003. Trained circuit: 3,000 steps, learning rate from 0.001, 0.003 and 0.01. Sections 2 and 5 use fixed baseline settings and three seeds; sections 4 and 6 use one seed.

## Reproduce it

The code is in [github.com/johndo31/upscale_net](https://github.com/johndo31/upscale_net) under `04-neural-graft/`. `robust_graft_exp.py` runs sections 1, 3, 4 and 6. `natural_graft_exp.py` runs sections 2, 5 and 7. Each script writes its results to one JSON file.
