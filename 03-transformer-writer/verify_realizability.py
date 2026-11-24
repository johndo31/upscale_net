"""Verify that TinyTransformer + sparse weight text format solves Tier 1 & Tier 2
algorithmic tasks at 100.0% accuracy from L=3..8 all the way to OOD L=64.

Also defines the 4 worked-example circuits (WORKED_EXAMPLES) used in prompts.
"""

from __future__ import annotations

from typing import Dict

from target_tf import parse_sparse_weights, serialize_sparse_weights
from tasks import TASKS_BY_NAME, evaluate_model, make_eval_suite


# ---------- 4 Worked Examples (shown in prompt; never in train/heldout) ----------

WORKED_EXAMPLES: Dict[str, str] = {
    "last_tok": """PLAN: Use L0.H0 with alibi=-10 to attend to previous token (at EOS, that is x_L), copy channels 0..3 into 7..10, and read out 7..10.
WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
OUT.W[7,0] = 1
OUT.W[8,1] = 1
OUT.W[9,2] = 1
OUT.W[10,3] = 1""",

    "has_0": """PLAN: Use L0.H0 with alibi=0; query on constant ch 6, key strongly on token 0 (ch 0), value reads ch 0 into ch 7. If token 0 appears anywhere, ch 7 ~ 1 at EOS, else 0.
WEIGHTS:
L0.H0.Q[6,0] = 1
L0.H0.K[0,0] = 40
L0.H0.V[0,0] = 1
L0.H0.O[0,7] = 1
OUT.W[7,1] = 2
OUT.b[0] = 1""",

    "majority_1": """PLAN: Use L0.H0 with alibi=0 to average +1 for token 1 (ch 1) and -1 for token 0 (ch 0) into ch 7. At EOS, ch 7 > 0 iff strictly more 1s than 0s.
WEIGHTS:
L0.H0.V[1,0] = 1
L0.H0.V[0,0] = -1
L0.H0.O[0,7] = 1
OUT.W[7,1] = 200
OUT.b[0] = 0.5""",

    "bigram_0_1": """PLAN: L0.H0 (alibi=-10) copies previous token 0 (ch 0) into ch 7. L0.M neuron 0 fires iff prev==0 (ch 7) AND cur==1 (ch 1), writing flag to ch 11. L1.H0 attends strongly to ch 11 and copies to ch 12. Read out ch 12 at EOS.
WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[0,0] = 1
L0.H0.O[0,7] = 1
L0.M.W1[7,0] = 1
L0.M.W1[1,0] = 1
L0.M.b1[0] = -1.5
L0.M.W2[0,11] = 2
L1.H0.Q[6,0] = 1
L1.H0.K[11,0] = 40
L1.H0.V[11,0] = 1
L1.H0.O[0,12] = 1
OUT.W[12,1] = 2
OUT.b[0] = 1""",
}


# ---------- Reference circuits proving realizability on harder Tier 1 & Tier 2 tasks ----------
# (Used ONLY by verify_realizability.py — NEVER shown to the writer model!)

REFERENCE_PROOF_CIRCUITS: Dict[str, str] = {
    "first_tok": """WEIGHTS:
L0.H0.alibi = 10
L0.H0.Q[6,0] = 1
L0.H0.K[4,0] = -200
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
OUT.W[7,0] = 1
OUT.W[8,1] = 1
OUT.W[9,2] = 1
OUT.W[10,3] = 1""",

    "max_tok": """WEIGHTS:
L0.H0.Q[6,0] = 1
L0.H0.K[4,0] = -200
L0.H0.K[1,0] = 20
L0.H0.K[2,0] = 40
L0.H0.K[3,0] = 60
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
OUT.W[7,0] = 1
OUT.W[8,1] = 1
OUT.W[9,2] = 1
OUT.W[10,3] = 1""",

    "min_tok": """WEIGHTS:
L0.H0.Q[6,0] = 1
L0.H0.K[4,0] = -200
L0.H0.K[1,0] = -20
L0.H0.K[2,0] = -40
L0.H0.K[3,0] = -60
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
OUT.W[7,0] = 1
OUT.W[8,1] = 1
OUT.W[9,2] = 1
OUT.W[10,3] = 1""",

    "has_0_and_1": """WEIGHTS:
L0.H0.Q[6,0] = 1
L0.H0.K[0,0] = 40
L0.H0.V[0,0] = 1
L0.H0.O[0,7] = 1
L0.H1.Q[6,0] = 1
L0.H1.K[1,0] = 40
L0.H1.V[1,0] = 1
L0.H1.O[0,8] = 1
OUT.W[7,1] = 1
OUT.W[8,1] = 1
OUT.b[0] = 1.5""",

    "all_same": """WEIGHTS:
L0.H0.Q[6,0] = 1
L0.H0.K[4,0] = -200
L0.H0.K[1,0] = 20
L0.H0.K[2,0] = 40
L0.H0.K[3,0] = 60
L0.H0.V[1,0] = 1
L0.H0.V[2,0] = 2
L0.H0.V[3,0] = 3
L0.H0.O[0,7] = 1
L0.H1.Q[6,0] = 1
L0.H1.K[4,0] = -200
L0.H1.K[1,0] = -20
L0.H1.K[2,0] = -40
L0.H1.K[3,0] = -60
L0.H1.V[1,0] = 1
L0.H1.V[2,0] = 2
L0.H1.V[3,0] = 3
L0.H1.O[0,8] = 1
OUT.W[7,0] = 10
OUT.W[8,0] = -10
OUT.b[1] = 5""",

    "palindrome_ends": """WEIGHTS:
L0.H0.alibi = 10
L0.H0.Q[6,0] = 1
L0.H0.K[4,0] = -200
L0.H0.V[1,0] = 1
L0.H0.V[2,0] = 2
L0.H0.V[3,0] = 3
L0.H0.O[0,7] = 1
L0.H1.alibi = -10
L0.H1.V[1,0] = 1
L0.H1.V[2,0] = 2
L0.H1.V[3,0] = 3
L0.H1.O[0,8] = 1
L0.M.W1[7,0] = 1
L0.M.W1[8,0] = -1
L0.M.W1[7,1] = -1
L0.M.W1[8,1] = 1
L0.M.W2[0,9] = 1
L0.M.W2[1,9] = 1
OUT.W[9,0] = 10
OUT.b[1] = 5""",

    "second_last": """WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
L1.H0.alibi = -10
L1.H0.V[7,0] = 1
L1.H0.V[8,1] = 1
L1.H0.V[9,2] = 1
L1.H0.V[10,3] = 1
L1.H0.O[0,11] = 1
L1.H0.O[1,12] = 1
L1.H0.O[2,13] = 1
L1.H0.O[3,14] = 1
OUT.W[11,0] = 1
OUT.W[12,1] = 1
OUT.W[13,2] = 1
OUT.W[14,3] = 1""",

    "two_same_adj": """WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
L0.M.W1[0,0] = 1
L0.M.W1[7,0] = 1
L0.M.b1[0] = -1.5
L0.M.W1[1,1] = 1
L0.M.W1[8,1] = 1
L0.M.b1[1] = -1.5
L0.M.W1[2,2] = 1
L0.M.W1[9,2] = 1
L0.M.b1[2] = -1.5
L0.M.W1[3,3] = 1
L0.M.W1[10,3] = 1
L0.M.b1[3] = -1.5
L0.M.W2[0,11] = 2
L0.M.W2[1,11] = 2
L0.M.W2[2,11] = 2
L0.M.W2[3,11] = 2
L1.H0.Q[6,0] = 1
L1.H0.K[11,0] = 40
L1.H0.V[11,0] = 1
L1.H0.O[0,12] = 1
OUT.W[12,1] = 2
OUT.b[0] = 1""",

    "sorted_asc": """WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[1,0] = 1
L0.H0.V[2,0] = 2
L0.H0.V[3,0] = 3
L0.H0.V[4,1] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.M.W1[7,0] = 1
L0.M.W1[1,0] = -1
L0.M.W1[2,0] = -2
L0.M.W1[3,0] = -3
L0.M.W1[8,0] = -10
L0.M.W1[5,0] = -10
L0.M.W1[4,0] = -10
L0.M.b1[0] = -0.5
L0.M.W2[0,11] = 2
L1.H0.Q[6,0] = 1
L1.H0.K[11,0] = 40
L1.H0.V[11,0] = 1
L1.H0.O[0,12] = 1
OUT.W[12,0] = 2
OUT.b[1] = 1""",

    "dyck1": """WEIGHTS:
L0.H0.V[0,0] = 1
L0.H0.V[1,0] = -1
L0.H0.O[0,7] = 1
L0.M.W1[7,0] = -200
L0.M.W2[0,8] = 1
L0.M.W1[7,1] = 200
L0.M.W2[1,9] = 1
L1.H0.Q[6,0] = 1
L1.H0.K[8,0] = 20
L1.H0.V[8,0] = 1
L1.H0.O[0,10] = 1
OUT.b[1] = 1
OUT.W[8,1] = -2
OUT.W[9,1] = -2
OUT.W[10,1] = -2""",

    "induction": """WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[0,0] = 1
L0.H0.V[1,1] = 1
L0.H0.V[2,2] = 1
L0.H0.V[3,3] = 1
L0.H0.O[0,7] = 1
L0.H0.O[1,8] = 1
L0.H0.O[2,9] = 1
L0.H0.O[3,10] = 1
L1.H0.alibi = -4
L1.H0.Q[7,0] = 40
L1.H0.Q[8,1] = 40
L1.H0.Q[9,2] = 40
L1.H0.Q[10,3] = 40
L1.H0.K[7,0] = 40
L1.H0.K[8,1] = 40
L1.H0.K[9,2] = 40
L1.H0.K[10,3] = 40
L1.H0.V[0,0] = 1
L1.H0.V[1,1] = 1
L1.H0.V[2,2] = 1
L1.H0.V[3,3] = 1
L1.H0.O[0,11] = 1
L1.H0.O[1,12] = 1
L1.H0.O[2,13] = 1
L1.H0.O[3,14] = 1
OUT.W[11,0] = 1
OUT.W[12,1] = 1
OUT.W[13,2] = 1
OUT.W[14,3] = 1""",
}


def main() -> None:
    all_circuits = {**WORKED_EXAMPLES, **REFERENCE_PROOF_CIRCUITS}
    print(f"{'task':18s} {'split':8s} {'nnz':>4s} {'verif':>7s} {'id':>7s} {'L16':>7s} {'L32':>7s} {'L64':>7s}")
    print("-" * 68)
    failures = 0
    for name, spec in all_circuits.items():
        task = TASKS_BY_NAME[name]
        res = parse_sparse_weights(spec)
        assert res.model is not None, f"{name}: parse error: {res.error}"
        # Check round-trip serialization
        rt = parse_sparse_weights(serialize_sparse_weights(res.model))
        assert rt.model is not None and rt.n_entries == res.n_entries, f"{name}: round-trip mismatch"
        suite = make_eval_suite(task)
        accs = {k: evaluate_model(res.model, task, b) for k, b in suite.items()}
        print(
            f"{name:18s} {task.split:8s} {res.n_entries:4d} "
            f"{accs['verif']*100:6.1f}% {accs['id']*100:6.1f}% "
            f"{accs['L16']*100:6.1f}% {accs['L32']*100:6.1f}% {accs['L64']*100:6.1f}%"
        )
        if any(v < 1.0 for v in accs.values()):
            failures += 1
    print("-" * 68)
    print(f"Verified {len(all_circuits)} circuits; failures={failures}")
    assert failures == 0, "Some reference circuits failed!"


if __name__ == "__main__":
    main()
