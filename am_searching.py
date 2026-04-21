"""
Intuition-building experiments for GLiNER2 / CountLSTMv2.
Each experiment is a function — run individually or all at once.

Run all:  python3 am_searching.py
Run one:  python3 -c "import am_searching; am_searching.exp_identical_amounts()"
"""

import torch
from gliner2 import GLiNER2

MODEL = "fastino/gliner2-base-v1"


def load():
    print(f"Loading {MODEL} ...")
    return GLiNER2.from_pretrained(MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 — Identical amounts, two fields (amount + company)
# Question: does the model handle two identical values correctly?
# Finding: YES — returns 2 instances both with $500M. Count prediction reads
#          sentence structure ("and"), not field values. Deduplication does NOT
#          collapse identical spans.
# ─────────────────────────────────────────────────────────────────────────────
def exp_identical_amounts_two_fields(m):
    print("\n" + "="*60)
    print("EXP 1 — Identical amounts, two fields (amount + company)")
    print("="*60)
    r = m.extract_json(
        "Goldman invested $500M in Stripe and $500M in SpaceX.",
        {"investment": ["amount::str", "company::str"]}
    )
    for i, inst in enumerate(r["investment"], 1):
        print(f"  [{i}] {inst}")
    print("Expected: 2 instances, both amount=$500M, companies differ")


# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 — Identical amounts, amount field only
# Question: without company to anchor instances, does count drop to 1?
# Finding: NO — still returns 2 instances. Count reads sentence structure.
# ─────────────────────────────────────────────────────────────────────────────
def exp_identical_amounts_one_field(m):
    print("\n" + "="*60)
    print("EXP 2 — Identical amounts, amount field only")
    print("="*60)
    r = m.extract_json(
        "Goldman invested $500M in Stripe and $500M in SpaceX.",
        {"investment": ["amount::str"]}
    )
    for i, inst in enumerate(r["investment"], 1):
        print(f"  [{i}] {inst}")
    print("Expected: still 2 instances (count from sentence structure not values)")


# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 — Zero out positional embeddings
# Question: if GRU gets same input every step, do queries collapse?
# Hypothesis: model may only return 1 instance or duplicate the same span.
# Finding: TBD
# ─────────────────────────────────────────────────────────────────────────────
def exp_zero_positional_embeddings(m):
    print("\n" + "="*60)
    print("EXP 3 — Zero out positional embeddings in CountLSTMv2")
    print("="*60)

    # Save original weights so we can restore after
    original = m.count_embed.pos_embedding.weight.data.clone()

    with torch.no_grad():
        m.count_embed.pos_embedding.weight.zero_()

    r = m.extract_json(
        "Goldman invested $500M in Stripe and $200M in SpaceX during Q3 2024.",
        {"investment": ["amount::str", "company::str"]}
    )
    for i, inst in enumerate(r["investment"], 1):
        print(f"  [{i}] {inst}")
    print("Expected: queries collapse → likely 1 instance or same span twice")

    # Restore
    with torch.no_grad():
        m.count_embed.pos_embedding.weight.copy_(original)
    print("  (positional embeddings restored)")


# ─────────────────────────────────────────────────────────────────────────────
# Add new experiments below as we discover more
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    m = load()
    exp_identical_amounts_two_fields(m)
    exp_identical_amounts_one_field(m)
    exp_zero_positional_embeddings(m)
