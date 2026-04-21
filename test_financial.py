"""
Financial query test: combined structured extraction + entity + relation extraction.
Run with: python test_financial.py
"""

import time
from gliner2 import GLiNER2

MODEL = "fastino/gliner2-base-v1"

FINANCIAL_TEXTS = [
    # M&A / deal text
    """
    Microsoft agreed to acquire Activision Blizzard for $68.7 billion on January 18, 2022.
    The deal, valued at $95 per share, was approved by the FTC in October 2023.
    Satya Nadella, CEO of Microsoft, called it a transformative moment for gaming.
    """,

    # Earnings / investment text
    """
    Goldman Sachs reported Q3 2024 net revenue of $12.7 billion, up 7% year-over-year.
    The firm invested $500M in Stripe and holds a 3% stake in SpaceX.
    David Solomon, Chairman and CEO, announced a $2B share buyback programme.
    """,

    # Trade / transaction text
    """
    JPMorgan executed a $150M block trade of Apple Inc. shares on behalf of Vanguard.
    The trade settled on March 10, 2024 at an average price of $182.50 per share.
    Commission charged: $75,000. Status: Settled.
    """,
]


def run_tests(extractor: GLiNER2):
    for i, text in enumerate(FINANCIAL_TEXTS, 1):
        print(f"\n{'='*70}")
        print(f"TEXT {i}:\n{text.strip()}")
        print('='*70)

        schema = (
            extractor.create_schema()
            # --- entities ---
            .entities(["company", "person", "financial instrument", "location", "date", "monetary amount"])

            # --- relations ---
            .relations({
                "acquired":       "Acquisition where one company buys another",
                "invested_in":    "Investment relationship between investor and company",
                "owns_stake_in":  "Ownership stake relationship",
                "executed_for":   "Trade executed by broker on behalf of client",
                "reported_by":    "Financial result reported by company",
                "leads":          "Executive leadership relationship",
            })

            # --- structured deal/transaction info ---
            .structure("deal")
                .field("acquirer",   dtype="str", description="Buying company")
                .field("target",     dtype="str", description="Target or acquired company")
                .field("value",      dtype="str", description="Deal or transaction value")
                .field("price_per_share", dtype="str")
                .field("date",       dtype="str")
                .field("status",     dtype="str", choices=["announced", "pending", "approved", "completed", "blocked"])

            .structure("financial_result")
                .field("company",   dtype="str")
                .field("period",    dtype="str", description="Reporting period e.g. Q3 2024")
                .field("revenue",   dtype="str")
                .field("change",    dtype="str", description="YoY change")
                .field("action",    dtype="str", description="Capital action e.g. buyback, dividend")

            .structure("trade")
                .field("broker",    dtype="str")
                .field("client",    dtype="str")
                .field("security",  dtype="str")
                .field("amount",    dtype="str")
                .field("price",     dtype="str")
                .field("date",      dtype="str")
                .field("commission", dtype="str")
                .field("status",    dtype="str", choices=["pending", "settled", "failed", "cancelled"])
        )

        t0 = time.perf_counter()
        results = extractor.extract(text, schema)
        elapsed = time.perf_counter() - t0
        print(f"\n[TIMING] {elapsed*1000:.1f} ms")

        # --- print entities ---
        print("\n[ENTITIES]")
        entities = results.get("entities", {})
        for etype, vals in entities.items():
            if vals:
                print(f"  {etype}: {vals}")

        # --- print relations ---
        print("\n[RELATIONS]")
        relations = results.get("relation_extraction", {})
        for rtype, pairs in relations.items():
            if pairs:
                print(f"  {rtype}:")
                for pair in pairs:
                    print(f"    {pair[0]}  -->  {pair[1]}")

        # --- print structures ---
        for struct_key in ("deal", "financial_result", "trade"):
            instances = results.get(struct_key, [])
            if instances:
                print(f"\n[{struct_key.upper()}]")
                for inst in instances:
                    non_empty = {k: v for k, v in inst.items() if v}
                    print(f"  {non_empty}")


if __name__ == "__main__":
    print(f"Loading model: {MODEL}")
    t_load = time.perf_counter()
    extractor = GLiNER2.from_pretrained(MODEL)
    print(f"Model loaded in {time.perf_counter() - t_load:.2f}s\n")

    t_total = time.perf_counter()
    run_tests(extractor)
    print(f"\n{'='*70}")
    print(f"Total inference time (all texts): {(time.perf_counter() - t_total)*1000:.1f} ms")
