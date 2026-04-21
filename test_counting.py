"""
Test CountLSTM behavior: multiple instances of the same field in a single text.
Run with: python test_counting.py
"""

import time
from gliner2 import GLiNER2

MODEL = "fastino/gliner2-base-v1"

# Each test has a text and expected number of instances per field
TESTS = [
    {
        "label": "Multiple investments (2 amounts, 2 companies)",
        "text": "Goldman Sachs invested $500M in Stripe and $200M in SpaceX during Q3 2024.",
        "schema": {
            "investment": [
                "amount::str::Investment amount",
                "company::str::Target company",
            ]
        }
    },
    {
        "label": "Multiple acquisitions by different buyers",
        "text": """
        Microsoft acquired GitHub for $7.5B in 2018.
        Salesforce acquired Slack for $27.7B in 2021.
        Adobe acquired Figma for $20B in 2022.
        """,
        "schema": {
            "acquisition": [
                "acquirer::str",
                "target::str",
                "price::str",
                "year::str",
            ]
        }
    },
    {
        "label": "Multiple trades in one report",
        "text": """
        Trade 1: JPMorgan bought 10,000 shares of Apple at $182 on March 5.
        Trade 2: JPMorgan sold 5,000 shares of Tesla at $245 on March 6.
        Trade 3: JPMorgan bought 8,000 shares of Nvidia at $820 on March 7.
        """,
        "schema": {
            "trade": [
                "action::[buy|sell]::str",
                "quantity::str",
                "security::str",
                "price::str",
                "date::str",
            ]
        }
    },
    {
        "label": "Multiple executives with roles",
        "text": """
        Satya Nadella serves as CEO of Microsoft.
        Amy Hood is the CFO of Microsoft.
        Brad Smith is the President of Microsoft.
        """,
        "schema": {
            "executive": [
                "name::str",
                "role::str",
                "company::str",
            ]
        }
    },
    {
        "label": "Multiple financial metrics in earnings",
        "text": """
        Apple reported Q4 2024 revenue of $94.9B, up 6% YoY.
        Net income was $14.7B, down 2% YoY.
        Gross margin came in at 46.2%, up from 45.2% last year.
        EPS was $0.97, beating estimates of $0.94.
        """,
        "schema": {
            "metric": [
                "name::str::Financial metric name",
                "value::str::Reported value",
                "change::str::YoY change",
            ]
        }
    },
]


def run(extractor: GLiNER2):
    for test in TESTS:
        print(f"\n{'='*65}")
        print(f"TEST: {test['label']}")
        print(f"TEXT: {test['text'].strip()}")
        print("-" * 65)

        t0 = time.perf_counter()
        results = extractor.extract_json(test["text"], test["schema"])
        elapsed = (time.perf_counter() - t0) * 1000

        for struct, instances in results.items():
            print(f"\n[{struct.upper()}] — {len(instances)} instance(s) found")
            for i, inst in enumerate(instances, 1):
                non_empty = {k: v for k, v in inst.items() if v}
                print(f"  [{i}] {non_empty}")

        print(f"\n  Inference: {elapsed:.1f} ms")


if __name__ == "__main__":
    print(f"Loading model: {MODEL}")
    extractor = GLiNER2.from_pretrained(MODEL)
    print("Ready.\n")
    run(extractor)
