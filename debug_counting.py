"""
Debug view into CountLSTMv2 internals during multi-instance extraction.

Shows per-step GRU hidden states, cross-step query similarity, and top
candidate spans per (instance, field) — letting you see exactly what the
model "sees" at each extraction step.

Run with: python debug_counting.py
"""

import torch
import torch.nn.functional as F
from gliner2 import GLiNER2
from typing import Dict, List, Any, Optional

MODEL = "fastino/gliner2-base-v1"

# ─────────────────────────────────────────────────────────────────────────────
# Global capture stores — filled during forward passes
# ─────────────────────────────────────────────────────────────────────────────

_current_schema: List[Optional[str]] = [None]   # set by override before super()
_schema_debug: Dict[str, Dict] = {}              # keyed by schema_name
_gru_steps: Dict[str, torch.Tensor] = {}         # schema_name -> (L, M, D)
_transformer_out: Dict[str, torch.Tensor] = {}   # schema_name -> (L, M, D)


# ─────────────────────────────────────────────────────────────────────────────
# Hook callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _hook_gru(module, input, output):
    """Captures GRU hidden states at each count step: (L, M, D)."""
    name = _current_schema[0]
    if name:
        _gru_steps[name] = output.detach().cpu()


def _hook_transformer(module, input, output):
    """Captures DownscaledTransformer output: (L, M, D)."""
    name = _current_schema[0]
    if name:
        _transformer_out[name] = output.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Subclass: override _extract_span_result to capture intermediates
# ─────────────────────────────────────────────────────────────────────────────

class DebugGLiNER2(GLiNER2):
    """Same as GLiNER2 but captures CountLSTM internals during inference."""

    def _extract_span_result(
        self,
        results: Dict,
        schema_name: str,
        task_type: str,
        embs: torch.Tensor,
        span_info: Dict,
        schema_tokens: List[str],
        text_tokens: List[str],
        text_len: int,
        original_text: str,
        start_mapping: List[int],
        end_mapping: List[int],
        threshold: float,
        metadata: Dict,
        cls_fields: Dict,
        include_confidence: bool,
        include_spans: bool,
    ):
        # ── field names ────────────────────────────────────────────────────
        field_names = [
            schema_tokens[j + 1]
            for j in range(len(schema_tokens) - 1)
            if schema_tokens[j] in ("[E]", "[C]", "[R]")
        ]
        if not field_names:
            results[schema_name] = [] if schema_name == "entities" else {}
            return

        # ── count prediction ───────────────────────────────────────────────
        count_logits = self.count_pred(embs[0].unsqueeze(0))   # (1, 20)
        pred_count   = int(count_logits.argmax(dim=1).item())

        _schema_debug[schema_name] = {
            "field_names":    field_names,
            "text_tokens":    text_tokens,
            "text_len":       text_len,
            "start_mapping":  start_mapping,
            "end_mapping":    end_mapping,
            "count_logits":   count_logits.detach().cpu(),
            "pred_count":     pred_count,
            "p_emb_norm":     embs[0].norm().item(),
            "field_embs":     embs[1:].detach().cpu() if len(embs) > 1 else None,
        }

        if pred_count <= 0 or span_info is None:
            if schema_name == "entities":
                results[schema_name] = []
            elif task_type == "relations":
                results[schema_name] = []
            else:
                results[schema_name] = {}
            return

        # ── count embeddings + span scores ─────────────────────────────────
        _current_schema[0] = schema_name          # hooks read this
        struct_proj = self.count_embed(embs[1:], pred_count)  # (L, M, D)
        _current_schema[0] = None

        span_scores = torch.sigmoid(
            torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj)
        )
        _schema_debug[schema_name]["struct_proj"]  = struct_proj.detach().cpu()
        _schema_debug[schema_name]["span_scores"]  = span_scores.detach().cpu()

        # ── delegate to original type-specific extractors ──────────────────
        if schema_name == "entities":
            results[schema_name] = self._extract_entities(
                field_names, span_scores, text_len, text_tokens,
                original_text, start_mapping, end_mapping,
                threshold, metadata, include_confidence, include_spans,
            )
        elif task_type == "relations":
            results[schema_name] = self._extract_relations(
                schema_name, field_names, span_scores, pred_count,
                text_len, text_tokens, original_text, start_mapping,
                end_mapping, threshold, metadata, include_confidence,
                include_spans,
            )
        else:
            results[schema_name] = self._extract_structures(
                schema_name, field_names, span_scores, pred_count,
                text_len, text_tokens, original_text, start_mapping,
                end_mapping, threshold, metadata, cls_fields,
                include_confidence, include_spans,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for the report
# ─────────────────────────────────────────────────────────────────────────────

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().unsqueeze(0),
                               b.flatten().unsqueeze(0)).item()


def _top_spans(score_2d: torch.Tensor, text_tokens: List[str], n: int = 3):
    """score_2d: (text_len, max_width) → list of (span_text, score)."""
    flat   = score_2d.reshape(-1)
    topk   = torch.topk(flat, min(n, flat.numel()))
    max_w  = score_2d.shape[1]
    spans  = []
    for idx, sc in zip(topk.indices.tolist(), topk.values.tolist()):
        start = idx // max_w
        width = idx % max_w
        end   = start + width + 1
        if end <= len(text_tokens):
            spans.append((" ".join(text_tokens[start:end]), sc))
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_debug_report(schema_name: str):
    d          = _schema_debug.get(schema_name)
    gru        = _gru_steps.get(schema_name)       # (L, M, D)
    trans_out  = _transformer_out.get(schema_name)  # (L, M, D)

    if d is None:
        print(f"  [no debug data for {schema_name!r}]")
        return

    fields      = d["field_names"]
    pred_count  = d["pred_count"]
    M           = len(fields)

    W = 70
    print(f"\n{'─'*W}")
    print(f"  DEBUG › {schema_name.upper()}   (predicted count = {pred_count})")
    print(f"{'─'*W}")

    # ── 1. Count prediction ────────────────────────────────────────────────
    logits   = d["count_logits"][0]            # (20,)
    probs    = torch.softmax(logits, dim=0)
    topk5    = torch.topk(probs, 5)
    p_norm   = d["p_emb_norm"]

    print(f"\n  [1] COUNT PREDICTION")
    print(f"      [P] embedding norm : {p_norm:.2f}")
    print(f"      Top-5 count predictions:")
    for cnt, prob in zip(topk5.indices.tolist(), topk5.values.tolist()):
        marker = "  ◄" if cnt == pred_count else ""
        print(f"        count={cnt:2d}  p={prob:.3f}{marker}")

    # ── 2. Field embeddings ────────────────────────────────────────────────
    if d["field_embs"] is not None:
        print(f"\n  [2] FIELD EMBEDDINGS  (M={M} fields, input to GRU)")
        for i, fname in enumerate(fields):
            norm = d["field_embs"][i].norm().item()
            print(f"      field[{i}] {fname!r:20s}  norm={norm:.2f}")

    # ── 3. GRU step analysis ───────────────────────────────────────────────
    if gru is not None:
        L = gru.shape[0]
        print(f"\n  [3] GRU STEPS  ({L} steps × {M} fields × {gru.shape[2]} hidden)")

        for l in range(L):
            print(f"\n      Step {l} (instance slot {l+1}):")
            for k, fname in enumerate(fields):
                h = gru[l, k]
                norm = h.norm().item()
                # cosine drift from previous step
                if l > 0:
                    drift = 1.0 - _cos(gru[l-1, k], h)
                    print(f"        {fname!r:20s}  norm={norm:.2f}  drift_from_prev={drift:.3f}")
                else:
                    print(f"        {fname!r:20s}  norm={norm:.2f}")

        if L >= 2:
            print(f"\n      Cross-step cosine similarity  (low = queries are DISTINCT):")
            for k, fname in enumerate(fields):
                sims = []
                for l in range(L - 1):
                    sims.append(_cos(gru[l, k], gru[l+1, k]))
                sim_str = "  ".join(f"step{i}↔{i+1}={s:.3f}" for i, s in enumerate(sims))
                print(f"        {fname!r:20s}  {sim_str}")

    # ── 4. Transformer output (CountLSTMv2) ────────────────────────────────
    if trans_out is not None and gru is not None:
        print(f"\n  [4] TRANSFORMER OUTPUT  (cross-field attention applied)")
        print(f"      Norm delta vs GRU output (transformer effect):")
        for l in range(trans_out.shape[0]):
            for k, fname in enumerate(fields):
                gru_norm   = gru[l, k].norm().item()
                trans_norm = trans_out[l, k].norm().item()
                delta_pct  = (trans_norm - gru_norm) / (gru_norm + 1e-8) * 100
                sign = "+" if delta_pct >= 0 else ""
                print(f"        step={l} {fname!r:20s}  "
                      f"gru={gru_norm:.2f}  trans={trans_norm:.2f}  "
                      f"Δ={sign}{delta_pct:.1f}%")

    # ── 5. Span scores ─────────────────────────────────────────────────────
    span_scores = d.get("span_scores")
    text_tokens = d["text_tokens"]

    if span_scores is not None:
        L      = span_scores.shape[0]
        n_flds = span_scores.shape[1]
        tl     = d["text_len"]
        print(f"\n  [5] TOP CANDIDATE SPANS PER (instance, field)")
        print(f"      span_scores shape: {tuple(span_scores.shape)}")
        for l in range(L):
            print(f"\n      Instance slot {l+1}:")
            for k in range(min(n_flds, len(fields))):
                fname  = fields[k]
                sc_2d  = span_scores[l, k, :tl, :]   # (text_len, max_width)
                tops   = _top_spans(sc_2d, text_tokens, n=3)
                top_str = "  |  ".join(f"{txt!r} {sc:.3f}" for txt, sc in tops)
                if not top_str:
                    top_str = "(no spans)"
                print(f"        {fname!r:20s}  {top_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    {
        "label": "Multiple investments (2 instances)",
        "text":  "Goldman Sachs invested $500M in Stripe and $200M in SpaceX during Q3 2024.",
        "schema": {"investment": ["amount::str::Investment amount", "company::str::Target company"]},
    },
    {
        "label": "Multiple executives (3 instances, same company repeated)",
        "text":  ("Satya Nadella serves as CEO of Microsoft.  "
                  "Amy Hood is the CFO of Microsoft.  "
                  "Brad Smith is the President of Microsoft."),
        "schema": {"executive": ["name::str", "role::str", "company::str"]},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading model: {MODEL}")
    extractor = DebugGLiNER2.from_pretrained(MODEL)

    # Register hooks (only on CountLSTMv2 sub-modules)
    _hooks = [
        extractor.count_embed.gru.register_forward_hook(_hook_gru),
        extractor.count_embed.transformer.register_forward_hook(_hook_transformer),
    ]
    print("Hooks registered on: count_embed.gru, count_embed.transformer\n")

    try:
        for test in TESTS:
            # Clear capture stores for each test
            _schema_debug.clear()
            _gru_steps.clear()
            _transformer_out.clear()

            print(f"\n{'='*70}")
            print(f"TEST: {test['label']}")
            print(f"TEXT: {test['text']}")
            print(f"{'='*70}")

            results = extractor.extract_json(test["text"], test["schema"])

            # Print extraction result
            for struct, instances in results.items():
                print(f"\n  RESULT [{struct.upper()}] — {len(instances)} instance(s):")
                for i, inst in enumerate(instances, 1):
                    print(f"    [{i}] { {k:v for k,v in inst.items() if v} }")

            # Print debug report for each schema
            for schema_name in _schema_debug:
                print_debug_report(schema_name)

    finally:
        for h in _hooks:
            h.remove()
        print("\nHooks removed.")
