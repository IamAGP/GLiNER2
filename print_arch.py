"""
Architecture viewer for GLiNER2.
Shows the full module tree, per-layer parameter counts, and a torchinfo summary.

Run with: python3 print_arch.py
"""

import torch
from gliner2 import GLiNER2
from torchinfo import summary

MODEL = "fastino/gliner2-base-v1"

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL} ...")
m = GLiNER2.from_pretrained(MODEL)
m.eval()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw PyTorch module tree
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  1. FULL MODULE TREE  (PyTorch repr)")
print("═" * 70)
print(m)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameter budget per top-level component
# ─────────────────────────────────────────────────────────────────────────────
def param_count(module):
    return sum(p.numel() for p in module.parameters())

components = {
    "encoder (DeBERTa-v3-base)": m.encoder,
    "span_rep (SpanRepLayer)":   m.span_rep,
    "count_embed (CountLSTMv2)": m.count_embed,
    "count_pred (MLP)":          m.count_pred,
    "classifier (MLP)":          m.classifier,
}

total = param_count(m)

print("\n" + "═" * 70)
print("  2. PARAMETER BUDGET PER COMPONENT")
print("═" * 70)
print(f"  {'Component':<35}  {'Params':>12}  {'Share':>7}")
print(f"  {'-'*35}  {'-'*12}  {'-'*7}")
for name, mod in components.items():
    n = param_count(mod)
    print(f"  {name:<35}  {n:>12,}  {n/total*100:>6.2f}%")
print(f"  {'─'*35}  {'─'*12}  {'─'*7}")
print(f"  {'TOTAL':<35}  {total:>12,}  100.00%")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CountLSTMv2 internals
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  3. CountLSTMv2 INTERNALS")
print("═" * 70)
ce = m.count_embed
D  = ce.hidden_size

sub = {
    "pos_embedding  (Embedding max_count×D)":  ce.pos_embedding,
    "gru            (CompileSafeGRU D→D)":     ce.gru,
    "transformer    (DownscaledTransformer)":  ce.transformer,
    "  in_projector   (Linear D→128)":         ce.transformer.in_projector,
    "  transformer    (2-layer, 4-head, 128d)": ce.transformer.transformer,
    "  out_projector  (MLP 256→D)":            ce.transformer.out_projector,
}
print(f"  hidden_size = {D},  max_count = {ce.max_count}")
print()
for name, mod in sub.items():
    print(f"  {name:<45}  {param_count(mod):>10,} params")


# ─────────────────────────────────────────────────────────────────────────────
# 4. torchinfo summary — encoder only (full model summary is huge)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  4. torchinfo SUMMARY — DeBERTa encoder  (batch=1, seq=64)")
print("═" * 70)

tok = m.processor.tokenizer
dummy_text   = "Goldman Sachs invested in Stripe ."
enc          = tok(dummy_text, return_tensors="pt")
input_ids    = enc["input_ids"]
attn_mask    = enc["attention_mask"]

summary(
    m.encoder,
    input_data={"input_ids": input_ids, "attention_mask": attn_mask},
    depth=3,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    verbose=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. torchinfo summary — CountLSTMv2
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  5. torchinfo SUMMARY — CountLSTMv2  (M=2 fields, L=3 steps)")
print("═" * 70)

M = 2   # number of fields
L = 3   # predicted count
pc_emb = torch.randn(M, D)

summary(
    m.count_embed,
    input_data=(pc_emb, L),
    depth=4,
    col_names=["input_size", "output_size", "num_params"],
    row_settings=["var_names"],
    verbose=1,
)
