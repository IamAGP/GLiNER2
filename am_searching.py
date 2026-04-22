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
# EXP 4 — Span score heatmap
# Question: which exact spans score high for each field?
# Shows: rows = start position (token), cols = width (0=single token, 1=2 tokens...)
# Reading: a bright cell at row 2, col 1 means span "$500M" scores high for that field
# ─────────────────────────────────────────────────────────────────────────────
def exp_span_heatmap(m):
    print("\n" + "="*60)
    print("EXP 4 — Span score heatmap (start position × width)")
    print("="*60)

    from debug_counting import DebugGLiNER2, _schema_debug, _gru_steps, _transformer_out
    import torch

    # reload as DebugGLiNER2 to capture span_scores
    dm = DebugGLiNER2.from_pretrained("fastino/gliner2-base-v1")

    text   = "Goldman invested $500M in Stripe and $200M in SpaceX during Q3 2024."
    schema = {"investment": ["amount::str::Investment amount", "company::str::Target company"]}

    _schema_debug.clear()
    dm.extract_json(text, schema)

    d           = _schema_debug.get("investment")
    tokens      = d["text_tokens"]
    span_scores = d["span_scores"]   # (L, M, text_len, max_width)
    fields      = d["field_names"]
    text_len    = d["text_len"]
    L           = span_scores.shape[0]

    print(f"\n  Tokens: {tokens}")
    print(f"  Text length: {text_len} tokens,  max_width: {span_scores.shape[3]}")

    for inst in range(L):
        print(f"\n  ── Instance slot {inst+1} ──")
        for k, fname in enumerate(fields):
            sc = span_scores[inst, k, :text_len, :]   # (text_len, max_width)
            print(f"\n    Field: '{fname}'")

            # Header row — widths
            header = f"    {'token':>12}  " + "  ".join(f"w={w}" for w in range(sc.shape[1]))
            print(header)
            print("    " + "-" * (len(header) - 4))

            for t in range(text_len):
                tok  = tokens[t] if t < len(tokens) else "?"
                row  = "  ".join(
                    f"{sc[t,w].item():>4.2f}" if sc[t,w].item() > 0.01 else "  . "
                    for w in range(sc.shape[1])
                )
                print(f"    {tok:>12}  {row}")


# ─────────────────────────────────────────────────────────────────────────────
# EXP 5 — Step through every single model step live
# Intervenes at each step and prints what the model actually has at that point.
# No hardcoding — the model does all the work, we just peek.
# ─────────────────────────────────────────────────────────────────────────────
def exp_step_by_step(m):
    print("\n" + "="*60)
    print("EXP 5 — Every single step the model takes")
    print("="*60)

    import torch
    import torch.nn.functional as F
    from debug_counting import DebugGLiNER2, _schema_debug, _gru_steps

    TEXT   = "Goldman invested $500M in Stripe and $200M in SpaceX."
    SCHEMA = {"investment": ["amount::str", "company::str"]}

    print(f"\n  Input text  : {TEXT}")
    print(f"  Input schema: {SCHEMA}")

    # ── STEP 1: Tokenisation ──────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 1 — Tokenisation")
    print("─"*60)
    tok            = m.processor.tokenizer
    enc            = tok(TEXT, return_tensors="pt")
    subword_tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"  Subword tokens (DeBERTa sees): {subword_tokens}")
    print(f"  Total: {len(subword_tokens)} subword tokens")

    # ── STEP 2: Full input sequence ───────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 2 — Full sequence DeBERTa receives")
    print("─"*60)
    print("  [P] investment [E] amount [E] company [SEP] <text tokens>")
    print("  Schema tokens prepended to text — all processed as one sequence")

    # ── STEP 3-9: Run through DebugGLiNER2 which captures all intermediates
    dm = DebugGLiNER2.from_pretrained(MODEL)

    # Hook encoder output
    encoder_out = {}
    def hook_enc(module, inp, out):
        encoder_out["hs"] = out.last_hidden_state.detach()
    h_enc = dm.encoder.register_forward_hook(hook_enc)

    # Hook GRU steps
    _schema_debug.clear()
    _gru_steps.clear()
    dm.extract_json(TEXT, SCHEMA)
    h_enc.remove()

    d          = _schema_debug["investment"]
    field_embs = d["field_embs"]       # (M, 768)
    fields     = d["field_names"]
    pred_count = d["pred_count"]
    count_logits = d["count_logits"]
    span_scores  = d["span_scores"]    # (L, M, text_len, max_w)
    struct_proj  = d["struct_proj"]    # (L, M, 768)
    text_tokens  = d["text_tokens"]
    text_len     = d["text_len"]
    gru_out      = _gru_steps.get("investment")  # (L, M, 768)

    # ── STEP 3 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 3 — DeBERTa encoder output")
    print("─"*60)
    if "hs" in encoder_out:
        hs = encoder_out["hs"]
        print(f"  Shape: {tuple(hs.shape)}  ({hs.shape[1]} tokens × {hs.shape[2]} dims)")
        print(f"  Every token now has a contextual 768-dim vector")
        print(f"  [P]   token norm: {hs[0,0].norm().item():.2f}")
        print(f"  [SEP] token norm: {hs[0,-1].norm().item():.2f}")

    # ── STEP 4 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 4 — Extract [P] and field embedding vectors")
    print("─"*60)
    print(f"  [P] token norm : {d['p_emb_norm']:.2f}  (goes to count_pred)")
    for i, fname in enumerate(fields):
        print(f"  field[{i}] '{fname}' norm: {field_embs[i].norm().item():.2f}  (goes to CountLSTMv2)")

    # ── STEP 5 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 5 — Count prediction")
    print("─"*60)
    probs = torch.softmax(count_logits[0], dim=0)
    top5  = torch.topk(probs, 5)
    print(f"  [P] vector (768d) → count_pred MLP → 20 scores")
    for cnt, prob in zip(top5.indices.tolist(), top5.values.tolist()):
        marker = "  ◄" if cnt == pred_count else ""
        print(f"    count={cnt}  p={prob:.4f}{marker}")
    print(f"  → L = {pred_count}  (GRU unrolls {pred_count} times)")

    # ── STEP 6 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 6 — Span representation")
    print("─"*60)
    max_w = span_scores.shape[3]
    print(f"  Text: {text_len} tokens × {max_w} widths = {text_len*max_w} total spans")
    print(f"  Each span → one 768-dim vector  (boundary tokens combined)")
    print(f"  Text tokens: {text_tokens}")

    # ── STEP 7 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 7 — CountLSTMv2 unrolls L times")
    print("─"*60)
    print(f"  Input : field_embs {tuple(field_embs.shape)}")
    print(f"  Output: struct_proj {tuple(struct_proj.shape)}  (L × M × 768)")
    if gru_out is not None:
        for l in range(pred_count):
            for k, fname in enumerate(fields):
                print(f"    step={l+1} field='{fname}'  "
                      f"GRU norm={gru_out[l,k].norm().item():.2f}  "
                      f"final norm={struct_proj[l,k].norm().item():.2f}")
    if pred_count >= 2:
        sim = F.cosine_similarity(
            struct_proj[0].flatten().unsqueeze(0),
            struct_proj[1].flatten().unsqueeze(0)
        ).item()
        print(f"  Cross-step similarity: {sim:.3f}  (lower = more distinct queries)")

    # ── STEP 8 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 8 — Span scoring  (query · span → score)")
    print("─"*60)
    print(f"  span_scores shape: {tuple(span_scores.shape)}")
    print(f"  = (L={pred_count} instances × M={len(fields)} fields × {text_len} positions × {max_w} widths)")
    print(f"  Total dot products computed: {span_scores.numel()}")

    # ── STEP 9 ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 9 — Pick winning span per (instance, field)")
    print("─"*60)
    for l in range(pred_count):
        print(f"\n  Instance {l+1}:")
        for k, fname in enumerate(fields):
            sc    = span_scores[l, k, :text_len, :]
            flat  = sc.reshape(-1)
            best  = int(flat.argmax().item())
            start = best // max_w
            width = best % max_w
            end   = start + width + 1
            score = flat[best].item()
            span_txt = " ".join(text_tokens[start:end]) if end <= len(text_tokens) else "?"
            print(f"    '{fname}': '{span_txt}'  (pos={start}, width={width}, score={score:.4f})")

    # ── FINAL ─────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("FINAL — extract_json output")
    print("─"*60)
    result = m.extract_json(TEXT, SCHEMA)
    for i, inst in enumerate(result["investment"], 1):
        print(f"  [{i}] {inst}")


# ─────────────────────────────────────────────────────────────────────────────
# Add new experiments below as we discover more
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    m = load()
    exp_identical_amounts_two_fields(m)
    exp_identical_amounts_one_field(m)
    exp_zero_positional_embeddings(m)
    exp_span_heatmap(m)
    exp_step_by_step(m)
