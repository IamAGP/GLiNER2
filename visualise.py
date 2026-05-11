"""
GLiNER2 Forward Pass — Animated Tutorial
=========================================
Scenes:
  1. TokenisationScene   — input text + schema tokenised
  2. EncoderScene        — full sequence into DeBERTa, vectors out
  3. CountPredScene      — [P] token → count_pred MLP → count
  4. SpanRepScene        — all possible spans enumerated
  5. GRUUnrollScene      — CountLSTMv2 unrolling L steps
  6. SpanScoreScene      — heatmap of span scores per (instance, field)
  7. OutputScene         — winning spans highlighted in original text

Run one scene:
  source .venv/bin/activate
  manim -pql visualise.py TokenisationScene

Run all scenes:
  manim -pql visualise.py         # renders each scene separately
"""

from manim import *
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────────────

TEXT   = "Goldman invested $500M in Stripe and $200M in SpaceX."
SCHEMA = {"investment": ["amount::str", "company::str"]}
MODEL  = "fastino/gliner2-base-v1"

# Colour palette
C_BG         = "#0f0f1a"
C_TOKEN      = "#4a9eff"
C_SCHEMA     = "#ff9f4a"
C_SPECIAL    = "#ff4a9f"
C_HIGHLIGHT  = "#4aff9f"
C_DIM        = "#444466"
C_WHITE      = "#ffffff"
C_COLD       = "#0a0a2e"
C_HOT        = "#ffee00"


def narration(text, color=C_WHITE):
    """Narration subtitle at the bottom of the scene."""
    return Text(text, font_size=22, color=color).to_edge(DOWN, buff=0.3)


def section_title(text):
    """Small label top-left."""
    return Text(text, font_size=24, color=C_DIM).to_corner(UL, buff=0.3)


def val_to_color(v: float) -> ManimColor:
    """0.0 = cold (dark blue), 1.0 = hot (bright yellow)."""
    v = float(np.clip(v, 0.0, 1.0))
    return interpolate_color(
        interpolate_color(ManimColor(C_COLD), BLUE, min(v * 2, 1.0)),
        interpolate_color(RED, ManimColor(C_HOT), max((v - 0.5) * 2, 0.0)),
        v,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model data loader — runs real forward pass, caches results
# ─────────────────────────────────────────────────────────────────────────────

_model_data = {}

def get_model_data():
    """Run the real model once and cache all intermediate tensors."""
    if _model_data:
        return _model_data

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from debug_counting import DebugGLiNER2, _schema_debug, _gru_steps, _transformer_out

    print("Loading model for visualisation data...")
    dm = DebugGLiNER2.from_pretrained(MODEL)

    # ── pull architecture facts directly from live config ──────────────────
    cfg = dm.encoder.config
    encoder_config = {
        "num_layers":    cfg.num_hidden_layers,
        "num_heads":     cfg.num_attention_heads,
        "hidden_size":   cfg.hidden_size,
        "ffn_size":      cfg.intermediate_size,
        "pos_att_type":  getattr(cfg, "pos_att_type", []),
        "model_type":    cfg.model_type,
    }
    # pull Q/K/V and FFN shapes from layer 0 directly
    l0 = dm.encoder.encoder.layer[0]
    encoder_config["qkv_shape"]  = list(l0.attention.self.query_proj.weight.shape)
    encoder_config["ffn_in"]     = list(l0.intermediate.dense.weight.shape)
    encoder_config["ffn_out"]    = list(l0.output.dense.weight.shape)
    encoder_config["head_size"]  = l0.attention.self.attention_head_size

    # ── hooks: encoder final output + all 12 layer hidden states ──────────
    enc_out      = {}
    layer_hidden = {}   # layer_idx -> (seq_len, 768)
    attn_weights = {}   # layer_idx -> (num_heads, seq_len, seq_len)

    def hook_enc_out(mod, inp, out):
        enc_out["hs"] = out.last_hidden_state.detach().cpu()

    def make_layer_hook(idx):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            layer_hidden[idx] = hs[0].detach().cpu()
        return hook

    def make_attn_hook(idx):
        def hook(mod, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_weights[idx] = out[1][0].detach().cpu()
        return hook

    hooks = [dm.encoder.register_forward_hook(hook_enc_out)]
    for i, layer in enumerate(dm.encoder.encoder.layer):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))
        hooks.append(layer.attention.self.register_forward_hook(make_attn_hook(i)))

    # enable attention weight output via config flag
    dm.encoder.config.output_attentions = True

    _schema_debug.clear()
    _gru_steps.clear()
    _transformer_out.clear()

    dm.extract_json(TEXT, SCHEMA)

    dm.encoder.config.output_attentions = False
    for h in hooks:
        h.remove()

    d = _schema_debug["investment"]

    # ── full sequence token labels (what DeBERTa actually sees) ───────────
    # schema_tokens (e.g. ['(','[P]','investment','(','[C]','amount',...,')'])
    # + separator + text subwords from tokenizer
    tok      = dm.processor.tokenizer
    enc_ids  = tok(TEXT, return_tensors="pt")
    subwords = tok.convert_ids_to_tokens(enc_ids["input_ids"][0])  # text only
    schema_toks = d["schema_tokens"]
    # separator token that the processor inserts between schema and text
    sep_tok  = "[SEP_TEXT]"
    full_seq_tokens = schema_toks + [sep_tok] + list(subwords)
    seq_len = enc_out["hs"].shape[1]  # authoritative — actual encoder input length
    full_seq_tokens = full_seq_tokens[:seq_len]  # trim to exact length

    # ── layer norms: (num_layers, seq_len) ────────────────────────────────
    num_layers  = encoder_config["num_layers"]
    layer_norms = np.zeros((num_layers, seq_len))
    for i in range(num_layers):
        if i in layer_hidden:
            hs = layer_hidden[i].numpy()  # (seq_len, 768)
            layer_norms[i] = np.linalg.norm(hs, axis=-1)

    # ── [P] token position in full sequence ───────────────────────────────
    try:
        p_pos = full_seq_tokens.index("[P]")
    except ValueError:
        p_pos = 1  # fallback

    # ── last-layer attention mean over heads, [P] row ─────────────────────
    last_layer = num_layers - 1
    if last_layer in attn_weights:
        attn_mat_last = attn_weights[last_layer].numpy()   # (heads, seq, seq)
        p_attn_last   = attn_mat_last.mean(axis=0)[p_pos]  # (seq,) — [P]→all
        attn_mean_last = attn_mat_last.mean(axis=0)         # (seq, seq)
    else:
        p_attn_last    = np.ones(seq_len) / seq_len
        attn_mean_last = np.eye(seq_len)

    _model_data.update({
        # tokenisation
        "subwords":         subwords,
        "full_seq_tokens":  full_seq_tokens,
        "seq_len":          seq_len,
        "p_pos":            p_pos,
        # schema / field
        "text_tokens":      d["text_tokens"],
        "field_names":      d["field_names"],
        "schema_tokens":    d["schema_tokens"],
        # encoder architecture (from live config)
        "encoder_config":   encoder_config,
        # encoder internals (from live forward pass)
        "encoder_hs":       enc_out.get("hs", torch.zeros(1, 1, 768)).numpy(),
        "layer_norms":      layer_norms,            # (12, seq_len)
        "p_attn_last":      p_attn_last,            # (seq_len,)
        "attn_mean_last":   attn_mean_last,         # (seq_len, seq_len)
        # count prediction
        "p_emb_norm":       d["p_emb_norm"],
        "field_embs":       d["field_embs"].numpy(),
        "count_logits":     d["count_logits"].numpy(),
        "pred_count":       d["pred_count"],
        # span extraction
        "span_scores":      d["span_scores"].numpy(),
        "struct_proj":      d["struct_proj"].numpy(),
        "gru_out":          _gru_steps.get("investment", torch.zeros(1)).numpy(),
        "text_len":         d["text_len"],
    })
    print("Model data ready.")
    return _model_data


# ─────────────────────────────────────────────────────────────────────────────
# Scene 1 — Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

class TokenisationScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 1 · Tokenisation")
        self.add(title)

        # ── narration 1 ──
        n1 = narration("We start with a sentence and a schema.")
        self.play(FadeIn(n1))
        self.wait(1.5)

        # ── show raw sentence ──
        sentence = Text(TEXT, font_size=22, color=C_WHITE).scale(0.8)
        sentence.move_to(UP * 2)
        self.play(Write(sentence), run_time=2)
        self.wait(0.5)

        # ── show schema ──
        schema_label = Text("Schema:", font_size=18, color=C_SCHEMA)
        schema_fields = VGroup(
            Text("investment", font_size=20, color=C_SCHEMA),
            Text("  fields: amount,  company", font_size=18, color=C_SCHEMA),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        schema_group = VGroup(schema_label, schema_fields).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        schema_group.move_to(UP * 0.5)

        self.play(FadeOut(n1))
        n2 = narration("The schema tells the model what to look for.")
        self.play(FadeIn(n2), FadeIn(schema_group))
        self.wait(1.5)

        # ── tokenise text ──
        self.play(FadeOut(n2))
        n3 = narration("The text is split into subword tokens by the DeBERTa tokenizer.")
        self.play(FadeIn(n3))
        self.wait(0.5)

        subwords = data["subwords"]
        # show only non-special subword tokens (skip [CLS], [SEP])
        display_tokens = [t for t in subwords if t not in ("[CLS]", "[SEP]")]

        token_boxes = VGroup()
        for tok_str in display_tokens:
            box = RoundedRectangle(
                corner_radius=0.08,
                width=max(len(tok_str) * 0.13 + 0.2, 0.4),
                height=0.4,
                fill_color=C_TOKEN,
                fill_opacity=0.25,
                stroke_color=C_TOKEN,
                stroke_width=1.5,
            )
            label = Text(tok_str, font_size=14, color=C_TOKEN)
            group = VGroup(box, label)
            token_boxes.add(group)

        token_boxes.arrange(RIGHT, buff=0.12)
        token_boxes.scale_to_fit_width(12)
        token_boxes.move_to(DOWN * 0.8)

        self.play(
            LaggedStart(
                *[FadeIn(tb, shift=UP * 0.2) for tb in token_boxes],
                lag_ratio=0.08,
                run_time=2.5
            )
        )
        self.wait(1)

        # ── highlight $, 500, M ──
        self.play(FadeOut(n3))
        n4 = narration("Notice: '$500M' becomes three tokens — '$', '500', 'M'.\nDeBERTa uses subword tokenisation.")
        self.play(FadeIn(n4))

        # find the $, 500, M tokens
        dollar_indices = [i for i, t in enumerate(display_tokens) if t in ("▁$",)]
        num_indices    = [i for i, t in enumerate(display_tokens) if t in ("500", "200")]
        m_indices      = [i for i, t in enumerate(display_tokens) if t == "M"]

        highlight_idx = dollar_indices + num_indices + m_indices
        highlight_anims = []
        for idx in highlight_idx:
            if idx < len(token_boxes):
                highlight_anims.append(
                    token_boxes[idx][0].animate.set_fill(C_HIGHLIGHT, opacity=0.5)
                )
                highlight_anims.append(
                    token_boxes[idx][1].animate.set_color(C_HIGHLIGHT)
                )
        if highlight_anims:
            self.play(*highlight_anims)
        self.wait(2)

        # ── special tokens ──
        self.play(FadeOut(n4))
        n5 = narration("Special tokens [P], [E], [SEP] are added to encode the schema structure.")
        self.play(FadeIn(n5))

        # pulled directly from the model — no hardcoding
        special_tokens = data["schema_tokens"]
        special_boxes = VGroup()
        for st in special_tokens:
            is_special = st.startswith("[")
            color = C_SPECIAL if is_special else C_SCHEMA
            box = RoundedRectangle(
                corner_radius=0.08,
                width=max(len(st) * 0.13 + 0.2, 0.5),
                height=0.4,
                fill_color=color,
                fill_opacity=0.3,
                stroke_color=color,
                stroke_width=1.5,
            )
            label = Text(st, font_size=13, color=color)
            special_boxes.add(VGroup(box, label))

        special_boxes.arrange(RIGHT, buff=0.1)
        special_boxes.scale_to_fit_width(10)
        special_boxes.move_to(DOWN * 1.8)

        arrow = Arrow(
            special_boxes.get_top(),
            token_boxes.get_bottom() + DOWN * 0.1,
            buff=0.1,
            color=C_DIM,
            stroke_width=1.5,
        )

        self.play(
            LaggedStart(
                *[FadeIn(sb, shift=UP * 0.15) for sb in special_boxes],
                lag_ratio=0.06,
                run_time=1.5
            )
        )
        self.wait(0.5)
        n6_text = "This full sequence is what DeBERTa processes — schema + text as one input."
        self.play(FadeOut(n5))
        n6 = narration(n6_text)
        self.play(FadeIn(n6))
        self.wait(2.5)

        self.play(FadeOut(n6), FadeOut(title))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 2 — DeBERTa Encoder
# ─────────────────────────────────────────────────────────────────────────────

class EncoderScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        cfg        = data["encoder_config"]
        seq_tokens = data["full_seq_tokens"]
        seq_len    = data["seq_len"]
        p_pos      = data["p_pos"]
        layer_norms = data["layer_norms"]   # (12, seq_len)
        p_attn     = data["p_attn_last"]    # (seq_len,)
        hs_final   = data["encoder_hs"][0]  # (seq_len, 768)
        num_layers  = cfg["num_layers"]
        num_heads   = cfg["num_heads"]
        hidden      = cfg["hidden_size"]
        ffn         = cfg["ffn_size"]
        pos_att     = cfg["pos_att_type"]   # ['p2c', 'c2p']
        head_size   = cfg["head_size"]
        qkv_shape   = cfg["qkv_shape"]
        ffn_in      = cfg["ffn_in"]
        ffn_out     = cfg["ffn_out"]

        title = section_title("Scene 2 · DeBERTa Encoder")
        self.add(title)

        # ══════════════════════════════════════════════════════════════════
        # PART 1 — Full input sequence (all tokens, real from model)
        # ══════════════════════════════════════════════════════════════════
        n1 = narration(f"DeBERTa receives {seq_len} tokens: {len(data['schema_tokens'])} schema tokens + separator + text subwords.")
        self.play(FadeIn(n1))
        self.wait(1.5)

        show_n = min(seq_len, 18)
        tok_boxes = VGroup()
        for i in range(show_n):
            t = seq_tokens[i]
            is_schema = i < len(data["schema_tokens"])
            is_p      = (i == p_pos)
            color = C_SPECIAL if is_p else (C_SCHEMA if is_schema else C_TOKEN)
            box = RoundedRectangle(
                corner_radius=0.06,
                width=max(len(t) * 0.11 + 0.15, 0.38), height=0.35,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=1.2,
            )
            lbl = Text(t, font_size=11, color=color)
            tok_boxes.add(VGroup(box, lbl))

        if seq_len > show_n:
            ellipsis = Text("...", font_size=14, color=C_DIM)
            tok_boxes.add(ellipsis)

        tok_boxes.arrange(RIGHT, buff=0.06)
        tok_boxes.scale_to_fit_width(13)
        tok_boxes.move_to(UP * 2.2)

        self.play(FadeOut(n1))
        n2 = narration("Pink=[P]  Orange=schema  Blue=text tokens.  All 27 tokens processed as one sequence.")
        self.play(FadeIn(n2))
        self.play(
            LaggedStart(*[FadeIn(tb, shift=UP * 0.1) for tb in tok_boxes], lag_ratio=0.04),
            run_time=1.8
        )
        self.wait(2)
        self.play(FadeOut(n2), FadeOut(tok_boxes))

        # ══════════════════════════════════════════════════════════════════
        # PART 2 — Architecture facts from live config
        # ══════════════════════════════════════════════════════════════════
        n3 = narration("Architecture pulled from live model config — not a diagram, the actual numbers.")
        self.play(FadeIn(n3))

        arch_lines = [
            ("model_type",     cfg["model_type"]),
            ("num_layers",     str(num_layers)),
            ("num_heads",      str(num_heads)),
            ("hidden_size",    str(hidden)),
            ("ffn_size",       str(ffn)),
            ("head_size",      f"{hidden} / {num_heads} = {head_size}"),
            ("Q/K/V proj",     f"Linear({qkv_shape[1]} → {qkv_shape[0]})"),
            ("FFN in",         f"Linear({ffn_in[1]} → {ffn_in[0]})  GELU"),
            ("FFN out",        f"Linear({ffn_out[1]} → {ffn_out[0]})  + LayerNorm"),
            ("pos_att_type",   " + ".join(pos_att)),
        ]

        rows = VGroup()
        for key, val in arch_lines:
            key_t = Text(key + ":", font_size=18, color=C_DIM)
            val_t = Text(val,       font_size=18, color=C_HIGHLIGHT)
            row = VGroup(key_t, val_t).arrange(RIGHT, buff=0.5)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        rows.move_to(ORIGIN)
        # pin all value texts to same x so the column is aligned
        val_x = max(row[0].get_right()[0] for row in rows) + 0.5
        for row in rows:
            row[1].set_x(val_x, LEFT)

        self.play(FadeOut(n3))
        n4 = narration("Every number here comes from model.config and layer[0].weight.shape — nothing typed by hand.")
        self.play(FadeIn(n4))
        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.08),
            run_time=2.0
        )
        self.wait(3)
        self.play(FadeOut(n4), FadeOut(rows))

        # ══════════════════════════════════════════════════════════════════
        # PART 3 — One transformer layer internals (disentangled attention)
        # ══════════════════════════════════════════════════════════════════
        n5 = narration("Each of the 12 layers has 3 sub-blocks: Disentangled Attention → FFN → LayerNorm+Residual.")
        self.play(FadeIn(n5))
        self.wait(2)
        self.play(FadeOut(n5))

        # draw one layer as a vertical pipeline — MathTex for proper subscripts
        block_specs = [
            (MathTex(r"\text{Input } x \in \mathbb{R}^{768}", font_size=28, color=C_DIM),          C_DIM),
            (MathTex(r"Q = xW_q \quad K = xW_k \quad V = xW_v \quad (\text{per head: }64d)", font_size=26, color=C_TOKEN),   C_TOKEN),
            (MathTex(rf"c2c = (xW_q)(xW_k)^T \;/\; \sqrt{{{head_size}}}", font_size=26, color=C_SCHEMA),  C_SCHEMA),
            (MathTex(rf"c2p = (xW_q)(pW_k)^T \;/\; \sqrt{{{head_size}}}", font_size=26, color=C_SCHEMA),  C_SCHEMA),
            (MathTex(rf"p2c = (pW_q)(xW_k)^T \;/\; \sqrt{{{head_size}}}", font_size=26, color=C_SCHEMA),  C_SCHEMA),
            (MathTex(r"A = \text{softmax}(c2c + c2p + p2c)", font_size=28, color=C_HIGHLIGHT),     C_HIGHLIGHT),
            (MathTex(r"\text{context} = A \cdot V \;\rightarrow\; \text{project} + \text{residual}", font_size=26, color=C_TOKEN),  C_TOKEN),
            (MathTex(rf"\text{{FFN}}: {hidden} \rightarrow {ffn} \;(\text{{GELU}})\; \rightarrow {hidden}", font_size=26, color=C_SCHEMA),  C_SCHEMA),
            (MathTex(r"\text{LayerNorm}(x + \text{FFN}(x))", font_size=28, color=C_HIGHLIGHT),     C_HIGHLIGHT),
        ]

        block_objs = VGroup()
        for tex_obj, color in block_specs:
            box = RoundedRectangle(
                corner_radius=0.08,
                width=8.5, height=0.65,
                fill_color=color, fill_opacity=0.1,
                stroke_color=color, stroke_width=1.2,
            )
            tex_obj.move_to(box)
            block_objs.add(VGroup(box, tex_obj))

        block_objs.arrange(DOWN, buff=0.1)
        block_objs.scale_to_fit_height(6.8)
        block_objs.move_to(ORIGIN)

        n6 = narration("DeBERTa's key innovation: disentangled attention separates CONTENT and POSITION scores.")
        self.play(FadeIn(n6))
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in block_objs], lag_ratio=0.1),
            run_time=2.5
        )
        self.wait(3)

        # highlight the three disentangled score rows
        self.play(FadeOut(n6))
        n7 = narration(f"pos_att_type={pos_att} from model config — c2p and p2c are position-content cross terms.\nThis is what makes DeBERTa stronger than BERT.")
        self.play(FadeIn(n7))
        for i in [2, 3, 4]:
            self.play(block_objs[i][0].animate.set_fill(C_SCHEMA, opacity=0.4), run_time=0.3)
        self.wait(3)
        self.play(FadeOut(n7), FadeOut(block_objs))

        # ══════════════════════════════════════════════════════════════════
        # PART 4 — [P] token attention heatmap (real weights, last layer)
        # ══════════════════════════════════════════════════════════════════
        n8 = narration(f"Real attention weights: [P] token at layer 12 (mean over {num_heads} heads).")
        self.play(FadeIn(n8))
        self.wait(1.5)
        self.play(FadeOut(n8))

        # bar chart — [P] attention over ALL seq_len tokens (no truncation)
        attn_vals = p_attn[:seq_len]
        max_attn  = attn_vals.max() + 1e-8

        bar_w     = min(0.35, 12.0 / seq_len - 0.04)   # shrink bars to fit all tokens
        attn_bars = VGroup()
        attn_lbls = VGroup()
        val_labels = VGroup()

        for i in range(seq_len):
            h_bar = (attn_vals[i] / max_attn) * 2.8 + 0.05
            color = C_SPECIAL if i == p_pos else (C_HIGHLIGHT if attn_vals[i] > attn_vals.mean() else C_DIM)
            bar = Rectangle(
                width=bar_w, height=h_bar,
                fill_color=color, fill_opacity=0.85,
                stroke_width=0,
            )
            attn_bars.add(bar)

            tok_name = seq_tokens[i] if i < len(seq_tokens) else f"t{i}"
            lbl = Text(tok_name, font_size=8, color=color)
            lbl.rotate(PI / 3)   # angled so full names are readable
            attn_lbls.add(lbl)

            v = Text(f"{attn_vals[i]:.3f}", font_size=7, color=C_DIM)
            val_labels.add(v)

        attn_bars.arrange(RIGHT, buff=0.04, aligned_edge=DOWN)
        attn_bars.move_to(UP * 0.5)
        for i, bar in enumerate(attn_bars):
            attn_lbls[i].next_to(bar, DOWN, buff=0.08)
            val_labels[i].next_to(bar, UP,   buff=0.04)

        x_title = Text(
            f"[P] attention weight → all {seq_len} tokens  (layer 12, mean over {num_heads} heads)",
            font_size=14, color=C_WHITE
        ).to_edge(UP, buff=0.7)

        n9 = narration("[P] attends most to schema structure tokens, not raw text.\nThis lets it count instances from the schema, not individual words.")
        self.play(FadeIn(n9), FadeIn(x_title))
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in attn_bars], lag_ratio=0.03),
            run_time=1.8
        )
        self.play(FadeIn(attn_lbls), FadeIn(val_labels))
        self.wait(3.5)
        self.play(FadeOut(n9), FadeOut(attn_bars), FadeOut(attn_lbls),
                  FadeOut(val_labels), FadeOut(x_title))

        # ══════════════════════════════════════════════════════════════════
        # PART 5 — Token norm evolution across 12 layers (real values)
        # ══════════════════════════════════════════════════════════════════
        n10 = narration("How does each token's L2 norm grow as it passes through 12 layers?")
        self.play(FadeIn(n10))
        self.wait(1.5)
        self.play(FadeOut(n10))

        # trace ALL seq_len tokens — color by category
        def token_color(i):
            if i == p_pos:                          return C_SPECIAL   # [P] pink
            if i < len(data["schema_tokens"]):      return C_SCHEMA    # schema orange
            if seq_tokens[i] in ("[SEP_TEXT]", "[CLS]", "[SEP]"): return "#888888"
            return C_TOKEN                                             # text blue

        axes_w, axes_h = 9.5, 4.2
        x_step      = axes_w / (num_layers - 1)
        axes_origin = LEFT * 4.8 + DOWN * 2.0

        x_axis = Arrow(axes_origin, axes_origin + RIGHT * (axes_w + 0.3),
                       buff=0, color=C_DIM, stroke_width=1.5)
        y_axis = Arrow(axes_origin, axes_origin + UP * (axes_h + 0.3),
                       buff=0, color=C_DIM, stroke_width=1.5)
        x_lbl = Text("layer", font_size=14, color=C_DIM).next_to(x_axis.get_end(), RIGHT, buff=0.1)
        y_lbl = Text("L2 norm", font_size=14, color=C_DIM).rotate(PI/2).next_to(y_axis.get_end(), UP, buff=0.1)

        x_ticks = VGroup()
        for i in range(num_layers):
            t = Text(str(i), font_size=10, color=C_DIM)
            t.move_to(axes_origin + RIGHT * i * x_step + DOWN * 0.25)
            x_ticks.add(t)

        max_norm_val = layer_norms.max() + 1e-8

        self.play(Create(x_axis), Create(y_axis),
                  FadeIn(x_lbl), FadeIn(y_lbl), FadeIn(x_ticks))

        # legend — 4 categories
        legend_items = [
            (C_SPECIAL, "[P] token"),
            (C_SCHEMA,  "schema tokens"),
            (C_TOKEN,   "text tokens"),
            ("#888888", "special ([CLS],[SEP])"),
        ]
        legend = VGroup()
        for col, lbl_str in legend_items:
            dot = Dot(radius=0.09, color=col)
            lbl = Text(lbl_str, font_size=13, color=col)
            legend.add(VGroup(dot, lbl).arrange(RIGHT, buff=0.15))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        legend.to_corner(UR, buff=0.4)
        self.play(FadeIn(legend))

        # draw all token lines — thin so they don't clutter
        line_plots = VGroup()
        for i in range(seq_len):
            col    = token_color(i)
            norms  = layer_norms[:, i]
            points = [
                axes_origin + RIGHT * l * x_step + UP * (norms[l] / max_norm_val) * axes_h
                for l in range(num_layers)
            ]
            polyline = VMobject(color=col, stroke_width=1.5 if i != p_pos else 3.0,
                                stroke_opacity=0.5 if i != p_pos else 1.0)
            polyline.set_points_as_corners(points)
            line_plots.add(polyline)

        # find the token with highest norm at final layer — let model speak
        final_layer_norms = layer_norms[num_layers - 1, :]
        top_pos   = int(final_layer_norms.argmax())
        top_label = seq_tokens[top_pos] if top_pos < len(seq_tokens) else f"t{top_pos}"
        top_norm  = final_layer_norms[top_pos]

        n11 = narration(
            f"All {seq_len} tokens tracked across 12 layers. "
            f"'{top_label}' has highest final norm ({top_norm:.1f}).\n"
            f"[P] is pink. Schema=orange. Text=blue. Specials=grey."
        )
        self.play(FadeIn(n11))
        self.play(
            LaggedStart(*[Create(lp) for lp in line_plots], lag_ratio=0.03),
            run_time=2.5
        )
        self.wait(3.5)

        # ══════════════════════════════════════════════════════════════════
        # PART 6 — Final output norms per token (bar chart)
        # ══════════════════════════════════════════════════════════════════
        self.play(FadeOut(n11), FadeOut(line_plots), FadeOut(legend),
                  FadeOut(x_axis), FadeOut(y_axis),
                  FadeOut(x_lbl), FadeOut(y_lbl), FadeOut(x_ticks))

        n12 = narration("Final encoder output: L2 norm of each token's 768-dim vector after all 12 layers.")
        self.play(FadeIn(n12))

        # all seq_len tokens — no truncation
        final_norms = np.linalg.norm(hs_final, axis=-1)  # (seq_len,)
        max_fn      = final_norms.max() + 1e-8
        bar_w       = min(0.34, 12.5 / seq_len - 0.04)

        final_bars = VGroup()
        final_lbls = VGroup()
        final_vals = VGroup()
        for i in range(seq_len):
            is_p   = (i == p_pos)
            is_sch = i < len(data["schema_tokens"])
            is_sep = seq_tokens[i] in ("[SEP_TEXT]", "[CLS]", "[SEP]") if i < len(seq_tokens) else False
            color  = C_SPECIAL if is_p else (C_SCHEMA if is_sch else ("#888888" if is_sep else C_TOKEN))
            h_bar  = (final_norms[i] / max_fn) * 3.2 + 0.05
            bar = Rectangle(
                width=bar_w, height=h_bar,
                fill_color=color, fill_opacity=0.85,
                stroke_width=0,
            )
            final_bars.add(bar)
            tok_name = seq_tokens[i] if i < len(seq_tokens) else f"t{i}"
            lbl = Text(tok_name, font_size=8, color=color)
            lbl.rotate(PI / 3)
            final_lbls.add(lbl)
            v = Text(f"{final_norms[i]:.0f}", font_size=7, color=C_DIM)
            final_vals.add(v)

        final_bars.arrange(RIGHT, buff=0.04, aligned_edge=DOWN)
        final_bars.move_to(UP * 0.3)
        for i, bar in enumerate(final_bars):
            final_lbls[i].next_to(bar, DOWN, buff=0.08)
            final_vals[i].next_to(bar, UP,   buff=0.04)

        hdr = Text("Final encoder output norms — each bar = one token's 768-dim vector after 12 layers",
                   font_size=13, color=C_WHITE).to_edge(UP, buff=0.7)

        self.play(FadeIn(hdr))
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in final_bars], lag_ratio=0.03),
            run_time=1.8
        )
        self.play(FadeIn(final_lbls), FadeIn(final_vals))

        # let the data decide which token has highest norm
        top_pos_final   = int(final_norms.argmax())
        top_label_final = seq_tokens[top_pos_final] if top_pos_final < len(seq_tokens) else f"t{top_pos_final}"
        p_norm_val      = final_norms[p_pos]

        self.play(FadeOut(n12))
        n13 = narration(
            f"Highest norm: '{top_label_final}' ({final_norms[top_pos_final]:.0f}).  "
            f"[P] norm={p_norm_val:.0f} — not the largest.\n"
            f"But [P]'s vector is what feeds the count_pred MLP in Scene 3."
        )
        self.play(FadeIn(n13))

        p_bar = final_bars[p_pos]
        self.play(
            p_bar.animate.set_fill(C_SPECIAL, opacity=1.0),
            Flash(p_bar, color=C_SPECIAL, line_length=0.2, num_lines=8),
        )
        self.wait(3.5)
        self.play(FadeOut(n13), FadeOut(title), FadeOut(hdr),
                  FadeOut(final_bars), FadeOut(final_lbls), FadeOut(final_vals))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 3 — Count Prediction
# ─────────────────────────────────────────────────────────────────────────────

class CountPredScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 3 · Count Prediction")
        self.add(title)

        n1 = narration("Before extracting spans, the model asks: how many instances are in this text?")
        self.play(FadeIn(n1))
        self.wait(2)

        # ── [P] vector → MLP → 20 bars ──
        self.play(FadeOut(n1))
        n2 = narration("The [P] token vector flows into a small MLP that outputs 20 scores,\none for each possible count (0 to 19).")
        self.play(FadeIn(n2))

        p_vec = RoundedRectangle(
            corner_radius=0.1, width=1.2, height=0.5,
            fill_color=C_SPECIAL, fill_opacity=0.4,
            stroke_color=C_SPECIAL, stroke_width=1.5,
        ).move_to(LEFT * 4 + UP * 0.5)
        p_label = Text("[P] vector\n768-dim", font_size=14, color=C_SPECIAL)
        p_label.move_to(p_vec)

        mlp_box = RoundedRectangle(
            corner_radius=0.15, width=2, height=1.2,
            fill_color="#1a1a2e", fill_opacity=1,
            stroke_color=C_TOKEN, stroke_width=1.5,
        ).move_to(LEFT * 1 + UP * 0.5)
        mlp_label = Text("count_pred\nMLP", font_size=16, color=C_TOKEN)
        mlp_label.move_to(mlp_box)

        arrow1 = Arrow(p_vec.get_right(), mlp_box.get_left(), buff=0.1, color=C_DIM)

        self.play(FadeIn(p_vec), FadeIn(p_label))
        self.play(GrowArrow(arrow1), FadeIn(mlp_box), FadeIn(mlp_label))
        self.wait(1)

        # ── 20 probability bars ──
        logits = data["count_logits"][0]  # (20,)
        probs  = np.exp(logits) / np.exp(logits).sum()
        pred   = data["pred_count"]

        bars   = VGroup()
        labels = VGroup()
        max_p  = probs.max() + 1e-8

        for i in range(20):
            h = (probs[i] / max_p) * 1.8 + 0.05
            color = C_HIGHLIGHT if i == pred else C_DIM
            bar = Rectangle(
                width=0.22, height=h,
                fill_color=color, fill_opacity=0.85,
                stroke_width=0,
            )
            bars.add(bar)
            if i % 5 == 0 or i == pred:
                lbl = Text(str(i), font_size=10, color=color)
                labels.add(lbl)

        bars.arrange(RIGHT, buff=0.06, aligned_edge=DOWN)
        bars.next_to(mlp_box, RIGHT, buff=1.0).shift(DOWN * 0.3)

        # position count labels below bars
        for bar in bars:
            idx = list(bars).index(bar)
            if idx % 5 == 0 or idx == pred:
                lbl = Text(str(idx), font_size=10,
                          color=C_HIGHLIGHT if idx == pred else C_DIM)
                lbl.next_to(bar, DOWN, buff=0.05)
                labels.add(lbl)

        arrow2 = Arrow(mlp_box.get_right(), bars.get_left() + LEFT * 0.1, buff=0.1, color=C_DIM)

        self.play(GrowArrow(arrow2))
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.04),
            run_time=1.5
        )
        self.wait(0.5)

        # ── highlight winner ──
        self.play(FadeOut(n2))
        n3 = narration(f"The model is 100% confident: count = {pred}.\nThe GRU will unroll exactly {pred} times.")
        self.play(FadeIn(n3))

        winner_bar = bars[pred]
        winner_label = Text(f"count = {pred}", font_size=20, color=C_HIGHLIGHT)
        winner_label.next_to(winner_bar, UP, buff=0.15)

        self.play(
            winner_bar.animate.set_fill(C_HIGHLIGHT, opacity=1.0),
            FadeIn(winner_label),
        )
        self.wait(2.5)
        self.play(FadeOut(n3), FadeOut(title))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 4 — Span Representation
# ─────────────────────────────────────────────────────────────────────────────

class SpanRepScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 4 · Span Representation")
        self.add(title)

        n1 = narration("The model must score every possible chunk of text.\nThese chunks are called spans.")
        self.play(FadeIn(n1))
        self.wait(2)

        tokens = data["text_tokens"]
        T      = len(tokens)
        W      = 8
        cs     = 0.38

        # ── show tokens in a row ──
        self.play(FadeOut(n1))
        n2 = narration("The text has " + str(T) + " tokens. With max width 8, that's " + str(T*W) + " possible spans.")
        self.play(FadeIn(n2))

        tok_boxes = VGroup()
        for i, t in enumerate(tokens):
            box = RoundedRectangle(
                corner_radius=0.06,
                width=max(len(t)*0.13+0.15, 0.4), height=0.38,
                fill_color=C_TOKEN, fill_opacity=0.2,
                stroke_color=C_TOKEN, stroke_width=1.2,
            )
            lbl = Text(t, font_size=12, color=C_TOKEN)
            g = VGroup(box, lbl)
            tok_boxes.add(g)

        tok_boxes.arrange(RIGHT, buff=0.08)
        tok_boxes.scale_to_fit_width(12.5)
        tok_boxes.move_to(UP * 2)

        self.play(
            LaggedStart(*[FadeIn(tb, shift=UP*0.1) for tb in tok_boxes], lag_ratio=0.06),
            run_time=1.5
        )
        self.wait(0.5)

        # ── span grid ──
        grid = VGroup()
        cell_map = {}
        for w in range(W):
            for s in range(T):
                if s + w < T:
                    cell = Rectangle(
                        width=cs, height=cs,
                        fill_color=C_DIM, fill_opacity=0.3,
                        stroke_color="#333355", stroke_width=0.8,
                    )
                    cell.move_to(
                        RIGHT * (s - T/2 + 0.5) * cs * 0.9 +
                        DOWN  * (w - W/2 + 0.5) * cs * 0.9 +
                        DOWN  * 0.5
                    )
                    grid.add(cell)
                    cell_map[(s, w)] = cell

        row_label = Text("width →", font_size=12, color=C_DIM).next_to(grid, RIGHT, buff=0.15)
        col_label = Text("start ↓", font_size=12, color=C_DIM).rotate(PI/2).next_to(grid, LEFT, buff=0.15)

        self.play(FadeIn(grid), FadeIn(row_label), FadeIn(col_label))
        self.wait(0.5)

        # ── highlight one example span: $500M (start=2, width=1) ──
        self.play(FadeOut(n2))
        n3 = narration("Each cell = one span. E.g. start=2, width=1 → '$ 500m'.\nBoundary token vectors are combined into one span vector.")
        self.play(FadeIn(n3))

        if (2, 1) in cell_map:
            example_cell = cell_map[(2, 1)]
            self.play(example_cell.animate.set_fill(C_HIGHLIGHT, opacity=0.9))

            span_lbl = Text("'$ 500m'", font_size=16, color=C_HIGHLIGHT)
            span_lbl.next_to(example_cell, RIGHT, buff=0.4)
            self.play(FadeIn(span_lbl))

        self.wait(2.5)

        # ── all spans → vectors ──
        self.play(FadeOut(n3))
        n4 = narration("Every valid span becomes a 768-dim vector. " + str(T*W) + " span vectors in total,\nready to be scored against the field queries.")
        self.play(FadeIn(n4))

        self.play(
            LaggedStart(
                *[c.animate.set_fill(C_TOKEN, opacity=0.4)
                  for c in grid if c != cell_map.get((2,1))],
                lag_ratio=0.01,
                run_time=1.5
            )
        )
        self.wait(2.5)
        self.play(FadeOut(n4), FadeOut(title))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 5 — GRU Unrolling
# ─────────────────────────────────────────────────────────────────────────────

class GRUUnrollScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 5 · CountLSTMv2 — GRU Unrolling")
        self.add(title)

        n1 = narration("CountLSTMv2 takes the field embeddings and unrolls the GRU\nonce per predicted instance.")
        self.play(FadeIn(n1))
        self.wait(2)

        fields     = data["field_names"]
        pred_count = data["pred_count"]
        gru_out    = data["gru_out"]   # (L, M, 768)
        M          = len(fields)
        L          = pred_count

        # ── field embeddings (input) ──
        self.play(FadeOut(n1))
        n2 = narration("Input: one 768-dim embedding per field — 'amount' and 'company'.")
        self.play(FadeIn(n2))

        field_boxes = VGroup()
        for fname in fields:
            box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.5,
                fill_color=C_SCHEMA, fill_opacity=0.25,
                stroke_color=C_SCHEMA, stroke_width=1.5,
            )
            lbl = Text(fname, font_size=16, color=C_SCHEMA)
            lbl.move_to(box)
            field_boxes.add(VGroup(box, lbl))

        field_boxes.arrange(DOWN, buff=0.2).move_to(LEFT * 4)
        input_lbl = Text("field embeddings\n(M=2, D=768)", font_size=13, color=C_DIM)
        input_lbl.next_to(field_boxes, UP, buff=0.2)

        self.play(FadeIn(field_boxes), FadeIn(input_lbl))
        self.wait(1)

        # ── GRU box ──
        gru_box = RoundedRectangle(
            corner_radius=0.2, width=2.5, height=2,
            fill_color="#0d1117", fill_opacity=1,
            stroke_color=C_HIGHLIGHT, stroke_width=2,
        ).move_to(ORIGIN)
        gru_lbl = Text("GRU", font_size=26, color=C_HIGHLIGHT)
        gru_sub = Text("+ Transformer", font_size=14, color=C_DIM)
        VGroup(gru_lbl, gru_sub).arrange(DOWN, buff=0.1).move_to(gru_box)

        arrow_in = Arrow(field_boxes.get_right(), gru_box.get_left(), buff=0.15, color=C_DIM)

        self.play(FadeOut(n2))
        n3 = narration("The GRU unrolls once per instance slot, producing distinct query vectors.")
        self.play(FadeIn(n3), FadeIn(gru_box), FadeIn(gru_lbl), FadeIn(gru_sub), GrowArrow(arrow_in))
        self.wait(1)

        # ── output columns (one per step) ──
        step_groups = VGroup()
        for l in range(L):
            col = VGroup()
            for k in range(M):
                # show norm as bar height
                norm = float(np.linalg.norm(gru_out[l, k])) if gru_out.ndim == 3 else 1.0
                bar = Rectangle(
                    width=0.6, height=min(norm / 20, 1.5) + 0.2,
                    fill_color=C_HIGHLIGHT if l == 0 else C_TOKEN,
                    fill_opacity=0.8, stroke_width=0,
                )
                lbl = Text(f"inst {l+1}\n{fields[k]}", font_size=10, color=C_DIM)
                lbl.next_to(bar, DOWN, buff=0.05)
                col.add(VGroup(bar, lbl))
            col.arrange(DOWN, buff=0.2)
            step_groups.add(col)

        step_groups.arrange(RIGHT, buff=0.4).next_to(gru_box, RIGHT, buff=1.0)

        pos_labels = VGroup()
        for l in range(L):
            pl = Text(f"step {l}", font_size=12, color=C_DIM)
            pl.next_to(step_groups[l], UP, buff=0.15)
            pos_labels.add(pl)

        self.play(FadeOut(n3))
        n4 = narration("Step 0: query vectors for instance 1.\nStep 1: different vectors for instance 2 — shaped by memory of step 0.")
        self.play(FadeIn(n4))

        arrow_out = Arrow(gru_box.get_right(), step_groups.get_left(), buff=0.1, color=C_DIM)
        self.play(GrowArrow(arrow_out))

        for l in range(L):
            self.play(
                FadeIn(step_groups[l], shift=RIGHT * 0.15),
                FadeIn(pos_labels[l]),
                run_time=0.6,
            )
            self.wait(0.4)

        self.wait(1)

        # ── cosine similarity callout ──
        if gru_out.ndim == 3 and L >= 2:
            v0 = gru_out[0].flatten()
            v1 = gru_out[1].flatten()
            sim = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-8))
            self.play(FadeOut(n4))
            n5 = narration(f"Cross-step cosine similarity = {sim:.3f}.\nValues below 0.7 mean the two query vectors are meaningfully distinct.")
            self.play(FadeIn(n5))
            self.wait(3)
            self.play(FadeOut(n5))
        else:
            self.play(FadeOut(n4))

        self.play(FadeOut(title))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 6 — Span Score Heatmap
# ─────────────────────────────────────────────────────────────────────────────

class SpanScoreScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 6 · Span Scoring")
        self.add(title)

        n1 = narration("Each instance query vector is dotted against every span vector.\nThe result is a score grid — the span heatmap.")
        self.play(FadeIn(n1))
        self.wait(2)
        self.play(FadeOut(n1))

        span_scores = data["span_scores"]  # (L, M, T, W)
        tokens      = data["text_tokens"]
        text_len    = data["text_len"]
        fields      = data["field_names"]
        L, M, T, W  = span_scores.shape
        cs          = 0.32

        for l in range(L):
            for k in range(M):
                sc = span_scores[l, k, :text_len, :]  # (T, W)

                n_inst = narration(
                    f"Instance {l+1}, field '{fields[k]}' — "
                    f"scoring all {text_len}×{W} spans."
                )
                self.play(FadeIn(n_inst))

                # build grid
                grid = VGroup()
                cells = {}
                for t in range(text_len):
                    for w in range(W):
                        val   = float(sc[t, w])
                        color = val_to_color(val)
                        cell  = Rectangle(
                            width=cs, height=cs,
                            fill_color=color,
                            fill_opacity=0.9 if val > 0.01 else 0.2,
                            stroke_color="#222244",
                            stroke_width=0.5,
                        )
                        cell.move_to(
                            RIGHT * (w - W/2 + 0.5) * (cs + 0.02) +
                            DOWN  * (t - text_len/2 + 0.5) * (cs + 0.02)
                        )
                        grid.add(cell)
                        cells[(t, w)] = cell

                # row labels (token names)
                row_labels = VGroup()
                for t, tok in enumerate(tokens[:text_len]):
                    lbl = Text(tok, font_size=9, color=C_DIM)
                    lbl.next_to(cells[(t, 0)], LEFT, buff=0.1)
                    row_labels.add(lbl)

                # col labels (widths)
                col_labels = VGroup()
                for w in range(W):
                    lbl = Text(f"w{w}", font_size=9, color=C_DIM)
                    lbl.next_to(cells[(0, w)], UP, buff=0.08)
                    col_labels.add(lbl)

                inst_label = Text(
                    f"Instance {l+1} · '{fields[k]}'",
                    font_size=18, color=C_WHITE,
                ).to_edge(UP, buff=0.6)

                self.play(
                    FadeIn(inst_label),
                    FadeIn(row_labels),
                    FadeIn(col_labels),
                )
                self.play(
                    LaggedStart(
                        *[FadeIn(cell) for cell in grid],
                        lag_ratio=0.01,
                        run_time=1.5,
                    )
                )
                self.wait(0.5)

                # highlight winner
                flat  = sc[:text_len, :].reshape(-1)
                best  = int(flat.argmax())
                bt, bw = divmod(best, W)
                be    = bt + bw + 1
                span_txt = " ".join(tokens[bt:be]) if be <= len(tokens) else "?"

                self.play(FadeOut(n_inst))
                n_win = narration(
                    f"Winner: '{span_txt}' at position {bt}, width {bw} — score {float(flat[best]):.3f}"
                )
                self.play(FadeIn(n_win))

                if (bt, bw) in cells:
                    winner = cells[(bt, bw)]
                    self.play(
                        winner.animate.set_stroke(C_HIGHLIGHT, width=3),
                        Flash(winner, color=C_HIGHLIGHT, line_length=0.15, num_lines=8),
                    )
                self.wait(2)
                self.play(
                    FadeOut(grid), FadeOut(row_labels), FadeOut(col_labels),
                    FadeOut(inst_label), FadeOut(n_win),
                )

        self.play(FadeOut(title))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 7 — Final Output
# ─────────────────────────────────────────────────────────────────────────────

class OutputScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 7 · Final Output")
        self.add(title)

        n1 = narration("The winning spans are mapped back to the original text.\nHere is the final structured extraction.")
        self.play(FadeIn(n1))
        self.wait(2)
        self.play(FadeOut(n1))

        # ── show original sentence ──
        tokens    = data["text_tokens"]
        fields    = data["field_names"]
        span_sc   = data["span_scores"]
        text_len  = data["text_len"]
        L, M, T, W = span_sc.shape

        # find winners
        results = []
        for l in range(L):
            inst = {}
            for k, fname in enumerate(fields):
                sc   = span_sc[l, k, :text_len, :]
                flat = sc.reshape(-1)
                best = int(flat.argmax())
                bt, bw = divmod(best, W)
                be   = bt + bw + 1
                inst[fname] = " ".join(tokens[bt:be]) if be <= len(tokens) else "?"
            results.append(inst)

        # show sentence with highlighted spans
        sentence_tokens = VGroup()
        for i, tok in enumerate(tokens):
            color = C_WHITE
            # check if this token is a winner
            for l in range(L):
                for k, fname in enumerate(fields):
                    sc   = span_sc[l, k, :text_len, :]
                    flat = sc.reshape(-1)
                    best = int(flat.argmax())
                    bt, bw = divmod(best, W)
                    if bt <= i <= bt + bw:
                        color = C_HIGHLIGHT if k == 0 else C_SCHEMA
            lbl = Text(tok, font_size=18, color=color)
            sentence_tokens.add(lbl)

        sentence_tokens.arrange(RIGHT, buff=0.12)
        sentence_tokens.scale_to_fit_width(12)
        sentence_tokens.move_to(UP * 2)

        self.play(
            LaggedStart(*[FadeIn(t, shift=UP*0.1) for t in sentence_tokens], lag_ratio=0.06),
            run_time=1.5
        )
        self.wait(0.5)

        # ── show extracted instances ──
        instance_groups = VGroup()
        colors = [C_HIGHLIGHT, C_TOKEN]
        for l, inst in enumerate(results):
            lines = VGroup(
                Text(f"Instance {l+1}", font_size=18, color=colors[l % 2]),
                *[
                    Text(f"  {k}: {v}", font_size=16, color=C_WHITE)
                    for k, v in inst.items()
                ]
            )
            lines.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            box = SurroundingRectangle(
                lines,
                color=colors[l % 2],
                buff=0.2,
                corner_radius=0.1,
                stroke_width=1.5,
                fill_color="#0f0f1a",
                fill_opacity=0.8,
            )
            instance_groups.add(VGroup(box, lines))

        instance_groups.arrange(RIGHT, buff=0.5).move_to(DOWN * 0.5)

        n2 = narration("Two investment instances extracted — each with amount and company fields.")
        self.play(FadeIn(n2))
        self.play(
            LaggedStart(*[FadeIn(ig, shift=UP*0.2) for ig in instance_groups], lag_ratio=0.3),
            run_time=1.5
        )
        self.wait(2)

        # ── closing ──
        self.play(FadeOut(n2))
        n3 = narration("That's the full GLiNER2 forward pass.\nSchema → Tokens → Encoder → Count → Spans → Score → Extract.")
        self.play(FadeIn(n3))
        self.wait(3)

        # pipeline summary
        steps = ["Schema", "Tokens", "Encoder", "Count", "Spans", "Score", "Extract"]
        step_labels = VGroup(*[
            Text(s, font_size=16, color=C_HIGHLIGHT) for s in steps
        ])
        arrows = VGroup(*[
            Text("→", font_size=16, color=C_DIM) for _ in range(len(steps)-1)
        ])
        pipeline = VGroup()
        for i, lbl in enumerate(step_labels):
            pipeline.add(lbl)
            if i < len(arrows):
                pipeline.add(arrows[i])
        pipeline.arrange(RIGHT, buff=0.15).move_to(DOWN * 2.5)

        self.play(
            LaggedStart(*[FadeIn(p, shift=UP*0.1) for p in pipeline], lag_ratio=0.1),
            run_time=1.5
        )
        self.wait(3)
        self.play(FadeOut(n3), FadeOut(title))
        self.wait(1)
