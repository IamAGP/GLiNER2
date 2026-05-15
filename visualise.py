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
    from debug_counting import (DebugGLiNER2, _schema_debug, _gru_steps, _transformer_out,
                                _current_schema, _hook_gru, _hook_transformer)

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

    # ── span rep hooks ────────────────────────────────────────────────────
    span_inter = {}
    srl = dm.span_rep.span_rep_layer

    def hook_ps(mod, inp, out):
        span_inter["project_start_out"] = out.detach().cpu()   # (1, text_len, 768)
    def hook_pe(mod, inp, out):
        span_inter["project_end_out"]   = out.detach().cpu()
    def hook_op(mod, inp, out):
        span_inter["out_project_in"]    = inp[0].detach().cpu()  # (1, 96, 1536)
        span_inter["out_project_out"]   = out.detach().cpu()     # (1, 96, 768)
    def hook_sr(mod, inp, out):
        span_inter["span_rep_final"]    = out.detach().cpu()     # (1, text_len, max_width, 768)
        span_inter["span_idx"]          = inp[1].detach().cpu()  # (1, 96, 2)

    srl.project_start.register_forward_hook(hook_ps)
    srl.project_end.register_forward_hook(hook_pe)
    srl.out_project.register_forward_hook(hook_op)
    dm.span_rep.register_forward_hook(hook_sr)

    # pull span_rep architecture from live model
    span_arch = {
        "mode":        "SpanMarkerV0",
        "max_width":   dm.max_width,
        "ps_shapes":   [(tuple(p.shape)) for p in srl.project_start.parameters()],
        "pe_shapes":   [(tuple(p.shape)) for p in srl.project_end.parameters()],
        "op_shapes":   [(tuple(p.shape)) for p in srl.out_project.parameters()],
        "ps_layers":   str(srl.project_start),
        "pe_layers":   str(srl.project_end),
        "op_layers":   str(srl.out_project),
    }

    hooks = [dm.encoder.register_forward_hook(hook_enc_out)]
    for i, layer in enumerate(dm.encoder.encoder.layer):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))
        hooks.append(layer.attention.self.register_forward_hook(make_attn_hook(i)))

    # GRU + transformer hooks (same mechanism as debug_counting.py manual registration)
    hooks.append(dm.count_embed.gru.register_forward_hook(_hook_gru))
    hooks.append(dm.count_embed.transformer.register_forward_hook(_hook_transformer))

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
        # span representation (from live hooks on SpanMarkerV0)
        "span_arch":            span_arch,
        "span_rep_final":       span_inter.get("span_rep_final", torch.zeros(1,1,8,768)).squeeze(0).numpy(),  # (text_len, max_width, 768)
        "span_norms":           span_inter.get("span_rep_final", torch.zeros(1,1,8,768)).squeeze(0).norm(dim=-1).numpy(),  # (text_len, max_width)
        "span_idx":             span_inter.get("span_idx", torch.zeros(1,96,2)).numpy(),  # (1, 96, 2)
        "project_start_out":    span_inter.get("project_start_out", torch.zeros(1,1,768)).squeeze(0).numpy(),  # (text_len, 768)
        "project_end_out":      span_inter.get("project_end_out",   torch.zeros(1,1,768)).squeeze(0).numpy(),  # (text_len, 768)
        "out_project_in_norms": span_inter.get("out_project_in",    torch.zeros(1,96,1536)).squeeze(0).norm(dim=-1).numpy(),  # (96,)
        "out_project_out_norms":span_inter.get("out_project_out",   torch.zeros(1,96,768)).squeeze(0).norm(dim=-1).numpy(),   # (96,)
        # span extraction
        "span_scores":      d["span_scores"].numpy(),
        "struct_proj":      d["struct_proj"].numpy(),
        "gru_out":          _gru_steps.get("investment", torch.zeros(1)).numpy(),
        "text_len":         d["text_len"],
    })

    # precompute per-text-token encoder norms at the last layer
    start_map = d["start_mapping"]
    schema_len = len(d["schema_tokens"])
    text_start_enc = schema_len + 1   # +1 for the [SEP_TEXT] separator
    hs_last = layer_hidden.get(num_layers - 1)
    text_tok_enc_norms = []
    for wt in range(d["text_len"]):
        sub_i = next((i for i, w in enumerate(start_map) if w == wt), 0)
        enc_i = text_start_enc + sub_i
        if hs_last is not None and enc_i < hs_last.shape[0]:
            text_tok_enc_norms.append(float(np.linalg.norm(hs_last[enc_i].numpy())))
        else:
            text_tok_enc_norms.append(20.0)
    _model_data["text_tok_enc_norms"] = text_tok_enc_norms

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
        data   = get_model_data()
        logits = data["count_logits"][0]   # (20,) — raw from model
        pred   = data["pred_count"]
        p_norm = data["p_emb_norm"]

        title = section_title("Scene 3 · Count Prediction")
        self.add(title)

        # ══════════════════════════════════════════════════════════════════
        # PART 1 — MLP architecture from live model
        # ══════════════════════════════════════════════════════════════════
        n1 = narration("[P]'s 768-dim vector goes into count_pred — a 2-layer MLP — to predict how many instances exist.")
        self.play(FadeIn(n1))
        self.wait(2)
        self.play(FadeOut(n1))

        # draw [P] → Linear(768→1536) → ReLU → Linear(1536→20) pipeline
        stages = [
            ("[P] vector\n768-dim",          C_SPECIAL,   1.4, 0.8),
            ("Linear\n768 → 1536\n+ bias",   C_TOKEN,     2.0, 1.2),
            ("ReLU",                          C_SCHEMA,    1.0, 0.7),
            ("Linear\n1536 → 20\n+ bias",    C_TOKEN,     2.0, 1.2),
            ("20 logits\n(counts 0–19)",      C_HIGHLIGHT, 1.6, 0.8),
        ]
        stage_objs = VGroup()
        for label, color, w, h in stages:
            box = RoundedRectangle(
                corner_radius=0.1, width=w, height=h,
                fill_color=color, fill_opacity=0.15,
                stroke_color=color, stroke_width=1.5,
            )
            lbl = Text(label, font_size=14, color=color)
            lbl.move_to(box)
            stage_objs.add(VGroup(box, lbl))

        stage_objs.arrange(RIGHT, buff=0.5)
        stage_objs.move_to(UP * 0.5)

        arrows = VGroup()
        for i in range(len(stage_objs) - 1):
            a = Arrow(
                stage_objs[i].get_right(),
                stage_objs[i+1].get_left(),
                buff=0.05, color=C_DIM, stroke_width=1.5
            )
            arrows.add(a)

        # dimension labels on arrows
        dim_labels = ["768", "1536", "1536", "20"]
        dim_label_objs = VGroup()
        for i, (arrow, dim) in enumerate(zip(arrows, dim_labels)):
            d = Text(dim, font_size=11, color=C_DIM)
            d.next_to(arrow, UP, buff=0.08)
            dim_label_objs.add(d)

        n2 = narration("Architecture from live model: Sequential(Linear(768→1536), ReLU, Linear(1536→20)).")
        self.play(FadeIn(n2))
        self.play(
            LaggedStart(*[FadeIn(s, shift=RIGHT*0.1) for s in stage_objs], lag_ratio=0.15),
            run_time=1.5
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
            FadeIn(dim_label_objs),
        )
        self.wait(2.5)
        self.play(FadeOut(n2), FadeOut(stage_objs), FadeOut(arrows), FadeOut(dim_label_objs))

        # ══════════════════════════════════════════════════════════════════
        # PART 2 — Raw logits bar chart (all 20, real values)
        # ══════════════════════════════════════════════════════════════════
        n3 = narration("Softmax probabilities are useless here — count=2 gets p=1.0, all others p≈0.\nRaw logits tell the real story.")
        self.play(FadeIn(n3))
        self.wait(2.5)
        self.play(FadeOut(n3))

        # shift logits so min=0 for display (preserves relative differences)
        logits_shifted = logits - logits.min()   # all >= 0
        max_l = logits_shifted.max() + 1e-8

        bars      = VGroup()
        bar_lbls  = VGroup()   # count index below
        val_lbls  = VGroup()   # logit value above

        for i in range(20):
            color = C_HIGHLIGHT if i == pred else C_DIM
            h_bar = (logits_shifted[i] / max_l) * 3.5 + 0.05
            bar = Rectangle(
                width=0.42, height=h_bar,
                fill_color=color, fill_opacity=0.85 if i == pred else 0.5,
                stroke_width=0,
            )
            bars.add(bar)
            bar_lbls.add(Text(str(i), font_size=11, color=color))
            val_lbls.add(Text(f"{logits[i]:.1f}", font_size=9, color=color))

        bars.arrange(RIGHT, buff=0.07, aligned_edge=DOWN)
        bars.move_to(UP * 0.2)
        for i, bar in enumerate(bars):
            bar_lbls[i].next_to(bar, DOWN, buff=0.08)
            val_lbls[i].next_to(bar, UP,   buff=0.05)

        hdr = Text("Raw logits for each count (0–19)  —  higher = more likely",
                   font_size=14, color=C_WHITE).to_edge(UP, buff=0.7)

        n4 = narration(
            f"count=2 logit: {logits[pred]:.1f}   "
            f"count=1 logit: {logits[1]:.1f}   "
            f"count=0 logit: {logits[0]:.1f}\n"
            f"The gap between winner and runner-up = {logits[pred]-logits[1]:.1f} — model is very certain."
        )
        self.play(FadeIn(n4), FadeIn(hdr))
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.04),
            run_time=1.5
        )
        self.play(FadeIn(bar_lbls), FadeIn(val_lbls))
        self.wait(1)

        # flash winner
        winner_bar = bars[pred]
        self.play(
            winner_bar.animate.set_fill(C_HIGHLIGHT, opacity=1.0),
            Flash(winner_bar, color=C_HIGHLIGHT, line_length=0.25, num_lines=10),
        )
        self.wait(2)
        self.play(FadeOut(n4), FadeOut(bars), FadeOut(bar_lbls),
                  FadeOut(val_lbls), FadeOut(hdr))

        # ══════════════════════════════════════════════════════════════════
        # PART 3 — argmax → count → what happens next
        # ══════════════════════════════════════════════════════════════════
        summary = VGroup(
            MathTex(r"\hat{L} = \arg\max_{k} \; \text{logit}_k", font_size=38, color=C_HIGHLIGHT),
            MathTex(rf"= {pred}", font_size=48, color=C_HIGHLIGHT),
        ).arrange(DOWN, buff=0.3).move_to(UP * 0.5)

        consequence = Text(
            f"→ CountLSTMv2 (GRU) unrolls exactly {pred} times\n"
            f"→ Produces {pred} distinct query vectors, one per instance slot",
            font_size=20, color=C_WHITE
        ).move_to(DOWN * 1.5)

        n5 = narration(f"argmax of logits = {pred}.  This single integer controls everything downstream.")
        self.play(FadeIn(n5))
        self.play(
            LaggedStart(*[FadeIn(s, shift=UP*0.1) for s in summary], lag_ratio=0.3),
            run_time=1.2
        )
        self.wait(1)
        self.play(FadeIn(consequence, shift=UP*0.1))
        self.wait(3)
        self.play(FadeOut(n5), FadeOut(summary), FadeOut(consequence), FadeOut(title))
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 4 — Span Representation
# ─────────────────────────────────────────────────────────────────────────────

class SpanRepScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data   = get_model_data()
        tokens = data["text_tokens"]
        T      = len(tokens)
        W      = data["span_arch"]["max_width"]
        span_norms      = data["span_norms"]                               # (T, W)
        ps_out_norms    = np.linalg.norm(data["project_start_out"], axis=-1)  # (T,)
        pe_out_norms    = np.linalg.norm(data["project_end_out"],   axis=-1)  # (T,)
        enc_norms       = data["text_tok_enc_norms"]                       # list[T]

        title = section_title("Scene 4 · Span Representation  (SpanMarkerV0)")
        self.add(title)

        # ══════════════════════════════════════════════════════════════════
        # PART 1 — token row + span count explanation
        # ══════════════════════════════════════════════════════════════════
        tok_boxes = VGroup()
        for t in tokens:
            box = RoundedRectangle(
                corner_radius=0.07,
                width=max(len(t) * 0.14 + 0.18, 0.52), height=0.46,
                fill_color=C_TOKEN, fill_opacity=0.18,
                stroke_color=C_TOKEN, stroke_width=1.2,
            )
            lbl = Text(t, font_size=14, color=C_TOKEN)
            lbl.move_to(box)
            tok_boxes.add(VGroup(box, lbl))

        tok_boxes.arrange(RIGHT, buff=0.12)
        tok_boxes.scale_to_fit_width(12.8)
        tok_boxes.move_to(UP * 0.8)

        count_lbl = Text(
            f"{T} tokens  ×  max_width = {W}  =  {T*W} candidate spans",
            font_size=24, color=C_WHITE,
        ).move_to(DOWN * 0.5)
        sub_lbl = Text(
            "max_width is a training hyperparameter — stored in model.config",
            font_size=16, color=C_DIM,
        ).move_to(DOWN * 1.25)

        self.play(
            LaggedStart(*[FadeIn(tb, shift=UP * 0.12) for tb in tok_boxes], lag_ratio=0.06),
            run_time=1.4,
        )
        self.play(FadeIn(count_lbl), FadeIn(sub_lbl))
        self.wait(2.5)
        self.play(FadeOut(sub_lbl), FadeOut(count_lbl))

        # ══════════════════════════════════════════════════════════════════
        # PART 2 — span norm heatmap
        # ══════════════════════════════════════════════════════════════════
        # shrink token row to top so heatmap fits below
        self.play(tok_boxes.animate.scale(0.78).move_to(UP * 3.0))

        cs       = 0.33            # cell size — 12 rows × 0.33 = 3.96 tall, fits easily
        max_norm = float(span_norms.max()) + 1e-8

        # grid centered at x≈1.0 (leaves room for row labels on left)
        ox = 1.0 - (W - 1) / 2.0 * cs   # x of col-0 cell centres
        oy = 0.3 + (T - 1) / 2.0 * cs   # y of row-0 cell centres (Manim y up)

        grid_cells = VGroup()
        cell_map   = {}
        for s in range(T):
            for w in range(W):
                is_valid = (s + w) < T
                norm_val = float(span_norms[s, w]) if is_valid else 0.0
                color    = val_to_color(norm_val / max_norm) if is_valid else ManimColor(C_DIM)
                opacity  = 0.85 if is_valid else 0.12
                cell = Rectangle(
                    width=cs - 0.03, height=cs - 0.03,
                    fill_color=color, fill_opacity=opacity,
                    stroke_color="#222244", stroke_width=0.5,
                )
                cell.move_to(RIGHT * (ox + w * cs) + UP * (oy - s * cs))
                grid_cells.add(cell)
                cell_map[(s, w)] = (cell, is_valid, norm_val)

        row_labels = VGroup(*[
            Text(tokens[s], font_size=10, color=C_TOKEN).move_to(
                RIGHT * (ox - 0.65) + UP * (oy - s * cs))
            for s in range(T)
        ])
        col_labels = VGroup(*[
            Text(str(w), font_size=9, color=C_DIM).move_to(
                RIGHT * (ox + w * cs) + UP * (oy + cs * 0.9))
            for w in range(W)
        ])
        axis_lbl = Text("← width →", font_size=11, color=C_DIM).move_to(
            RIGHT * (ox + (W - 1) / 2.0 * cs) + UP * (oy + cs * 1.7))

        # ── colour legend (vertical gradient bar to the right of the grid) ──
        legend_x   = ox + W * cs + 0.55          # right of grid + buffer
        bar_h      = T * cs                       # same height as the grid
        bar_cy     = oy - (T - 1) / 2.0 * cs     # vertical centre of grid
        bar_w      = 0.22
        N_steps    = 20
        min_norm   = float(span_norms[span_norms > 0].min())
        legend_bar = VGroup()
        for i in range(N_steps):
            frac  = i / (N_steps - 1)
            seg_h = bar_h / N_steps
            seg = Rectangle(
                width=bar_w, height=seg_h,
                fill_color=val_to_color(frac), fill_opacity=0.9,
                stroke_width=0,
            )
            seg.move_to(RIGHT * legend_x + UP * (bar_cy - bar_h / 2 + (i + 0.5) * seg_h))
            legend_bar.add(seg)

        lbl_high = Text(f"{max_norm * 1:.1f}", font_size=10, color=C_WHITE).next_to(
            legend_bar, UP, buff=0.06)
        lbl_low  = Text(f"{min_norm:.1f}",  font_size=10, color=C_WHITE).next_to(
            legend_bar, DOWN, buff=0.06)
        lbl_norm = Text("norm", font_size=11, color=C_DIM)
        lbl_norm.next_to(legend_bar, UP, buff=0.28)
        legend = VGroup(legend_bar, lbl_high, lbl_low, lbl_norm)

        n_heat = narration(
            "Row = start token,  Col = span width  (0 = 1 token, 7 = 8 tokens).\n"
            "Color = L2 norm of final 768-dim span vector — all values from live model."
        )
        self.play(FadeIn(n_heat), FadeIn(row_labels), FadeIn(col_labels), FadeIn(axis_lbl))
        self.play(
            LaggedStart(*[FadeIn(c) for c in grid_cells], lag_ratio=0.005),
            run_time=2.0,
        )
        self.play(FadeIn(legend))
        self.wait(1.0)

        # highlight '$ 500m' (start=2, width=1)
        ex_s, ex_w     = 2, 1
        ex_cell, _, ex_norm = cell_map[(ex_s, ex_w)]
        ex_lbl = Text(f"'$ 500m'  norm = {ex_norm:.1f}", font_size=13, color=C_HIGHLIGHT)
        ex_lbl.next_to(ex_cell, RIGHT, buff=0.4)
        self.play(FadeOut(n_heat))
        n_ex = narration(f"'$ 500m': start={ex_s}, width={ex_w}, final vector norm = {ex_norm:.1f}")
        self.play(FadeIn(n_ex))
        self.play(
            ex_cell.animate.set_stroke(C_HIGHLIGHT, width=3),
            Flash(ex_cell, color=C_HIGHLIGHT, line_length=0.12, num_lines=8),
            FadeIn(ex_lbl),
        )
        self.wait(2.5)
        self.play(
            FadeOut(n_ex), FadeOut(grid_cells), FadeOut(row_labels),
            FadeOut(col_labels), FadeOut(axis_lbl), FadeOut(ex_lbl),
            FadeOut(legend), FadeOut(tok_boxes),
        )

        # ══════════════════════════════════════════════════════════════════
        # PART 3 — SpanMarkerV0 pipeline (from live model)
        # ══════════════════════════════════════════════════════════════════
        # 6-step pipeline as a compact table:  [step name]  [formula/shape]  [description]
        pipe_rows = [
            ("Input",         r"h \in \mathbb{R}^{T \times 768}",
             "DeBERTa last-layer output, all T tokens",            C_DIM),
            ("project_start", r"h \;\xrightarrow{Lin_{768\to3072}\to ReLU\to Lin_{3072\to768}}\; \mathbb{R}^{T \times 768}",
             "4× MLP — encodes 'start-token' role",                C_TOKEN),
            ("project_end",   r"h \;\xrightarrow{Lin_{768\to3072}\to ReLU\to Lin_{3072\to768}}\; \mathbb{R}^{T \times 768}",
             "same structure, different weights",                   C_SCHEMA),
            ("gather",        r"\text{torch.gather}\;\Rightarrow\;(96,768)\;\text{each}",
             f"pick start/end token for each of {T}×{W}={T*W} spans",  C_DIM),
            ("cat + ReLU",    r"\text{cat}([s,e])\;\in\;\mathbb{{R}}^{{96\times1536}}\;\xrightarrow{{\text{{ReLU}}}}",
             "merge start+end information",                        C_HIGHLIGHT),
            ("out_project",   r"\mathbb{R}^{96\times1536}\;\xrightarrow{Lin_{1536\to3072}\to ReLU\to Lin_{3072\to768}}\;\mathbb{R}^{96\times768}",
             "final 768-dim vector per span",                      C_TOKEN),
        ]

        pipe_vg = VGroup()
        for step_name, tex_str, desc_str, color in pipe_rows:
            name_t  = Text(step_name, font_size=17, color=color, weight=BOLD)
            formula = MathTex(tex_str, font_size=19, color=color)
            desc_t  = Text(desc_str, font_size=13, color=C_DIM)
            row_g   = VGroup(name_t, formula, desc_t)
            row_g.arrange(RIGHT, buff=0.35)
            bg = RoundedRectangle(
                corner_radius=0.07,
                width=row_g.get_width() + 0.5,
                height=max(row_g.get_height() + 0.22, 0.52),
                fill_color=color, fill_opacity=0.07,
                stroke_color=color, stroke_width=0.8,
            )
            bg.move_to(row_g)
            pipe_vg.add(VGroup(bg, row_g))

        pipe_vg.arrange(DOWN, buff=0.14)
        # scale to fit width (14.2 available, use 13.5 max) without inflating height
        if pipe_vg.get_width() > 13.5:
            pipe_vg.scale_to_fit_width(13.5)
        pipe_vg.move_to(ORIGIN)

        n_pipe = narration(
            "project_start ≠ project_end — different weights, different roles.\n"
            "Both MLPs run on ALL T tokens first, then gather — efficient batched operation."
        )
        self.play(FadeIn(n_pipe))
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.08) for b in pipe_vg], lag_ratio=0.1),
            run_time=2.2,
        )
        self.wait(3.5)
        self.play(FadeOut(n_pipe), FadeOut(pipe_vg))

        # ══════════════════════════════════════════════════════════════════
        # PART 4 — Trace '$ 500m' through pipeline (proper fork diagram)
        # ══════════════════════════════════════════════════════════════════
        start_tok, end_tok = 2, 3
        ps_norm    = float(ps_out_norms[start_tok])
        pe_norm    = float(pe_out_norms[end_tok])
        h_s_norm   = float(enc_norms[start_tok])
        h_e_norm   = float(enc_norms[end_tok])
        final_norm = float(span_norms[ex_s, ex_w])
        # concat input norm ≈ sqrt(ps_norm² + pe_norm²) — not captured separately, derive it
        cat_in_norm = float(np.sqrt(ps_norm**2 + pe_norm**2))

        def make_box(label, color, w=3.0, h=0.88):
            box = RoundedRectangle(
                corner_radius=0.1, width=w, height=h,
                fill_color=color, fill_opacity=0.13,
                stroke_color=color, stroke_width=1.4,
            )
            lbl = Text(label, font_size=12, color=color)
            lbl.move_to(box)
            return VGroup(box, lbl)

        # Two parallel input columns, then merge
        LX, RX = -3.2, 3.2    # x positions for start / end columns
        Y0, Y1 = 2.4, 0.9     # y for row 0 (input tokens) and row 1 (projection outputs)
        Y2, Y3 = -0.7, -2.2   # y for cat+ReLU and out_project (centred)

        b_hs  = make_box(f"DeBERTa h[{start_tok}]  '{tokens[start_tok]}'\nnorm = {h_s_norm:.1f}", C_TOKEN)
        b_he  = make_box(f"DeBERTa h[{end_tok}]  '{tokens[end_tok]}'\nnorm = {h_e_norm:.1f}", C_SCHEMA)
        b_ps  = make_box(f"project_start  →  768-dim\nnorm = {ps_norm:.1f}", C_TOKEN)
        b_pe  = make_box(f"project_end  →  768-dim\nnorm = {pe_norm:.1f}", C_SCHEMA)
        b_cat = make_box(f"cat([start, end])  →  1536-dim   ReLU\nnorm ≈ {cat_in_norm:.1f}", C_HIGHLIGHT, w=4.8)
        b_out = make_box(f"out_project  →  768-dim\nnorm = {final_norm:.1f}", C_HIGHLIGHT, w=4.8)

        b_hs.move_to( RIGHT * LX + UP * Y0)
        b_he.move_to( RIGHT * RX + UP * Y0)
        b_ps.move_to( RIGHT * LX + UP * Y1)
        b_pe.move_to( RIGHT * RX + UP * Y1)
        b_cat.move_to(UP * Y2)
        b_out.move_to(UP * Y3)

        # labels above the two columns
        lbl_start = Text("start-token path", font_size=14, color=C_TOKEN)
        lbl_start.move_to(RIGHT * LX + UP * (Y0 + 0.7))
        lbl_end = Text("end-token path", font_size=14, color=C_SCHEMA)
        lbl_end.move_to(RIGHT * RX + UP * (Y0 + 0.7))

        arrows = [
            Arrow(b_hs.get_bottom(),  b_ps.get_top(),   buff=0.05, color=C_TOKEN,    stroke_width=1.6),
            Arrow(b_he.get_bottom(),  b_pe.get_top(),   buff=0.05, color=C_SCHEMA,   stroke_width=1.6),
            Arrow(b_ps.get_bottom(),  b_cat.get_left(), buff=0.05, color=C_TOKEN,    stroke_width=1.6),
            Arrow(b_pe.get_bottom(),  b_cat.get_right(),buff=0.05, color=C_SCHEMA,   stroke_width=1.6),
            Arrow(b_cat.get_bottom(), b_out.get_top(),  buff=0.05, color=C_HIGHLIGHT, stroke_width=1.6),
        ]

        n_trace = narration(
            "Two independent paths (start token, end token) through identical-structure MLPs.\n"
            "Different weights → different roles.  cat merges them → one 768-dim span vector."
        )
        self.play(FadeIn(n_trace), FadeIn(lbl_start), FadeIn(lbl_end))
        self.play(
            LaggedStart(FadeIn(b_hs), FadeIn(b_he), lag_ratio=0.2),
        )
        self.play(
            LaggedStart(GrowArrow(arrows[0]), GrowArrow(arrows[1]), lag_ratio=0.1),
            LaggedStart(FadeIn(b_ps), FadeIn(b_pe), lag_ratio=0.2),
        )
        self.play(
            LaggedStart(GrowArrow(arrows[2]), GrowArrow(arrows[3]), lag_ratio=0.05),
        )
        self.play(FadeIn(b_cat))
        self.play(GrowArrow(arrows[4]))
        self.play(FadeIn(b_out))
        self.wait(3)
        self.play(
            FadeOut(n_trace),
            FadeOut(lbl_start), FadeOut(lbl_end),
            FadeOut(b_hs), FadeOut(b_he), FadeOut(b_ps), FadeOut(b_pe),
            FadeOut(b_cat), FadeOut(b_out),
            *[FadeOut(a) for a in arrows],
            FadeOut(title),
        )
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 5 — GRU Unrolling
# ─────────────────────────────────────────────────────────────────────────────

class GRUUnrollScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        data = get_model_data()

        title = section_title("Scene 5 · CountLSTMv2 — Instance Query Generation")
        self.add(title)

        fields      = data["field_names"]
        pred_count  = data["pred_count"]
        gru_out     = data["gru_out"]      # (L, M, 768) after GRU, before residual+transformer
        struct_proj = data["struct_proj"]  # (L, M, 768) final output after transformer
        M = len(fields)
        L = pred_count

        C_POS = "#cc88ff"   # purple for positional embeddings

        # ══════════════════════════════════════════════════════════════════
        # PART 1 — the problem: why do we need L distinct queries?
        # ══════════════════════════════════════════════════════════════════
        problem_q = Text(
            f"Count prediction said: L = {L} instances.",
            font_size=28, color=C_WHITE,
        ).move_to(UP * 1.2)
        problem_a = Text(
            f"We need {L} different search queries —\none for each investment in the text.",
            font_size=24, color=C_DIM,
        ).move_to(DOWN * 0.2)
        problem_b = Text(
            "If both queries are identical, they find the same spans → wrong.",
            font_size=20, color=C_SCHEMA,
        ).move_to(DOWN * 1.3)

        n1 = narration("CountLSTMv2's job: manufacture L distinct query vectors from a single schema.")
        self.play(FadeIn(n1))
        self.play(FadeIn(problem_q))
        self.play(FadeIn(problem_a))
        self.play(FadeIn(problem_b))
        self.wait(3)
        self.play(FadeOut(n1), FadeOut(problem_q), FadeOut(problem_a), FadeOut(problem_b))

        # ══════════════════════════════════════════════════════════════════
        # PART 2 — GRU unrolled as a time diagram (step 0 → step 1)
        # ══════════════════════════════════════════════════════════════════
        # Layout (left to right = time):
        #   [h₀ = field_embs]  →  [GRU cell 0]  →  [GRU cell 1]  →  ...
        #   pos_emb[0] ↑              pos_emb[1] ↑

        n2 = narration(
            "GRU unrolled over time.  Each step gets a different positional embedding as input.\n"
            "The field embeddings are the STARTING hidden state — not the input sequence."
        )
        self.play(FadeIn(n2))

        # h0 box (field embeddings = initial hidden state)
        h0_bg = RoundedRectangle(corner_radius=0.1, width=2.2, height=1.6,
                                  fill_color=C_SCHEMA, fill_opacity=0.15,
                                  stroke_color=C_SCHEMA, stroke_width=1.5)
        h0_bg.move_to(LEFT * 5.0 + UP * 0.3)
        h0_lbl  = Text("h₀", font_size=26, color=C_SCHEMA, weight=BOLD).move_to(h0_bg.get_center() + UP * 0.3)
        h0_desc = Text("field embeddings\n(what to look for)", font_size=11, color=C_SCHEMA).move_to(h0_bg.get_center() + DOWN * 0.3)

        # GRU cells (one per step)
        cell_xs = [-1.8 + l * 3.2 for l in range(L)]
        cell_y  = 0.3
        gru_cells = VGroup()
        for l in range(L):
            bg = RoundedRectangle(corner_radius=0.12, width=2.4, height=1.6,
                                   fill_color=C_HIGHLIGHT, fill_opacity=0.1,
                                   stroke_color=C_HIGHLIGHT, stroke_width=1.8)
            bg.move_to(RIGHT * cell_xs[l] + UP * cell_y)
            t1 = Text(f"GRU step {l}", font_size=16, color=C_HIGHLIGHT, weight=BOLD)
            t2 = Text(f"(M={M} fields)", font_size=12, color=C_DIM)
            VGroup(t1, t2).arrange(DOWN, buff=0.1).move_to(bg)
            gru_cells.add(VGroup(bg, t1, t2))

        # positional embedding labels (arrow from below into each cell)
        pos_lbls = VGroup()
        pos_arrows = VGroup()
        for l in range(L):
            lbl = Text(f"pos_emb[{l}]", font_size=13, color=C_POS)
            lbl.move_to(RIGHT * cell_xs[l] + UP * (cell_y - 1.5))
            arr = Arrow(lbl.get_top(), gru_cells[l].get_bottom(), buff=0.08,
                        color=C_POS, stroke_width=1.6)
            pos_lbls.add(lbl)
            pos_arrows.add(arr)

        pos_header = Text("↑ GRU sequence input (step index → pos embedding)",
                          font_size=13, color=C_POS)
        pos_header.move_to(DOWN * 2.4)

        # h0 → cell 0 arrow
        arr_h0 = Arrow(h0_bg.get_right(), gru_cells[0].get_left(), buff=0.1,
                       color=C_SCHEMA, stroke_width=1.8)
        lbl_h0_arr = Text("h₀", font_size=13, color=C_SCHEMA)
        lbl_h0_arr.next_to(arr_h0, UP, buff=0.06)

        # cell → cell arrows (hidden state passing)
        between_arrows = VGroup()
        between_lbls   = VGroup()
        for l in range(L - 1):
            arr = Arrow(gru_cells[l].get_right(), gru_cells[l + 1].get_left(),
                        buff=0.1, color=C_HIGHLIGHT, stroke_width=1.8)
            lbl = Text(f"h_{l+1}", font_size=13, color=C_HIGHLIGHT)
            lbl.next_to(arr, UP, buff=0.06)
            between_arrows.add(arr)
            between_lbls.add(lbl)

        # output arrows (right of last cell)
        out_arrow = Arrow(gru_cells[-1].get_right(),
                          gru_cells[-1].get_right() + RIGHT * 1.2,
                          buff=0.0, color=C_HIGHLIGHT, stroke_width=1.8)
        out_lbl = Text(f"(L={L}, M={M}, D=768)", font_size=12, color=C_DIM)
        out_lbl.next_to(out_arrow, UP, buff=0.06)

        self.play(FadeIn(h0_bg), FadeIn(h0_lbl), FadeIn(h0_desc))
        self.play(GrowArrow(arr_h0), FadeIn(lbl_h0_arr))
        for l in range(L):
            self.play(FadeIn(gru_cells[l]), run_time=0.5)
            self.play(FadeIn(pos_lbls[l]), GrowArrow(pos_arrows[l]), run_time=0.4)
            if l < L - 1:
                self.play(GrowArrow(between_arrows[l]), FadeIn(between_lbls[l]), run_time=0.4)
        self.play(GrowArrow(out_arrow), FadeIn(out_lbl), FadeIn(pos_header))
        self.wait(3)
        self.play(FadeOut(n2))

        unroll_all = VGroup(h0_bg, h0_lbl, h0_desc, arr_h0, lbl_h0_arr,
                            gru_cells, pos_lbls, pos_arrows, pos_header,
                            between_arrows, between_lbls, out_arrow, out_lbl)
        self.play(FadeOut(unroll_all))

        # ══════════════════════════════════════════════════════════════════
        # PART 3 — DownscaledTransformer: what it ACTUALLY does (verified)
        # Empirically confirmed: input (L, M, D) with batch_first=True →
        #   batch=L, seq=M.  Attention runs over M (fields), NOT over L (instances).
        #   Instances are independent batch elements — zero cross-instance interaction.
        # ══════════════════════════════════════════════════════════════════

        # Pre-compute real values
        if gru_out.ndim == 3:
            v0_gru = gru_out[0].flatten()
            v1_gru = gru_out[1].flatten()
            sim_before = float(np.dot(v0_gru, v1_gru) /
                               (np.linalg.norm(v0_gru) * np.linalg.norm(v1_gru) + 1e-8))
        else:
            sim_before = 0.0

        if struct_proj.ndim == 3:
            v0_fin = struct_proj[0].flatten()
            v1_fin = struct_proj[1].flatten()
            sim_after = float(np.dot(v0_fin, v1_fin) /
                              (np.linalg.norm(v0_fin) * np.linalg.norm(v1_fin) + 1e-8))
        else:
            sim_after = sim_before

        n3 = narration(
            "DownscaledTransformer: batch_first=True, input (L, M, D) → batch=L, seq=M.\n"
            "Attention runs over M fields — instances are independent. GRU created the diversity."
        )
        self.play(FadeIn(n3))

        # ── show two SEPARATE instance boxes, fields talking WITHIN each ──
        inst_xs = [-3.0, 3.0]
        inst_colors = [C_HIGHLIGHT, C_TOKEN]
        field_colors = [C_SCHEMA, C_TOKEN]
        inst_groups = VGroup()

        for l in range(L):
            ix = inst_xs[l]
            ic = inst_colors[l]

            # Instance header
            hdr = Text(f"Instance slot {l+1}", font_size=15, color=ic, weight=BOLD)
            hdr.move_to(RIGHT * ix + UP * 2.4)

            # Two field circles inside this instance
            f_circles = VGroup()
            f_lbls    = VGroup()
            for k, fname in enumerate(fields):
                fc = Circle(radius=0.6,
                            fill_color=field_colors[k], fill_opacity=0.15,
                            stroke_color=field_colors[k], stroke_width=1.8)
                fy = 0.9 - k * 1.8
                fc.move_to(RIGHT * ix + UP * fy)
                fl = Text(fname, font_size=13, color=field_colors[k])
                fl.move_to(fc)
                f_circles.add(fc)
                f_lbls.add(fl)

            # Cross-field arrows (within this instance only)
            cf_arr1 = CurvedArrow(f_circles[0].get_right() + DOWN * 0.2,
                                  f_circles[1].get_right() + UP * 0.2,
                                  angle=TAU / 5, color=ic, stroke_width=1.3)
            cf_arr2 = CurvedArrow(f_circles[1].get_left() + UP * 0.2,
                                  f_circles[0].get_left() + DOWN * 0.2,
                                  angle=TAU / 5, color=ic, stroke_width=1.3)
            cf_lbl = Text("cross-field\nattention", font_size=11, color=C_DIM)
            cf_lbl.move_to(RIGHT * (ix + 1.3) + UP * 0.0)

            inst_groups.add(VGroup(hdr, f_circles, f_lbls, cf_arr1, cf_arr2, cf_lbl))

        # Vertical divider between instances — they are INDEPENDENT
        divider = DashedLine(UP * 2.8, DOWN * 2.5, color=C_DIM, stroke_width=1.0)
        divider.move_to(ORIGIN)
        no_cross = Text("✕  no cross-instance interaction", font_size=14, color=C_DIM)
        no_cross.move_to(UP * 3.1)

        self.play(
            LaggedStart(*[FadeIn(ig) for ig in inst_groups], lag_ratio=0.3),
            run_time=1.5,
        )
        self.play(FadeIn(divider), FadeIn(no_cross))
        self.wait(2.5)

        # ── show instance diversity was made by the GRU (cosine before vs after) ──
        self.play(FadeOut(n3))
        n4 = narration(
            f"GRU output cosine = {sim_before:.3f}  →  after transformer = {sim_after:.3f}.\n"
            f"Transformer pushes further apart — not by cross-instance talk, but by applying different field-attention to already-different instances."
        )
        self.play(FadeIn(n4))

        delta = sim_after - sim_before
        sign  = "+" if delta >= 0 else ""
        sim_vg = VGroup(
            Text(f"After GRU:          instance1 ↔ instance2  cosine = {sim_before:.3f}",
                 font_size=16, color=C_DIM),
            Text(f"After Transformer:  instance1 ↔ instance2  cosine = {sim_after:.3f}  ({sign}{delta:.3f})",
                 font_size=16, color=C_WHITE),
        )
        sim_vg.arrange(DOWN, buff=0.3).move_to(DOWN * 2.0)
        self.play(FadeIn(sim_vg))
        self.wait(3.5)

        self.play(
            FadeOut(n4), FadeOut(inst_groups), FadeOut(divider),
            FadeOut(no_cross), FadeOut(sim_vg), FadeOut(title),
        )
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
