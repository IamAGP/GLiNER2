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
    from gliner2 import GLiNER2

    print("Loading model for visualisation data...")
    dm = DebugGLiNER2.from_pretrained(MODEL)

    # Hook encoder
    enc_out = {}
    def hook_enc(mod, inp, out):
        enc_out["hs"] = out.last_hidden_state.detach().cpu()
    h = dm.encoder.register_forward_hook(hook_enc)

    _schema_debug.clear()
    _gru_steps.clear()
    _transformer_out.clear()

    dm.extract_json(TEXT, SCHEMA)
    h.remove()

    d = _schema_debug["investment"]

    # Subword tokens
    tok = dm.processor.tokenizer
    enc = tok(TEXT, return_tensors="pt")
    subwords = tok.convert_ids_to_tokens(enc["input_ids"][0])

    _model_data.update({
        "subwords":       subwords,
        "text_tokens":    d["text_tokens"],
        "field_names":    d["field_names"],
        "schema_tokens":  d["schema_tokens"],   # exact token list the model built
        "p_emb_norm":     d["p_emb_norm"],
        "field_embs":     d["field_embs"].numpy(),       # (M, 768)
        "count_logits":   d["count_logits"].numpy(),     # (1, 20)
        "pred_count":     d["pred_count"],
        "span_scores":    d["span_scores"].numpy(),      # (L, M, T, W)
        "struct_proj":    d["struct_proj"].numpy(),      # (L, M, 768)
        "gru_out":        _gru_steps.get("investment", torch.zeros(1)).numpy(),
        "encoder_hs":     enc_out.get("hs", torch.zeros(1, 1, 768)).numpy(),
        "text_len":       d["text_len"],
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

        title = section_title("Scene 2 · DeBERTa Encoder")
        self.add(title)

        n1 = narration("DeBERTa processes the full sequence — schema tokens + text tokens together.")
        self.play(FadeIn(n1))
        self.wait(1.5)

        # ── encoder box ──
        enc_box = RoundedRectangle(
            corner_radius=0.2, width=5, height=2.5,
            fill_color="#1a1a3a", fill_opacity=1,
            stroke_color=C_TOKEN, stroke_width=2,
        ).move_to(ORIGIN)
        enc_label = Text("DeBERTa\n(205M params)", font_size=22, color=C_TOKEN)
        enc_label.move_to(enc_box)
        enc_group = VGroup(enc_box, enc_label)

        self.play(FadeIn(enc_group))
        self.wait(0.5)

        # ── input tokens flowing in ──
        self.play(FadeOut(n1))
        n2 = narration("Every token goes in — and comes out as a 768-dimensional contextual vector.")
        self.play(FadeIn(n2))

        input_labels = ["[P]", "[E]amount", "[E]company", "goldman", "$", "500M", "stripe", "..."]
        in_tokens = VGroup(*[
            Text(t, font_size=14, color=C_SCHEMA if t.startswith("[") else C_WHITE)
            for t in input_labels
        ]).arrange(DOWN, buff=0.18).next_to(enc_box, LEFT, buff=1.2)

        out_bars = VGroup()
        hs = data["encoder_hs"][0]  # (seq_len, 768)
        bar_norms = [float(np.linalg.norm(hs[i])) for i in range(min(8, hs.shape[0]))]
        max_norm  = max(bar_norms) + 1e-8
        for norm in bar_norms:
            h_scaled = (norm / max_norm) * 1.2 + 0.1
            bar = Rectangle(
                width=0.25, height=h_scaled,
                fill_color=C_HIGHLIGHT, fill_opacity=0.8,
                stroke_width=0,
            )
            out_bars.add(bar)
        out_bars.arrange(DOWN, buff=0.05).next_to(enc_box, RIGHT, buff=1.2)
        out_label = Text("768-dim\nvectors", font_size=14, color=C_HIGHLIGHT)
        out_label.next_to(out_bars, RIGHT, buff=0.2)

        arrow_in  = Arrow(in_tokens.get_right(), enc_box.get_left(),  buff=0.1, color=C_DIM)
        arrow_out = Arrow(enc_box.get_right(), out_bars.get_left(), buff=0.1, color=C_DIM)

        self.play(
            LaggedStart(*[FadeIn(t, shift=RIGHT*0.2) for t in in_tokens], lag_ratio=0.1),
            run_time=1.5
        )
        self.play(GrowArrow(arrow_in))
        self.wait(0.5)
        self.play(GrowArrow(arrow_out))
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in out_bars], lag_ratio=0.08),
            FadeIn(out_label),
            run_time=1.5
        )
        self.wait(1)

        # ── key point: contextual ──
        self.play(FadeOut(n2))
        n3 = narration("Unlike word2vec, each vector is CONTEXTUAL — the same word gets\na different vector depending on surrounding words.")
        self.play(FadeIn(n3))
        self.wait(2.5)

        # ── highlight [P] ──
        self.play(FadeOut(n3))
        n4 = narration("The [P] token is special — it attends to everything\nand summarises the whole input for count prediction.")
        self.play(FadeIn(n4))

        p_box = SurroundingRectangle(in_tokens[0], color=C_SPECIAL, buff=0.08, stroke_width=2)
        self.play(Create(p_box))

        # draw attention lines from [P] to all others
        attention_lines = VGroup(*[
            Line(
                in_tokens[0].get_right(),
                in_tokens[i].get_right() + RIGHT * 0.1,
                stroke_width=1,
                stroke_opacity=0.4,
                color=C_SPECIAL,
            )
            for i in range(1, len(input_labels))
        ])
        self.play(
            LaggedStart(*[Create(l) for l in attention_lines], lag_ratio=0.1),
            run_time=1.2
        )
        self.wait(2)
        self.play(FadeOut(n4), FadeOut(title))
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
