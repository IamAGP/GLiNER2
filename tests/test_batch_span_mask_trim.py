"""
Test: span_mask and spans_idx are trimmed to match span_rep in batched computation.

Regression test for the bug where compute_span_rep_batched returned untrimmed
span_mask/spans_idx (padded to max_text_len * max_width) while span_rep was
correctly trimmed to each sample's actual text_len. This caused tensor shape
mismatches in compute_struct_loss when batch_size > 1 and samples had different
text lengths.

Run with: pytest tests/test_batch_span_mask_trim.py -v
"""

import pytest
import torch

from gliner2 import GLiNER2


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    m = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    m.eval()
    return m


class TestSpanMaskTrim:
    """Verify span_mask and spans_idx are trimmed to match span_rep per sample."""

    def test_batched_span_mask_matches_span_rep(self, model):
        """span_mask length == text_len * max_width for each sample in a batch."""
        device = next(model.parameters()).device
        max_width = model.max_width

        # Variable lengths to trigger different padding amounts
        lengths = [3, 12, 7, 1]
        embs_list = [
            torch.randn(l, model.hidden_size, device=device) for l in lengths
        ]

        results = model.compute_span_rep_batched(embs_list)

        assert len(results) == len(lengths)
        for i, (result, tl) in enumerate(zip(results, lengths)):
            expected_spans = tl * max_width
            span_rep = result["span_rep"]
            span_mask = result["span_mask"]
            spans_idx = result["spans_idx"]

            # span_rep shape: (text_len, max_width, hidden_size)
            assert span_rep.shape[0] == tl, (
                f"Sample {i}: span_rep dim0 should be {tl}, got {span_rep.shape[0]}"
            )

            # span_mask shape: (1, n_spans) — n_spans must equal text_len * max_width
            assert span_mask.shape[1] == expected_spans, (
                f"Sample {i}: span_mask should have {expected_spans} spans, "
                f"got {span_mask.shape[1]}"
            )

            # spans_idx shape: (1, n_spans, 2) — must also match
            assert spans_idx.shape[1] == expected_spans, (
                f"Sample {i}: spans_idx should have {expected_spans} spans, "
                f"got {spans_idx.shape[1]}"
            )

    def test_batched_span_mask_consistent_with_single(self, model):
        """Batched span_mask/spans_idx shapes match single-sample computation."""
        device = next(model.parameters()).device

        lengths = [5, 15, 2]
        embs_list = [
            torch.randn(l, model.hidden_size, device=device) for l in lengths
        ]

        batched_results = model.compute_span_rep_batched(embs_list)
        single_results = [model.compute_span_rep(e) for e in embs_list]

        for i in range(len(lengths)):
            b_mask = batched_results[i]["span_mask"]
            s_mask = single_results[i]["span_mask"]
            assert b_mask.shape == s_mask.shape, (
                f"Sample {i}: batched span_mask shape {b_mask.shape} != "
                f"single span_mask shape {s_mask.shape}"
            )

            b_idx = batched_results[i]["spans_idx"]
            s_idx = single_results[i]["spans_idx"]
            assert b_idx.shape == s_idx.shape, (
                f"Sample {i}: batched spans_idx shape {b_idx.shape} != "
                f"single spans_idx shape {s_idx.shape}"
            )

    def test_uniform_lengths_no_mismatch(self, model):
        """When all samples have the same length, no trimming issue arises."""
        device = next(model.parameters()).device
        max_width = model.max_width

        length = 8
        embs_list = [
            torch.randn(length, model.hidden_size, device=device)
            for _ in range(4)
        ]

        results = model.compute_span_rep_batched(embs_list)

        for i, result in enumerate(results):
            expected_spans = length * max_width
            assert result["span_mask"].shape[1] == expected_spans
            assert result["spans_idx"].shape[1] == expected_spans
