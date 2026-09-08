import pytest
import torch
import os


def test_generator_gating_disabled_parity_tiny_model():
    if os.environ.get("RUN_FAESM_PARITY_TEST") != "1":
        pytest.skip("Set RUN_FAESM_PARITY_TEST=1 to run the FAESM parity test.")
    pytest.importorskip("faesm")
    pytest.importorskip("transformers")

    from inverse_psalm.models.generator import InversePSALM

    model_name = "facebook/esm2_t6_8M_UR50D"
    gen = InversePSALM(
        model_name=model_name,
        num_annotation_labels=4,
        inverse_psalm_config={"model": {"annotation_gating": {"enabled": False}}},
        freeze_generator_trunk=True,
        freeze_lm_head=True,
        use_fa=False,
    )
    gen.eval()
    toks = gen.tokenizer("ACDE", return_tensors="pt")
    annotation_ids = torch.zeros_like(toks["input_ids"])
    with torch.no_grad():
        annotated = gen(
            input_ids=toks["input_ids"],
            annotation_ids=annotation_ids,
            attention_mask=toks["attention_mask"],
        )["logits"]
        base = gen.fae(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
        )["logits"]
    assert torch.allclose(annotated, base, atol=1e-5, rtol=1e-4)
