import torch
import torch.nn as nn

from inverse_psalm.models.annotation_head import (
    AnnotationEncoderA,
    AnnotationEncoderB,
    build_annotation_encoder,
    decode_psalm_labels,
)


def test_annotation_encoder_a_shape_and_ignore_zeroing():
    encoder = AnnotationEncoderA(
        vocab_size=10,
        hidden_size=8,
        embedding_dim=4,
        mlp_dim=12,
        ignore_label=-100,
    )
    out = encoder(torch.tensor([[0, 1, -100, 9]]))
    assert out.shape == (1, 4, 8)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 2], torch.zeros_like(out[0, 2]))
    assert not any(isinstance(module, nn.LayerNorm) for module in encoder.proj.modules())


def test_decode_psalm_labels():
    family_ids, state_types = decode_psalm_labels(torch.tensor([[0, 1, 2, 3, 4, 5, 6, -100]]))
    assert family_ids.tolist() == [[0, 1, 1, 1, 2, 2, 2, 0]]
    assert state_types.tolist() == [[0, 1, 2, 3, 1, 2, 3, 0]]


def test_annotation_encoder_b_shape_and_ignore_zeroing():
    encoder = AnnotationEncoderB(
        num_labels=7,
        hidden_size=8,
        family_embedding_dim=4,
        state_embedding_dim=2,
        mlp_dim=12,
        ignore_label=-100,
    )
    out = encoder(torch.tensor([[0, 1, 6, -100]]))
    assert out.shape == (1, 4, 8)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 3], torch.zeros_like(out[0, 3]))
    assert not any(isinstance(module, nn.LayerNorm) for module in encoder.proj.modules())


def test_build_annotation_encoder_from_config():
    config = {
        "model": {
            "annotation_encoder": {
                "type": "factorized",
                "factorized": {
                    "family_embedding_dim": 4,
                    "state_embedding_dim": 2,
                    "mlp_dim": 12,
                },
            }
        }
    }
    encoder = build_annotation_encoder(
        config=config,
        num_labels=7,
        hidden_size=8,
        dropout=0.0,
        ignore_label=-100,
    )
    assert isinstance(encoder, AnnotationEncoderB)
