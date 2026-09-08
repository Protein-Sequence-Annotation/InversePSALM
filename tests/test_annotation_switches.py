import pytest
import torch.nn as nn

from inverse_psalm.models.inverse_psalm import build_inverse_psalm


class _DummyGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_annotation_labels = int(kwargs["num_annotation_labels"])
        self.tokenizer = object()


class _DummyVerifier(nn.Module):
    num_labels = 17


def test_annotation_loss_false_skips_verifier_build(monkeypatch):
    import inverse_psalm.models.inverse_psalm as inverse_psalm_module

    def fail_build_verifier(_config):
        raise AssertionError("verifier should not be built")

    monkeypatch.setattr(inverse_psalm_module, "InversePSALM", _DummyGenerator)
    monkeypatch.setattr(inverse_psalm_module, "build_verifier_from_config", fail_build_verifier)

    model, _tokenizer = build_inverse_psalm(
        {
            "model": {
                "annotation_track": True,
                "annotation_loss": False,
                "annotation_vocab_size": 72229,
            }
        }
    )

    assert model.verifier is None
    assert model.generator.num_annotation_labels == 72229


def test_old_style_config_still_builds_verifier_and_infers_num_labels(monkeypatch):
    import inverse_psalm.models.inverse_psalm as inverse_psalm_module

    monkeypatch.setattr(inverse_psalm_module, "InversePSALM", _DummyGenerator)
    monkeypatch.setattr(inverse_psalm_module, "build_verifier_from_config", lambda _config: _DummyVerifier())

    model, _tokenizer = build_inverse_psalm({"model": {}})

    assert isinstance(model.verifier, _DummyVerifier)
    assert model.generator.num_annotation_labels == _DummyVerifier.num_labels


def test_annotation_loss_requires_annotation_track():
    with pytest.raises(ValueError, match="annotation_loss=true requires model.annotation_track=true"):
        build_inverse_psalm(
            {
                "model": {
                    "annotation_track": False,
                    "annotation_loss": True,
                    "annotation_vocab_size": 72229,
                }
            }
        )
