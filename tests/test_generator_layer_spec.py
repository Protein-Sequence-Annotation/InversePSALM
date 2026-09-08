import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from inverse_psalm.models.generator import (
    GENERATOR_CONFIG_NAME,
    GENERATOR_WEIGHTS_NAME,
    InversePSALM,
    parse_annotation_injection_blocks,
)


class _DummyConfig:
    hidden_size = 2
    layer_norm_eps = 1e-5
    vocab_size = 4

    def save_pretrained(self, save_directory):
        Path(save_directory, "config.json").write_text("{}\n", encoding="utf-8")


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.position_embedding_type = "rotary"
        self.embeddings.position_embeddings = nn.Embedding(8, 2)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_DummyLayer()])


class _DummyFAE(nn.Module):
    def __init__(self, config=None, use_fa=True):
        super().__init__()
        self.config = config or _DummyConfig()
        self.use_fa = use_fa
        self.esm = _DummyBase()
        self.lm_head = nn.Linear(2, 4)
        self.linear = nn.Linear(2, 2)

    @classmethod
    def from_pretrained(cls, _model_name, use_fa=True):
        return cls(_DummyConfig(), use_fa=use_fa)


class _DummyTokenizer:
    def save_pretrained(self, save_directory):
        Path(save_directory, "tokenizer_marker.txt").write_text("saved\n", encoding="utf-8")


def test_parse_annotation_injection_blocks_single_int():
    assert parse_annotation_injection_blocks(1, 33) == {1}


def test_parse_annotation_injection_blocks_range():
    assert parse_annotation_injection_blocks("1-3", 33) == {1, 2, 3}


def test_parse_annotation_injection_blocks_mixed():
    assert parse_annotation_injection_blocks("1,17,26-33", 33) == {1, 17, 26, 27, 28, 29, 30, 31, 32, 33}


def test_annotation_gate_init_and_l2():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.annotation_gate_min = 0.0
    gen.annotation_gate_max = 1.0
    gen.annotation_injection_blocks = {1, 2}
    raw = torch.logit(torch.tensor(0.005))
    gen.annotation_gate_logits = nn.ParameterDict(
        {
            "block_01": nn.Parameter(raw.clone()),
            "block_02": nn.Parameter(raw.clone()),
        }
    )

    values = gen.annotation_gate_values()
    assert abs(values["block_01"] - 0.005) < 1e-6
    assert abs(values["block_02"] - 0.005) < 1e-6
    assert torch.isfinite(gen.annotation_gate_l2_loss())
    assert abs(float(gen.annotation_gate_l2_loss().item()) - 0.005**2) < 1e-8


def test_generator_logit_soft_cap_is_noop_when_unset():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.generator_logit_soft_cap = None
    logits = torch.tensor([[-100.0, 0.0, 100.0]])

    capped = gen._apply_logit_soft_cap(logits)

    assert capped is logits


def test_generator_logit_soft_cap_bounds_logits():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.generator_logit_soft_cap = 30.0
    logits = torch.tensor([[-1000.0, 0.0, 1000.0]])

    capped = gen._apply_logit_soft_cap(logits)

    assert torch.all(capped <= 30.0)
    assert torch.all(capped >= -30.0)
    assert capped[0, 1].item() == 0.0


def test_ignored_annotation_positions_do_not_inject_after_norm():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.annotation_gate_min = 0.0
    gen.annotation_gate_max = 1.0
    gen.annotation_gate_logits = nn.ParameterDict({"block_01": nn.Parameter(torch.logit(torch.tensor(0.5)))})
    gen.annotation_delta_norm = nn.LayerNorm(2)
    gen.annotation_delta_norm.bias.data.fill_(3.0)
    gen._active_annotation_embeds_padded = torch.zeros(1, 2, 2)
    gen._active_annotation_embeds_packed = None
    gen._active_annotation_valid_padded = torch.tensor([[False, True]])
    gen._active_annotation_valid_packed = None
    gen._last_effective_injection_norm_ratios = {}
    hidden = torch.zeros(1, 2, 2)

    injected = gen._add_annotation_before_block(1, hidden)

    assert torch.allclose(injected[:, 0], hidden[:, 0])
    assert not torch.allclose(injected[:, 1], hidden[:, 1])


def test_freeze_unused_position_embeddings_for_rotary_models():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.fae = nn.Module()
    gen.fae.esm = nn.Module()
    gen.fae.esm.embeddings = nn.Module()
    gen.fae.esm.embeddings.position_embedding_type = "rotary"
    gen.fae.esm.embeddings.position_embeddings = nn.Embedding(16, 8)

    gen._freeze_unused_position_embeddings()

    assert not gen.fae.esm.embeddings.position_embeddings.weight.requires_grad


def test_keep_absolute_position_embeddings_trainable():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.fae = nn.Module()
    gen.fae.esm = nn.Module()
    gen.fae.esm.embeddings = nn.Module()
    gen.fae.esm.embeddings.position_embedding_type = "absolute"
    gen.fae.esm.embeddings.position_embeddings = nn.Embedding(16, 8)

    gen._freeze_unused_position_embeddings()

    assert gen.fae.esm.embeddings.position_embeddings.weight.requires_grad


def test_set_freeze_state_keeps_rotary_position_embeddings_frozen():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.fae = nn.Module()
    gen.fae.esm = nn.Module()
    gen.fae.esm.embeddings = nn.Module()
    gen.fae.esm.embeddings.position_embedding_type = "rotary"
    gen.fae.esm.embeddings.position_embeddings = nn.Embedding(16, 8)
    gen.fae.lm_head = nn.Linear(8, 16)

    gen.set_freeze_state(freeze_generator_trunk=False, freeze_lm_head=False)

    assert not gen.fae.esm.embeddings.position_embeddings.weight.requires_grad
    assert gen.fae.lm_head.weight.requires_grad


def test_save_pretrained_writes_reloadable_generator_artifact(tmp_path):
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.fae = _DummyFAE()
    gen.tokenizer = _DummyTokenizer()
    gen.model_name = "facebook/esm2_t33_650M_UR50D"
    gen.num_annotation_labels = 10
    gen.annotation_dropout = 0.1
    gen.ignore_label = -100
    gen.generator_model_config = {
        "annotation_injection_blocks": "1",
        "annotation_encoder": {"type": "full_token", "full_token": {"embedding_dim": 2, "mlp_dim": 2}},
    }
    gen.use_fa = True
    gen.annotation = nn.Linear(2, 2)
    gen.annotation_delta_norm = nn.LayerNorm(2)
    gen.annotation_gate_logits = nn.ParameterDict({"block_01": nn.Parameter(torch.zeros(()))})

    gen.save_pretrained(tmp_path)

    artifact_config = json.loads((tmp_path / GENERATOR_CONFIG_NAME).read_text(encoding="utf-8"))
    generator_state = load_file(tmp_path / GENERATOR_WEIGHTS_NAME)
    assert artifact_config["num_annotation_labels"] == 10
    assert "training" not in artifact_config
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "tokenizer_marker.txt").exists()
    assert "fae.linear.weight" in generator_state
    assert "annotation.weight" in generator_state
    assert "annotation_gate_logits.block_01" in generator_state


def test_annotation_ids_like_defaults_to_no_injection():
    gen = InversePSALM.__new__(InversePSALM)
    nn.Module.__init__(gen)
    gen.ignore_label = -100

    annotation_ids = gen.annotation_ids_like(torch.tensor([[1, 2, 3]]))

    assert annotation_ids.tolist() == [[-100, -100, -100]]


def test_from_pretrained_loads_whole_generator_safetensors(tmp_path, monkeypatch):
    import inverse_psalm.models.generator as generator_module

    monkeypatch.setattr(generator_module, "_require_faesm", lambda: _DummyFAE)
    monkeypatch.setattr(generator_module.AutoConfig, "from_pretrained", lambda _path: _DummyConfig())
    monkeypatch.setattr(generator_module.AutoTokenizer, "from_pretrained", lambda _path: _DummyTokenizer())
    config = {
        "model": {
            "annotation_injection_blocks": "1",
            "annotation_encoder": {
                "type": "full_token",
                "full_token": {"embedding_dim": 2, "mlp_dim": 2},
            },
        }
    }
    gen = InversePSALM(
        model_name="dummy-esm",
        num_annotation_labels=5,
        inverse_psalm_config=config,
        freeze_generator_trunk=False,
        freeze_lm_head=False,
        use_fa=False,
    )
    with torch.no_grad():
        gen.annotation.embedding.weight.fill_(1.25)
        gen.fae.linear.weight.fill_(2.5)

    gen.save_pretrained(tmp_path)
    loaded = InversePSALM.from_pretrained(tmp_path, use_fa=False)

    assert torch.allclose(loaded.annotation.embedding.weight, gen.annotation.embedding.weight)
    assert torch.allclose(loaded.fae.linear.weight, gen.fae.linear.weight)
    assert loaded.num_annotation_labels == 5


def test_annotation_track_false_freezes_and_hides_annotation_track(monkeypatch):
    import inverse_psalm.models.generator as generator_module

    monkeypatch.setattr(generator_module, "_require_faesm", lambda: _DummyFAE)
    monkeypatch.setattr(generator_module.AutoTokenizer, "from_pretrained", lambda _path: _DummyTokenizer())
    config = {
        "model": {
            "annotation_track": False,
            "annotation_injection_blocks": "1",
            "annotation_encoder": {
                "type": "full_token",
                "full_token": {"embedding_dim": 2, "mlp_dim": 2},
            },
        }
    }

    gen = InversePSALM(
        model_name="dummy-esm",
        num_annotation_labels=5,
        inverse_psalm_config=config,
        freeze_generator_trunk=False,
        freeze_lm_head=False,
        use_fa=False,
    )

    assert "block_01" in gen.annotation_gate_logits
    assert not any(parameter.requires_grad for parameter in gen.annotation.parameters())
    assert not any(parameter.requires_grad for parameter in gen.annotation_delta_norm.parameters())
    assert not any(parameter.requires_grad for parameter in gen.annotation_gate_logits.parameters())
    assert gen.annotation_gate_values() == {}
    assert gen.annotation_effective_injection_norm_ratios() == {}
    assert gen.annotation_gate_l2_loss().item() == 0.0
