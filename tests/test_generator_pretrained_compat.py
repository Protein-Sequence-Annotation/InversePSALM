import torch

from inverse_psalm.models.generator import InversePSALM


class _DummyEmbeddings:
    def __init__(self):
        self.position_embeddings = torch.nn.Embedding(4098, 8)


class _DummyBase:
    def __init__(self):
        self.embeddings = _DummyEmbeddings()


class _DummyConfig:
    position_embedding_type = "rotary"


class _DummyGenerator:
    def __init__(self):
        self.fae = type("DummyFae", (), {"config": _DummyConfig()})()
        self._base = _DummyBase()

    @property
    def base_model(self):
        return self._base


class _DummyLmHead:
    def __init__(self, tied_weight):
        self.decoder = type("DummyDecoder", (), {"weight": tied_weight})()


class _DummyTiedGenerator(_DummyGenerator):
    def __init__(self):
        super().__init__()
        self.base_model.embeddings.word_embeddings = torch.nn.Embedding(33, 8)
        self._lm_head = _DummyLmHead(self.base_model.embeddings.word_embeddings.weight)

    @property
    def lm_head(self):
        return self._lm_head


def test_drop_legacy_rotary_position_embedding_mismatch_keeps_runtime_shape():
    state_dict = {"fae.esm.embeddings.position_embeddings.weight": torch.zeros(1026, 8)}
    generator = _DummyGenerator()

    ignored = InversePSALM._drop_legacy_rotary_position_embedding_mismatch(generator, state_dict)

    assert ignored == {"fae.esm.embeddings.position_embeddings.weight"}
    assert "fae.esm.embeddings.position_embeddings.weight" not in state_dict
    assert tuple(generator.base_model.embeddings.position_embeddings.weight.shape) == (4098, 8)


def test_expected_missing_tied_lm_head_weight_is_ignored_only_when_tied():
    tied_generator = _DummyTiedGenerator()

    ignored = InversePSALM._expected_missing_tied_weight_keys(
        tied_generator,
        ["fae.lm_head.decoder.weight"],
    )

    assert ignored == {"fae.lm_head.decoder.weight"}
