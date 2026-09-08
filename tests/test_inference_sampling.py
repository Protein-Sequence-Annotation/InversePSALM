import pytest
import torch

from inverse_psalm.inference.design import _resolve_dtype
from inverse_psalm.inference.sampling import (
    resolve_step_size,
    sample_amino_acids,
    select_positions,
    token_probabilities,
)


def test_resolve_step_size_from_aa_or_fraction():
    assert resolve_step_size(length=100, step_aa=12) == 12
    assert resolve_step_size(length=100, step_frac=0.05) == 5
    assert resolve_step_size(length=3, step_frac=0.01) == 1

    with pytest.raises(ValueError, match="exactly one"):
        resolve_step_size(length=100, step_aa=10, step_frac=0.1)


def test_position_confidence_and_entropy_choose_expected_positions():
    logits = torch.zeros(5, 8)
    allowed = [4, 5, 6]
    masked = torch.tensor([False, True, True, True, False])
    logits[1, 4] = 5.0
    logits[2, 4:7] = torch.tensor([1.0, 1.0, 1.0])
    logits[3, 5] = 4.0

    confident = select_positions(
        logits,
        masked,
        count=2,
        allowed_token_ids=allowed,
        policy="confidence",
    )
    assert confident.tolist() == [1, 3]

    low_entropy = select_positions(
        logits,
        masked,
        count=1,
        allowed_token_ids=allowed,
        policy="entropy",
    )
    assert low_entropy.tolist() == [1]

    uncertain = select_positions(
        logits,
        masked,
        count=1,
        allowed_token_ids=allowed,
        policy="confidence",
        uncertain=True,
    )
    assert uncertain.tolist() == [2]


def test_position_random_is_seeded():
    logits = torch.zeros(6, 8)
    masked = torch.tensor([False, True, True, True, True, False])
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)

    first = select_positions(logits, masked, count=3, allowed_token_ids=[4, 5], policy="random", generator=gen_a)
    second = select_positions(logits, masked, count=3, allowed_token_ids=[4, 5], policy="random", generator=gen_b)
    assert first.tolist() == second.tolist()


def test_amino_acid_argmax_and_top_p_sampling():
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 4.0, 0.0],
        ]
    )
    allowed = [4, 5, 6]

    assert sample_amino_acids(logits, allowed_token_ids=allowed, policy="argmax").tolist() == [4, 5]
    assert sample_amino_acids(
        logits,
        allowed_token_ids=allowed,
        policy="top-p",
        top_p=0.01,
        generator=torch.Generator().manual_seed(1),
    ).tolist() == [4, 5]
    assert sample_amino_acids(
        logits,
        allowed_token_ids=allowed,
        policy="min-p",
        min_p=0.99,
        generator=torch.Generator().manual_seed(1),
    ).tolist() == [4, 5]


def test_top_k_requires_k():
    logits = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="top-k"):
        sample_amino_acids(logits, allowed_token_ids=[4, 5], policy="top-k")


def test_token_probabilities_for_allowed_ids():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0, 1.0]])
    probs = token_probabilities(
        logits,
        allowed_token_ids=[4, 5],
        token_ids=torch.tensor([4]),
    )
    assert probs.item() > 0.5


def test_resolve_dtype_auto_and_cuda_fp32_guard():
    assert _resolve_dtype("auto", torch.device("cpu")) == torch.float32
    assert _resolve_dtype("auto", torch.device("cuda")) == torch.bfloat16
    assert _resolve_dtype("bf16", torch.device("cuda")) == torch.bfloat16
    with pytest.raises(ValueError, match="FlashAttention"):
        _resolve_dtype("fp32", torch.device("cuda"))
