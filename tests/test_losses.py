import torch

from inverse_psalm.training.losses import masked_cross_entropy


def test_masked_cross_entropy_all_ignored_is_finite_with_nonfinite_logits():
    logits = torch.tensor([[[float("nan"), float("inf"), -float("inf")]]], requires_grad=True)
    labels = torch.tensor([[-100]])

    loss = masked_cross_entropy(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
