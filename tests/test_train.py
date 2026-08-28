import math
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.optimizer import AdamW
from src.log import ExperimentLogger
from src.train import cross_entropy, evaluate, gradient_clipping, train
from src.transformer import TransformerLM


class ConstantLogitsModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))
        self.observed_training_modes: list[bool] = []
        self.observed_grad_modes: list[bool] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.observed_training_modes.append(self.training)
        self.observed_grad_modes.append(torch.is_grad_enabled())
        return self.logits.expand(*token_ids.shape, -1)


def test_cross_entropy_matches_pytorch_reference():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    flat_targets = targets.reshape(-1)

    actual = cross_entropy(flat_inputs, flat_targets)
    expected = F.cross_entropy(flat_inputs, flat_targets)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cross_entropy_is_stable_for_large_logits():
    inputs = torch.tensor(
        [[1000.0, 2000.0, 3000.0], [3000.0, 1000.0, 2000.0]]
    )
    targets = torch.tensor([2, 0])

    actual = cross_entropy(inputs, targets)
    expected = F.cross_entropy(inputs, targets)

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)


def test_cross_entropy_returns_scalar_and_backpropagates():
    inputs = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1])

    loss = cross_entropy(inputs, targets)
    loss.backward()

    assert loss.shape == ()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_gradient_clipping_matches_pytorch_reference():
    tensors = [torch.randn(5, 5) for _ in range(6)]
    max_l2_norm = 1e-2

    reference_parameters = tuple(
        torch.nn.Parameter(tensor.clone()) for tensor in tensors
    )
    reference_parameters[-1].requires_grad_(False)
    torch.cat(reference_parameters).sum().backward()
    torch.nn.utils.clip_grad_norm_(reference_parameters, max_l2_norm)

    actual_parameters = tuple(
        torch.nn.Parameter(tensor.clone()) for tensor in tensors
    )
    actual_parameters[-1].requires_grad_(False)
    torch.cat(actual_parameters).sum().backward()
    gradient_clipping(actual_parameters, max_l2_norm)

    expected_grads = [
        parameter.grad
        for parameter in reference_parameters
        if parameter.grad is not None
    ]
    actual_grads = [
        parameter.grad
        for parameter in actual_parameters
        if parameter.grad is not None
    ]
    assert len(actual_grads) == len(expected_grads)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


def test_gradient_clipping_accepts_parameter_generator():
    model = torch.nn.Linear(3, 2)
    model(torch.ones(1, 3)).sum().backward()

    gradient_clipping(model.parameters(), max_l2_norm=0.1)

    total_norm = torch.sqrt(
        sum(
            parameter.grad.norm(2) ** 2
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    assert total_norm <= 0.1


def test_evaluate_computes_loss_without_grad_and_restores_model_mode():
    model = ConstantLogitsModel(vocab_size=3)
    model.train()
    dataset = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

    loss = evaluate(
        model,
        dataset,
        batch_size=2,
        context_length=2,
        device="cpu",
        num_batches=3,
    )

    assert loss == pytest.approx(math.log(3))
    assert model.training is True
    assert model.observed_training_modes == [False, False, False]
    assert model.observed_grad_modes == [False, False, False]
    assert model.logits.grad is None


def test_evaluate_rejects_non_positive_num_batches():
    model = ConstantLogitsModel(vocab_size=3)
    dataset = np.array([0, 1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="positive"):
        evaluate(model, dataset, 1, 1, "cpu", num_batches=0)


def test_train_updates_parameters_and_returns_finite_losses():
    model = ConstantLogitsModel(vocab_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    dataset = np.zeros(8, dtype=np.int64)
    initial_logits = model.logits.detach().clone()

    losses = train(
        model,
        optimizer,
        dataset,
        batch_size=2,
        context_length=2,
        device="cpu",
        max_steps=3,
        max_learning_rate=0.1,
        min_learning_rate=0.1,
        warmup_iters=0,
        cosine_cycle_iters=3,
        max_grad_norm=1.0,
    )

    assert len(losses) == 3
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
    assert not torch.equal(model.logits.detach(), initial_logits)
    assert model.training is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_train_clears_stale_gradients_before_first_step():
    clean_model = ConstantLogitsModel(vocab_size=3)
    stale_model = ConstantLogitsModel(vocab_size=3)
    stale_model.load_state_dict(clean_model.state_dict())
    stale_model.logits.grad = torch.ones_like(stale_model.logits)

    clean_optimizer = torch.optim.SGD(clean_model.parameters(), lr=1.0)
    stale_optimizer = torch.optim.SGD(stale_model.parameters(), lr=1.0)
    dataset = np.zeros(8, dtype=np.int64)
    train_kwargs = {
        "batch_size": 2,
        "context_length": 2,
        "device": "cpu",
        "max_steps": 1,
        "max_learning_rate": 0.1,
        "min_learning_rate": 0.1,
        "warmup_iters": 0,
        "cosine_cycle_iters": 1,
        "max_grad_norm": 10.0,
    }

    train(clean_model, clean_optimizer, dataset, **train_kwargs)
    train(stale_model, stale_optimizer, dataset, **train_kwargs)

    torch.testing.assert_close(stale_model.logits, clean_model.logits)


def test_train_rejects_non_positive_max_steps():
    model = ConstantLogitsModel(vocab_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataset = np.zeros(3, dtype=np.int64)

    with pytest.raises(ValueError, match="positive"):
        train(
            model,
            optimizer,
            dataset,
            batch_size=1,
            context_length=1,
            device="cpu",
            max_steps=0,
            max_learning_rate=0.1,
            min_learning_rate=0.1,
            warmup_iters=0,
            cosine_cycle_iters=1,
            max_grad_norm=1.0,
        )


def test_train_updates_real_transformer_with_finite_gradients():
    torch.manual_seed(42)
    np.random.seed(42)
    model = TransformerLM(
        vocab_size=8,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10_000.0,
        device="cpu",
    )
    optimizer = AdamW(model.parameters(), lr=1e-2)
    dataset = np.tile(np.arange(8, dtype=np.int64), 8)
    initial_embedding = model.token_embeddings.weight.detach().clone()

    losses = train(
        model,
        optimizer,
        dataset,
        batch_size=2,
        context_length=4,
        device="cpu",
        max_steps=2,
        max_learning_rate=1e-2,
        min_learning_rate=1e-2,
        warmup_iters=0,
        cosine_cycle_iters=2,
        max_grad_norm=1.0,
    )

    assert len(losses) == 2
    assert all(math.isfinite(loss) for loss in losses)
    assert not torch.equal(model.token_embeddings.weight, initial_embedding)

    representative_gradients = [
        model.token_embeddings.weight.grad,
        model.layers[0].attn.q_proj.weight.grad,
        model.lm_head.weight.grad,
    ]
    assert all(gradient is not None for gradient in representative_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in representative_gradients
        if gradient is not None
    )


def test_train_logs_completed_steps_and_validation(tmp_path: Path):
    model = ConstantLogitsModel(vocab_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataset = np.zeros(8, dtype=np.int64)
    log_path = tmp_path / "metrics.jsonl"

    with ExperimentLogger(log_path) as logger:
        train(
            model,
            optimizer,
            dataset,
            batch_size=2,
            context_length=2,
            device="cpu",
            max_steps=2,
            max_learning_rate=0.1,
            min_learning_rate=0.1,
            warmup_iters=0,
            cosine_cycle_iters=2,
            max_grad_norm=1.0,
            validation_dataset=dataset,
            logger=logger,
            log_interval=1,
            validation_interval=1,
            validation_batches=1,
        )

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    train_records = [record for record in records if record["event"] == "train"]
    validation_records = [
        record for record in records if record["event"] == "validation"
    ]

    assert [record["step"] for record in train_records] == [1, 2]
    assert [record["tokens_processed"] for record in train_records] == [4, 8]
    assert all("train_loss" in record for record in train_records)
    assert [record["step"] for record in validation_records] == [1, 2]
    assert all("validation_loss" in record for record in validation_records)
    assert all("perplexity" in record for record in validation_records)
