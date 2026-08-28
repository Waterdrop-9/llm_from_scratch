from pathlib import Path

import torch

from src.optimizer import AdamW
from src.train import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_iteration(tmp_path: Path):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, iteration=3, out=checkpoint_path)

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    restored_iteration = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
    )

    assert restored_iteration == 3
    for actual, expected in zip(
        restored_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    actual_state = restored_optimizer.state_dict()
    expected_state = optimizer.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]
    for parameter_id in expected_state["state"]:
        expected_parameter_state = expected_state["state"][parameter_id]
        actual_parameter_state = actual_state["state"][parameter_id]
        assert actual_parameter_state["step"] == expected_parameter_state["step"]
        torch.testing.assert_close(
            actual_parameter_state["exp_avg"],
            expected_parameter_state["exp_avg"],
        )
        torch.testing.assert_close(
            actual_parameter_state["exp_avg_sq"],
            expected_parameter_state["exp_avg_sq"],
        )
