import numpy as np
import torch

from src.optimizer import AdamW
from src.train import get_lr_cosine_schedule


def _optimize(optimizer_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = optimizer_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(1000):
        optimizer.zero_grad()
        inputs = torch.rand(model.in_features)
        predictions = model(inputs)
        targets = torch.tensor([inputs[0] + inputs[1], -inputs[2]])
        loss = ((targets - predictions) ** 2).sum()
        loss.backward()
        optimizer.step()

    return model.weight.detach()


def test_adamw_matches_pytorch_reference_training():
    expected = _optimize(torch.optim.AdamW)
    actual = _optimize(AdamW)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_get_lr_cosine_schedule_matches_cs336_reference():
    expected = np.array(
        [
            0.0,
            0.14285714285714285,
            0.2857142857142857,
            0.42857142857142855,
            0.5714285714285714,
            0.7142857142857143,
            0.8571428571428571,
            1.0,
            0.9887175604818206,
            0.9554359905560885,
            0.9018241671106134,
            0.8305704108364301,
            0.7452476826029011,
            0.6501344202803414,
            0.55,
            0.44986557971965857,
            0.3547523173970989,
            0.26942958916356996,
            0.19817583288938662,
            0.14456400944391146,
            0.11128243951817937,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
    )
    actual = np.array(
        [
            get_lr_cosine_schedule(
                it=it,
                max_learning_rate=1.0,
                min_learning_rate=0.1,
                warmup_iters=7,
                cosine_cycle_iters=21,
            )
            for it in range(25)
        ]
    )

    np.testing.assert_allclose(actual, expected)
