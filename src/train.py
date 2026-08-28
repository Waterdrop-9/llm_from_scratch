import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO, IO

import numpy as np
import torch

from .data_loader import get_batch
from .log import Logger
from .optimizer import AdamW
from .transformer import TransformerLM

def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the log-sum-exp of a tensor along a specified dimension.

    Args:
        x:
            Input tensor.
        dim:
            Dimension along which to compute the log-sum-exp.
    Returns:
        A tensor containing the log-sum-exp values, with the reduced dimension kept.
    """
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    return max_vals + torch.log(torch.sum(torch.exp(x - max_vals), dim=dim, keepdim=True))


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean cross-entropy loss.

    Args:
        inputs:
            Unnormalized logits with shape [N, V].
            N is the number of prediction positions.
            V is the vocabulary size.
        targets:
            Correct token IDs with shape [N].

    Returns:
        A scalar tensor containing the mean cross-entropy loss.
    """
    log_partition = logsumexp(inputs, dim=-1).squeeze(-1)
    row_indices = torch.arange(inputs.shape[0], device=inputs.device)
    correct_logits = inputs[row_indices, targets]
    return torch.mean(log_partition - correct_logits)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clip the gradients of the given parameters to have a maximum L2 norm.

    Args:
        parameters:
            An iterable of torch.nn.Parameter objects whose gradients will be clipped.
        max_l2_norm:
            The maximum allowed L2 norm for the gradients.
    """
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return
    squared_norms = torch.stack(
        [torch.sum(grad ** 2) for grad in grads]
    )
    total_norm = torch.sqrt(squared_norms.sum())
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Compute the learning rate using cosine decay with linear warmup."""
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it <= cosine_cycle_iters:
        cycle_position = (it - warmup_iters) / (
            cosine_cycle_iters - warmup_iters
        )
        return min_learning_rate + 0.5 * (
            max_learning_rate - min_learning_rate
        ) * (1 + math.cos(math.pi * cycle_position))
    return min_learning_rate


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model, optimizer, and training progress."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model, optimizer, and return the saved iteration."""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


# training loop
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    
    if num_batches <= 0:
        raise ValueError("num_batches must be a positive integer.")

    was_training = model.training
    model.eval()

    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(
                dataset,
                batch_size,
                context_length,
                device
            )

            logits = model(inputs)
            vocab_size = logits.shape[-1]

            flat_logits = logits.reshape(-1, vocab_size)
            flat_targets = targets.reshape(-1)

            loss_fn = cross_entropy
            loss = loss_fn(flat_logits, flat_targets)

            losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses)

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    max_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    max_grad_norm: float,
    validation_dataset: np.ndarray | None = None,
    logger: Logger | None = None,
    log_interval: int = 10,
    validation_interval: int = 100,
    validation_batches: int = 10,
) -> list[float]:
    """
    Train the model using the provided dataset and hyperparameters.

    Args:
        model:
            The neural network model to be trained.
        optimizer:
            The optimizer used for updating model parameters.
        train_dataset:
            The training dataset as a NumPy array.
        batch_size:
            Number of samples per batch.
        context_length:
            Length of the context window for each sample.
        device:
            Device to run the training on (e.g., 'cpu' or 'cuda').
        max_steps:
            Total number of training steps.
        max_learning_rate:
            Maximum learning rate for the cosine schedule.
        min_learning_rate:
            Minimum learning rate for the cosine schedule.
        warmup_iters:
            Number of iterations for linear warmup.
        cosine_cycle_iters:
            Total iterations for one cosine cycle.
        max_grad_norm:
            Maximum L2 norm for gradient clipping.
        validation_dataset:
            Optional validation token stream. Validation runs only when both
            this dataset and logger are provided.
        logger:
            Optional JSONL experiment logger.
        log_interval:
            Number of completed optimizer steps between training records.
        validation_interval:
            Number of completed optimizer steps between validation records.
        validation_batches:
            Number of random validation batches used for each validation loss.
    Returns:
        A list of loss values recorded at each training step.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if logger is not None and log_interval <= 0:
        raise ValueError("log_interval must be a positive integer.")
    if logger is not None and validation_dataset is not None:
        if validation_interval <= 0:
            raise ValueError("validation_interval must be a positive integer.")
        if validation_batches <= 0:
            raise ValueError("validation_batches must be a positive integer.")
    model.train()
    loss_fn = cross_entropy
    losses: list[float] = []

    for step in range(max_steps):
        lr = get_lr_cosine_schedule(
            step,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- foward pass ---
        # 清空grad，防止每一步的grad累加
        optimizer.zero_grad()
        

        inputs, targets = get_batch(
            train_dataset,
            batch_size,
            context_length,
            device
        )

        logits = model(inputs)
        vocab_size = logits.shape[-1]

        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)

        loss = loss_fn(flat_logits, flat_targets)

        #--- backpropagation and optimization ---
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        completed_step = step + 1
        tokens_processed = completed_step * batch_size * context_length

        if logger is not None and (
            completed_step == 1
            or completed_step % log_interval == 0
            or completed_step == max_steps
        ):
            logger.log(
                "train",
                step=completed_step,
                tokens_processed=tokens_processed,
                train_loss=loss_value,
                learning_rate=lr,
            )

        if (
            logger is not None
            and validation_dataset is not None
            and (
                completed_step % validation_interval == 0
                or completed_step == max_steps
            )
        ):
            validation_loss = evaluate(
                model=model,
                dataset=validation_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                num_batches=validation_batches,
            )
            try:
                perplexity = math.exp(validation_loss)
            except OverflowError:
                perplexity = float("inf")
            logger.log(
                "validation",
                step=completed_step,
                tokens_processed=tokens_processed,
                validation_loss=validation_loss,
                perplexity=perplexity,
            )
    return losses

def main():
    # Set a fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'mps'

    model_config = {
        "vocab_size": 8,
        "context_length": 8,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 64,
        "rope_theta": 10_000.0,
    }
    train_dataset = np.tile(
        np.arange(model_config["vocab_size"], dtype=np.int64),
        128,
    )

    model = TransformerLM(**model_config, device=device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    logger = Logger("artifacts/day4_smoke_metrics.jsonl")
    logger.log(
        "config",
        seed=42,
        device=device,
        model_config=model_config,
        batch_size=4,
        max_steps=200,
        max_learning_rate=1e-2,
        min_learning_rate=1e-3,
        warmup_iters=10,
        max_grad_norm=1.0,
    )

    initial_validation_loss = evaluate(
        model,
        train_dataset,
        batch_size=4,
        context_length=model_config["context_length"],
        device=device,
        num_batches=10,
    )
    logger.log(
        "validation",
        step=0,
        tokens_processed=0,
        validation_loss=initial_validation_loss,
        perplexity=math.exp(initial_validation_loss),
    )

    losses = train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        batch_size=4,
        context_length=model_config["context_length"],
        device=device,
        max_steps=200,
        max_learning_rate=1e-2,
        min_learning_rate=1e-3,
        warmup_iters=10,
        cosine_cycle_iters=200,
        max_grad_norm=1.0,
        validation_dataset=train_dataset,
        logger=logger,
        log_interval=20,
        validation_interval=50,
        validation_batches=10,
    )

    final_validation_loss = evaluate(
        model=model,
        dataset=train_dataset,
        batch_size=4,
        context_length=model_config["context_length"],
        device=device,
        num_batches=10,
    )

    print(f"Initial validation loss: {initial_validation_loss:.4f}")
    print(f"First training loss:     {losses[0]:.4f}")
    print(f"Final training loss:     {losses[-1]:.4f}")
    print(f"Final validation loss:   {final_validation_loss:.4f}")
    logger.log(
        "summary",
        step=len(losses),
        tokens_processed=len(losses) * 4 * model_config["context_length"],
        first_train_loss=losses[0],
        final_train_loss=losses[-1],
        initial_validation_loss=initial_validation_loss,
        final_validation_loss=final_validation_loss,
    )

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert losses[-1] < losses[0]
    assert final_validation_loss < initial_validation_loss

    checkpoint_directory = Path("artifacts")
    checkpoint_directory.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_directory / "day4_smoke.pt"

    model.eval()
    probe = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits_before_save = model(probe)

    save_checkpoint(
        model,
        optimizer,
        iteration=len(losses),
        out=checkpoint_path,
    )
    logger.log(
        "checkpoint",
        step=len(losses),
        path=str(checkpoint_path),
    )

    restored_model = TransformerLM(
        **model_config,
        device=device,
    )
    restored_optimizer = AdamW(
        restored_model.parameters(),
        lr=1e-2,
        weight_decay=0.01,
    )

    restored_iteration = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
    )

    restored_model.eval()

    with torch.no_grad():
        logits_after_load = restored_model(probe)

    assert restored_iteration == len(losses)
    torch.testing.assert_close(
        logits_after_load,
        logits_before_save,
    )

    print(f"Checkpoint restored at step: {restored_iteration}")
    print(f"Checkpoint path: {checkpoint_path}")
    logger.log(
        "checkpoint_restored",
        step=restored_iteration,
        path=str(checkpoint_path),
    )
    logger.close()


if __name__ == "__main__":
    main()
