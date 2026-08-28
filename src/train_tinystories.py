"""Single command-line entry point for preparing and training TinyStories."""

import argparse
import math
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .log import Logger
from .optimizer import AdamW
from .prepare_data import load_tokenized_dataset, tokenize_file_to_uint16
from .tokenizer import Tokenizer
from .train import evaluate, save_checkpoint, train
from .train_bpe import train_bpe
from .transformer import TransformerLM


def _save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(
            {
                "vocab": tokenizer.vocab,
                "merges": list(tokenizer.merge_to_rank),
                "special_tokens": [
                    token.decode("utf-8") for token in tokenizer.special_tokens
                ],
            },
            stream,
        )


def _load_tokenizer(path: Path) -> Tokenizer:
    # Only load tokenizer pickle files produced locally by this project.
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return Tokenizer(
        vocab=payload["vocab"],
        merges=payload["merges"],
        special_tokens=payload["special_tokens"],
    )


def _run_prepare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.tokenizer_path.exists():
        tokenizer = _load_tokenizer(args.tokenizer_path)
        tokenizer_action = "loaded"
    else:
        if args.tokenizer_corpus is None:
            parser.error(
                "--tokenizer-corpus is required when --tokenizer-path does not exist"
            )
        special_tokens = args.special_tokens or ["<|endoftext|>"]
        vocab, merges = train_bpe(
            input_path=args.tokenizer_corpus,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
        )
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        _save_tokenizer(tokenizer, args.tokenizer_path)
        tokenizer_action = "trained"

    token_count = tokenize_file_to_uint16(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        buffer_size=args.buffer_size,
    )
    print(
        f"Tokenizer {tokenizer_action}: {args.tokenizer_path}\n"
        f"Encoded {token_count} tokens: {args.output}"
    )


def _run_training(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available; pass --device cpu instead")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        parser.error("--warmup-steps must be non-negative and smaller than --max-steps")

    run_directory = args.output_root / args.run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    metrics_path = run_directory / "metrics.jsonl"
    checkpoint_path = run_directory / "checkpoint.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = load_tokenized_dataset(args.train_data)
    valid_dataset = load_tokenized_dataset(args.valid_data)
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = TransformerLM(**model_config, device=args.device)
    optimizer = AdamW(model.parameters(), lr=args.max_learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    with Logger(metrics_path) as logger:
        logger.log(
            "config",
            run_name=args.run_name,
            seed=args.seed,
            device=args.device,
            model_config=model_config,
            parameter_count=parameter_count,
            train_tokens=len(train_dataset),
            valid_tokens=len(valid_dataset),
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
        )

        initial_validation_loss = evaluate(
            model=model,
            dataset=valid_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            num_batches=args.validation_batches,
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
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            max_steps=args.max_steps,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.max_steps,
            max_grad_norm=args.max_grad_norm,
            validation_dataset=valid_dataset,
            logger=logger,
            log_interval=args.log_interval,
            validation_interval=args.validation_interval,
            validation_batches=args.validation_batches,
        )

        final_validation_loss = evaluate(
            model=model,
            dataset=valid_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            num_batches=args.validation_batches,
        )
        tokens_processed = args.max_steps * args.batch_size * args.context_length
        logger.log(
            "summary",
            step=args.max_steps,
            tokens_processed=tokens_processed,
            first_train_loss=losses[0],
            final_train_loss=losses[-1],
            initial_validation_loss=initial_validation_loss,
            final_validation_loss=final_validation_loss,
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=args.max_steps,
            out=checkpoint_path,
        )
        logger.log(
            "checkpoint",
            step=args.max_steps,
            path=str(checkpoint_path),
        )

    print(f"Run complete: {run_directory}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data and train TinyLLM on TinyStories."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare",
        help="train/load a tokenizer and encode one text file",
    )
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer-path", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer-corpus", type=Path)
    prepare_parser.add_argument("--vocab-size", type=int, default=1_000)
    prepare_parser.add_argument("--buffer-size", type=int, default=1_000_000)
    prepare_parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
    )

    train_parser = commands.add_parser(
        "train",
        help="train TinyLLM from prepared uint16 data",
    )
    train_parser.add_argument("--run-name", required=True)
    train_parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/data/pilot/train.bin"),
    )
    train_parser.add_argument(
        "--valid-data",
        type=Path,
        default=Path("artifacts/data/pilot/valid.bin"),
    )
    train_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs"),
    )
    train_parser.add_argument("--device", default="mps")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--vocab-size", type=int, default=1_000)
    train_parser.add_argument("--context-length", type=int, default=128)
    train_parser.add_argument("--d-model", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--d-ff", type=int, default=384)
    train_parser.add_argument("--rope-theta", type=float, default=10_000.0)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--max-steps", type=int, default=1_000)
    train_parser.add_argument("--max-learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=50)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--validation-interval", type=int, default=100)
    train_parser.add_argument("--validation-batches", type=int, default=10)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        _run_prepare(args, parser)
    else:
        _run_training(args, parser)


if __name__ == "__main__":
    main()
