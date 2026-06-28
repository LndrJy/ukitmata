"""LoRA fine-tuning of the vision engine on human-corrected data.

This is the offline half of the self-learning loop. Run it on the training box on a
schedule (e.g. weekly), NOT in the request path. Workflow:

    1. dataset.build_dataset()      -> JSONL of (image, corrected target) pairs
    2. train a LoRA adapter on Qwen2-VL                    [this file]
    3. evaluate the adapter on a held-out set              [evaluate()]
    4. PROMOTE the adapter only if accuracy improved       [gated by you]

Promotion is deliberately manual/gated: never let the model overwrite itself with an
adapter that did not beat the current one. This prevents drift and feedback poisoning.

Requires the training extras:  pip install "ukitmata[train]"

NOTE: This is a working skeleton with the right structure and dependencies wired up.
The training step is left as a guarded stub so it cannot run accidentally on a server
without a GPU and a reviewed dataset. Fill in `train_lora()` when you have collected
enough corrections (a few hundred documents is a reasonable first target).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ukitmata.config import settings
from ukitmata.learning.dataset import build_dataset


def train_lora(dataset_path: Path, epochs: int = 3) -> Path:
    """Fine-tune a LoRA adapter on the corrections dataset.

    Returns the path to the saved adapter. Intentionally guarded — wire in the PEFT
    training loop here once you have a reviewed dataset and a GPU box.
    """
    raise NotImplementedError(
        "Fill in the PEFT/LoRA training loop. Suggested setup:\n"
        "  - load base model: transformers.Qwen2VLForConditionalGeneration\n"
        "  - wrap with peft.LoraConfig(target_modules=['q_proj','v_proj'], r=16)\n"
        "  - dataset: load JSONL, render each image, tokenize the target string\n"
        "  - trainer: transformers.Trainer or trl.SFTTrainer, bf16, grad checkpointing\n"
        f"  - save adapter -> {settings.adapters_dir}"
    )


def evaluate(adapter_path: Path) -> float:
    """Return field-level accuracy of an adapter on a held-out set (0..1).

    Promote the adapter only if this beats the currently deployed one.
    """
    raise NotImplementedError("Implement held-out evaluation before promoting adapters.")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="UkitMata self-learning fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min-examples", type=int, default=200)
    args = parser.parse_args()

    dataset = build_dataset(min_examples=args.min_examples)
    if dataset is None:
        print(f"Not enough corrected documents yet (need >= {args.min_examples}).")
        return
    print(f"Dataset ready: {dataset}")
    print("Training is a guarded stub — see train_lora() docstring to enable.")


if __name__ == "__main__":
    main()
