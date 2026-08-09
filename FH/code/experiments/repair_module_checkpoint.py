"""Remove fixed sparse operators from a legacy module-ablation checkpoint."""

import argparse
import os
from pathlib import Path

import torch


def load_without_sparse_validation(path):
    validator = torch._utils._validate_loaded_sparse_tensors
    try:
        torch._utils._validate_loaded_sparse_tensors = lambda: None
        return torch.load(path, map_location="cpu")
    finally:
        torch._utils._validate_loaded_sparse_tensors = validator
        # The bypassed validator normally clears this private deserialization
        # queue in its ``finally`` block.  Clear it explicitly so a subsequent
        # ordinary load does not revalidate tensors removed from the payload.
        torch._utils._sparse_tensors_to_validate.clear()


DROP = object()


def remove_sparse_tensors(value, path, removed):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            removed.append(path)
            return DROP
        return value
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            result = remove_sparse_tensors(item, f"{path}.{key}" if path else str(key), removed)
            if result is not DROP:
                cleaned[key] = result
        return cleaned
    if isinstance(value, list):
        return [
            result
            for index, item in enumerate(value)
            if (result := remove_sparse_tensors(item, f"{path}[{index}]", removed)) is not DROP
        ]
    if isinstance(value, tuple):
        return tuple(
            result
            for index, item in enumerate(value)
            if (result := remove_sparse_tensors(item, f"{path}[{index}]", removed)) is not DROP
        )
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_checkpoint")
    parser.add_argument("output_checkpoint")
    args = parser.parse_args()

    source = Path(args.input_checkpoint)
    destination = Path(args.output_checkpoint)
    payload = load_without_sparse_validation(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("expected a training checkpoint containing a model state dictionary")

    removed = []
    payload = remove_sparse_tensors(payload, "", removed)
    if not removed:
        raise ValueError("checkpoint contains no sparse model buffers to remove")
    payload["removed_fixed_sparse_buffers"] = removed

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"removed fixed sparse buffers: {removed}", flush=True)
    verified = torch.load(destination, map_location="cpu")
    print(f"verified checkpoint: {destination}")
    print(f"epoch={verified.get('epoch')} best_val_psnr={verified.get('best_val_psnr')}")


if __name__ == "__main__":
    main()
