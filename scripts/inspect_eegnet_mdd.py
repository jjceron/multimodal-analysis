from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.mdd_db import MDDDataset, parse_optional_float
from src.models.eegnet import EEGNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def module_param_count(module: nn.Module):
    total = sum(p.numel() for p in module.parameters(recurse=False))
    trainable = sum(
        p.numel()
        for p in module.parameters(recurse=False)
        if p.requires_grad
    )
    non_trainable = total - trainable
    return total, trainable, non_trainable


def model_size_mb(model: nn.Module):
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)


def collate_list(batch):
    names, X, y = zip(*batch)
    return list(names), list(X), torch.stack(y)


def standardize_eeg(X, eps: float = 1e-6):
    if isinstance(X, torch.Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (X - mean) / std

    return [standardize_eeg(x, eps=eps) for x in X]


def shape_of(obj):
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)

    if isinstance(obj, (list, tuple)):
        return [shape_of(x) for x in obj]

    return str(type(obj))


def register_shape_hooks(model: nn.Module):
    records = []

    hook_types = (
        nn.Conv2d,
        nn.AvgPool2d,
        nn.Dropout2d,
        nn.BatchNorm2d,
        nn.GroupNorm,
        nn.ELU,
    )

    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            total_params, trainable_params, non_trainable_params = module_param_count(
                module
            )

            records.append(
                {
                    "name": name,
                    "module": module.__class__.__name__,
                    "input_shape": shape_of(inputs),
                    "output_shape": shape_of(output),
                    "params": total_params,
                    "trainable_params": trainable_params,
                    "non_trainable_params": non_trainable_params,
                }
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, hook_types):
            handles.append(module.register_forward_hook(make_hook(name)))

    return records, handles


def print_model_summary(model: nn.Module):
    total, trainable, non_trainable = count_parameters(model)

    print("\nModel summary")
    print(f"Class:                 {model.__class__.__name__}")
    print(f"n_channels:            {model.n_channels}")
    print(f"n_outputs/classes:     {model.n_classes}")
    print(f"pp_as:                 {model.pp_as}")
    print(f"aggregate:             {model.aggregate}")
    print(f"norm:                  {model.norm}")
    print(f"meanmax_alpha:         {model.meanmax_alpha}")
    print(f"total_pool:            {model.total_pool}")
    print(f"parameters total:      {total:,}")
    print(f"parameters trainable:  {trainable:,}")
    print(f"parameters frozen:     {non_trainable:,}")
    print(f"model size:            {model_size_mb(model):.3f} MB")

    print("\nArchitecture")
    print(model)


def print_batch_summary(names, X, y, pp_as: str):
    print("\nBatch summary")
    print(f"B: {len(names)}")
    print(f"names: {names}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"y: {y.tolist()}")
    print("labels: 0=H/control, 1=MDD")

    print("\nInput X")

    if pp_as == "tensor":
        print("type: Tensor[B, C, T]")
        print(f"shape: {tuple(X.shape)}")
        print(f"B:     {X.shape[0]}")
        print(f"C:     {X.shape[1]}")
        print(f"T:     {X.shape[2]}")
        return

    lengths = [x.shape[-1] for x in X]
    channels = [x.shape[0] for x in X]

    print("type: list[Tensor[C, T_i]]")
    print(f"C unique: {sorted(set(channels))}")
    print(f"T min:   {min(lengths)}")
    print(f"T mean:  {sum(lengths) / len(lengths):.1f}")
    print(f"T max:   {max(lengths)}")
    print(f"shapes:  {[tuple(x.shape) for x in X]}")


def inspect_outputs(logits, logits_time):
    print("\nForward output")

    if isinstance(logits, torch.Tensor):
        print("logits type:  Tensor")
        print(f"logits shape: {tuple(logits.shape)}")
    else:
        print("logits type:  list")
        print(f"logits shapes: {[tuple(z.shape) for z in logits]}")

    if isinstance(logits_time, torch.Tensor):
        print("logits_time type:  Tensor")
        print(f"logits_time shape: {tuple(logits_time.shape)}")
    else:
        print("logits_time type:  list")
        print(f"logits_time shapes: {[tuple(z.shape) for z in logits_time]}")

    if isinstance(logits, torch.Tensor) and logits.ndim == 2:
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        print("\nSubject-level classification")
        print(f"probs shape: {tuple(probs.shape)}")
        print(f"pred shape:  {tuple(pred.shape)}")
        print(f"pred:        {pred.detach().cpu().tolist()}")
        print(f"probs:       {probs.detach().cpu().tolist()}")
        print("class order: [H/control, MDD]")

    elif isinstance(logits, torch.Tensor) and logits.ndim == 3:
        pred_time = logits.argmax(dim=-1)
        print("\nTemporal classification")
        print(f"pred_time shape: {tuple(pred_time.shape)}")
        print("aggregate=False returns temporal logits.")

    elif isinstance(logits, list):
        pred_time = [z.argmax(dim=-1) for z in logits]
        print("\nTemporal classification")
        print(f"pred_time shapes: {[tuple(z.shape) for z in pred_time]}")
        print("aggregate=False returns temporal logits as list.")


def append_final_output_records(records, logits, logits_time, aggregate: bool):
    if isinstance(logits_time, list):
        logits_time_shape = [tuple(z.shape) for z in logits_time]
    else:
        logits_time_shape = tuple(logits_time.shape)

    if isinstance(logits, torch.Tensor):
        logits_shape = tuple(logits.shape)
    else:
        logits_shape = [tuple(z.shape) for z in logits]

    records.append(
        {
            "name": "model.logits_time",
            "module": "Permute/Squeeze",
            "input_shape": "classifier output",
            "output_shape": logits_time_shape,
            "params": 0,
            "trainable_params": 0,
            "non_trainable_params": 0,
        }
    )

    records.append(
        {
            "name": "model.output",
            "module": "MeanMaxPool" if aggregate else "TemporalLogits",
            "input_shape": logits_time_shape,
            "output_shape": logits_shape,
            "params": 0,
            "trainable_params": 0,
            "non_trainable_params": 0,
        }
    )

    if isinstance(logits, torch.Tensor) and logits.ndim == 2:
        records.append(
            {
                "name": "model.prediction",
                "module": "ArgMax/Softmax",
                "input_shape": logits_shape,
                "output_shape": tuple(logits.argmax(dim=-1).shape),
                "params": 0,
                "trainable_params": 0,
                "non_trainable_params": 0,
            }
        )


def print_hook_records(records, max_rows: int = 120):
    print("\nLayer shape trace")
    print(
        f"{'#':>3s} | "
        f"{'name':35s} | "
        f"{'module':14s} | "
        f"{'input':30s} | "
        f"{'output':30s} | "
        f"{'params':>10s}"
    )
    print("-" * 145)

    for i, row in enumerate(records[:max_rows], start=1):
        print(
            f"{i:3d} | "
            f"{row['name'][:35]:35s} | "
            f"{row['module'][:14]:14s} | "
            f"{str(row['input_shape'])[:30]:30s} | "
            f"{str(row['output_shape'])[:30]:30s} | "
            f"{row['params']:10,d}"
        )

    if len(records) > max_rows:
        print(f"... truncated: {len(records) - max_rows} more rows")


def print_parameter_footer(model: nn.Module):
    total, trainable, non_trainable = count_parameters(model)

    print("\nFinal parameter count")
    print(f"total params:          {total:,}")
    print(f"total trainable:       {trainable:,}")
    print(f"total non-trainable:   {non_trainable:,}")


def parse_bool(value: str) -> bool:
    value = value.lower()

    if value in {"true", "1", "yes"}:
        return True

    if value in {"false", "0", "no"}:
        return False

    raise argparse.ArgumentTypeError("Expected true/false.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "data/raw/mdd_db"))
    parser.add_argument("--condition", type=str, choices=["EO", "EC"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=0)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)
    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])
    parser.add_argument(
        "--channel-strategy",
        type=str,
        default="common",
        choices=["common", "all"],
    )

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--temporal-kern", type=int, default=63)
    parser.add_argument("--separable-kern", type=int, default=15)
    parser.add_argument("--pool1", type=int, default=8)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--meanmax-alpha", type=float, default=0.0)
    parser.add_argument("--aggregate", type=parse_bool, default=True)
    parser.add_argument(
        "--norm",
        type=str,
        default="auto",
        choices=["auto", "batch", "group"],
    )

    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trace-layers", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")

    dataset = MDDDataset(
        root=args.root,
        condition=args.condition,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
        channel_strategy=args.channel_strategy,
    )

    print("\nDataset")
    print(f"condition:        {dataset.condition}")
    print("target/classes:   H/control=0 vs MDD=1")
    print(f"subjects/files:   {len(dataset)}")
    print(f"unique subjects:  {len(set(sample['subject'] for sample in dataset.samples))}")
    print(f"pp_as:            {dataset.pp_as}")
    print(f"channel_strategy: {dataset.channel_strategy}")
    print(f"n_channels:       {len(dataset.channel_names)}")
    print(f"channels:         {dataset.channel_names}")
    print(f"lowcut:           {dataset.lowcut}")
    print(f"highcut:          {dataset.highcut}")
    print(f"notch:            {dataset.notch}")
    print(f"target_fs:        {dataset.target_fs}")
    print(f"duration_sec:     {dataset.duration_sec}")

    name0, X0, y0 = dataset[args.sample_idx]
    n_channels = X0.shape[0]
    n_classes = 2

    print("\nSingle sample")
    print(f"idx:     {args.sample_idx}")
    print(f"name:    {name0}")
    print(f"X:       {tuple(X0.shape)}")
    print(f"sec@256: {X0.shape[-1] / 256.0:.2f}")
    print(f"y:       {int(y0.item())}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_list if args.pp_as == "list" else None,
    )

    names, X, y = next(iter(loader))

    print_batch_summary(names, X, y, pp_as=args.pp_as)

    if isinstance(X, torch.Tensor):
        X = X.to(device)
    else:
        X = [x.to(device) for x in X]

    y = y.to(device)

    if not args.no_standardize:
        X = standardize_eeg(X)

    model = EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        temporal_kern=args.temporal_kern,
        separable_kern=args.separable_kern,
        pool1=args.pool1,
        pool2=args.pool2,
        dropout=args.dropout,
        meanmax_alpha=args.meanmax_alpha,
        pp_as=args.pp_as,
        aggregate=args.aggregate,
        norm=args.norm,
    ).to(device)

    model.eval()
    print_model_summary(model)

    records = []
    handles = []

    if args.trace_layers:
        records, handles = register_shape_hooks(model)
        X_forward = X[:1] if isinstance(X, torch.Tensor) else [X[0]]
    else:
        X_forward = X

    with torch.no_grad():
        logits, logits_time = model(X_forward)

    for handle in handles:
        handle.remove()

    inspect_outputs(logits, logits_time)

    if args.trace_layers:
        append_final_output_records(
            records=records,
            logits=logits,
            logits_time=logits_time,
            aggregate=args.aggregate,
        )
        print_hook_records(records)

    print_parameter_footer(model)


if __name__ == "__main__":
    main()
