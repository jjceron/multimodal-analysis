# scripts/inspect_eegnet_hbn.py

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.hbn_db import HBNRestingStateDataset
from src.models.eegnet import EEGNet


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


def print_batch_summary(names, X, y):
    print("\nBatch summary")
    print(f"B: {len(names)}")
    print(f"names: {names}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"y: {y.tolist()}")

    lengths = [x.shape[-1] for x in X]
    channels = [x.shape[0] for x in X]

    print("\nInput X")
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

    if isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[-1] == 1:
        pred = logits.squeeze(-1)
        print(f"prediction shape after squeeze: {tuple(pred.shape)}")
        print(f"prediction values: {pred.detach().cpu().tolist()}")


def append_final_output_records(records, logits, logits_time):
    """
    Add non-module rows so the trace does not stop at classifier output
    (B, 1, 1, T'). These rows show the model-level transformation to
    logits_time and the final subject-level prediction.
    """
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
            "name": "model.agg_meanmax",
            "module": "MeanMaxPool",
            "input_shape": logits_time_shape,
            "output_shape": logits_shape,
            "params": 0,
            "trainable_params": 0,
            "non_trainable_params": 0,
        }
    )

    if isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[-1] == 1:
        records.append(
            {
                "name": "model.prediction",
                "module": "Squeeze",
                "input_shape": logits_shape,
                "output_shape": tuple(logits.squeeze(-1).shape),
                "params": 0,
                "trainable_params": 0,
                "non_trainable_params": 0,
            }
        )


def print_hook_records(records, max_rows: int = 120):
    print("\nLayer shape trace")
    print(
        f"{'#':>3s} | "
        f"{'name':30s} | "
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/hbn_db/R1_L100_bdf",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["EO", "EC"],
        required=True,
    )
    parser.add_argument("--target", type=str, default="externalizing")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=0)

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--temporal-kern", type=int, default=63)
    parser.add_argument("--separable-kern", type=int, default=15)
    parser.add_argument("--pool1", type=int, default=8)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--meanmax-alpha", type=float, default=0.0)
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

    dataset = HBNRestingStateDataset(
        root=args.root,
        condition=args.condition,
        target=args.target,
        preload=False,
    )

    print("\nDataset")
    print(f"condition: {dataset.condition}")
    print(f"target:    {dataset.target}")
    print(f"subjects:  {len(dataset)}")
    print(f"pp_as:     {dataset.pp_as}")

    name0, X0, y0 = dataset[args.sample_idx]
    n_channels = X0.shape[0]

    print("\nSingle sample")
    print(f"idx:  {args.sample_idx}")
    print(f"name: {name0}")
    print(f"X:    {tuple(X0.shape)}")
    print(f"sec:  {X0.shape[-1] / 100.0:.2f}")
    print(f"y:    {float(y0):.4f}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_list,
    )

    names, X, y = next(iter(loader))

    print_batch_summary(names, X, y)

    X = [x.to(device) for x in X]
    y = y.to(device)

    if not args.no_standardize:
        X = standardize_eeg(X)

    model = EEGNet(
        n_channels=n_channels,
        n_classes=1,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        temporal_kern=args.temporal_kern,
        separable_kern=args.separable_kern,
        pool1=args.pool1,
        pool2=args.pool2,
        dropout=args.dropout,
        meanmax_alpha=args.meanmax_alpha,
        pp_as="list",
        aggregate=True,
        norm=args.norm,
    ).to(device)

    model.eval()

    print_model_summary(model)

    records = []
    handles = []

    if args.trace_layers:
        # Trace one subject only, so the table is readable.
        # In normal forward without tracing, the full batch is used.
        records, handles = register_shape_hooks(model)
        X_forward = [X[0]]
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
        )
        print_hook_records(records)

    print_parameter_footer(model)


if __name__ == "__main__":
    main()