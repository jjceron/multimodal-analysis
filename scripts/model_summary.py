"""
Print model layer summary table: Layer | Class | Input Shape | Output Shape | Params
Usage:  python scripts/model_summary.py --model window_regressor --channels 90 --samples 500
        python scripts/model_summary.py --model eegnet --channels 22 --samples 256 --classes 2
"""
import argparse, sys, warnings
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

MODELS = {}


def register(name):
    def decorator(fn):
        MODELS[name] = fn
        return fn
    return decorator


@register('eegnet')
def build_eegnet(args):
    from src.models.eegnet import EEGNet
    return EEGNet(n_channels=args.channels, n_classes=args.classes,
                  F1=getattr(args,'F1',8), D=getattr(args,'D',2),
                  F2=getattr(args,'F2',16),
                  temporal_kern=getattr(args,'temporal_kern',63),
                  separable_kern=getattr(args,'separable_kern',15),
                  pool1=getattr(args,'pool1',8), pool2=getattr(args,'pool2',8),
                  dropout=getattr(args,'dropout',0.5), meanmax_alpha=0.0)


@register('window_regressor')
def build_window_regressor(args):
    from src.models.eegnet import EEGNet

    class WindowRegressor(nn.Module):
        def __init__(self, n_channels=90, dropout=0.5):
            super().__init__()
            self.eegnet = EEGNet(n_channels=n_channels, n_classes=1, F1=8, D=2, F2=16,
                                 temporal_kern=63, separable_kern=15,
                                 pool1=8, pool2=8, dropout=dropout, meanmax_alpha=0.0)
        def forward(self, x):
            logits, _ = self.eegnet(x)
            return logits.squeeze(-1)

    return WindowRegressor(n_channels=args.channels, dropout=getattr(args,'dropout',0.5))


@register('shallowconvnet')
def build_shallowconvnet(args):
    from src.models.shallowconvnet import ShallowConvNet
    return ShallowConvNet(n_channels=args.channels, n_classes=args.classes,
                          n_samples=args.samples, dropout=getattr(args,'dropout',0.5))


@register('cnn_lstm')
def build_cnnlstm(args):
    from src.models.cnn_lstm import CNNLSTM
    return CNNLSTM(n_channels=args.channels, n_classes=args.classes,
                   n_samples=args.samples, dropout=getattr(args,'dropout',0.5))


def print_summary(model, input_shape, model_name='model'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    shapes_data = {}
    def make_hook(layer_name):
        def hook(m, inp, out):
            # Handle tuple outputs (e.g. EEGNet returns (logits, logits_time))
            if isinstance(out, tuple):
                out_t = out[0]
            else:
                out_t = out
            in_t = inp[0] if isinstance(inp, (tuple, list)) else inp
            in_shape = tuple(in_t.shape) if isinstance(in_t, torch.Tensor) else '–'
            out_shape = tuple(out_t.shape) if isinstance(out_t, torch.Tensor) else '–'
            params = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            shapes_data[layer_name] = (m.__class__.__name__, in_shape, out_shape, params, trainable)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    dummy = torch.randn(*input_shape).to(device)
    try:
        output = model(dummy)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        output = None

    for h in hooks:
        h.remove()

    # ── Calculate output shape ────────────────────────────────────────
    if output is not None and isinstance(output, torch.Tensor):
        output_shape = tuple(output.shape)
    elif output is not None and isinstance(output, tuple):
        output_shape = tuple(output[0].shape) if isinstance(output[0], torch.Tensor) else '?'
    else:
        output_shape = '?'

    # ── Print table ───────────────────────────────────────────────────
    w_name, w_cls, w_in, w_out, w_par = 48, 20, 24, 24, 8
    divider = f"  {'-'*(w_name + w_cls + w_in + w_out + w_par + 8)}"

    print(f"\n{divider}")
    print(f"  {'Layer':<{w_name}} {'Class':<{w_cls}} {'Input Shape':<{w_in}} {'Output Shape':<{w_out}} {'Params':>{w_par}}")
    print(f"{divider}")

    # Input row
    print(f"  {'input':<{w_name}} {'':<{w_cls}} {str(tuple(input_shape)):<{w_in}} {str(tuple(input_shape)):<{w_out}} {0:>{w_par},}")

    total_params, total_trainable = 0, 0
    for name, _ in model.named_modules():
        if not name or name not in shapes_data:
            continue
        # Skip parent modules (those with children)
        if any(k.startswith(name + '.') for k in shapes_data):
            continue
        cls, ins, outs, par, tr_par = shapes_data[name]
        display = name if len(name) <= w_name else '...' + name[-(w_name-3):]
        print(f"  {display:<{w_name}} {cls:<{w_cls}} {str(ins):<{w_in}} {str(outs):<{w_out}} {par:>{w_par},}")
        total_params += par
        total_trainable += tr_par

    # Output row → last leaf module's output (skip parents)
    last_out_shape = tuple(input_shape)
    for name in reversed(list(shapes_data.keys())):
        if any(k.startswith(name + '.') for k in shapes_data):
            continue
        _, _, out_s, par, _ = shapes_data[name]
        if par > 0:
            last_out_shape = out_s
            break
    # Determine what transforms classifier output → final shape
    # EEGNet internally: squeeze(2) + permute + agg_meanmax → [B, n_classes]
    # WindowRegressor: squeeze(-1) → [B]
    out_ops = []
    if output is not None and isinstance(output, torch.Tensor) and last_out_shape != output_shape:
        out_ops.append('squeeze')
    if not out_ops:
        out_ops.append('output')
    out_op_cls = '/'.join(out_ops)

    print(f"  {'output':<{w_name}} {out_op_cls:<{w_cls}} {str(last_out_shape):<{w_in}} {str(output_shape):<{w_out}} {0:>{w_par},}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"{divider}")
    print(f"  Total params:      {total:>12,}")
    print(f"  Trainable:         {trainable:>12,}")
    print(f"  Non-trainable:     {non_trainable:>12,}")
    print(f"{divider}")
    print(f"  Input shape:  {tuple(input_shape)}")
    print(f"  Output shape: {output_shape}")
    print(f"  Model name:   {model_name}")
    print(f"{divider}\n")


def parse_args():
    p = argparse.ArgumentParser(description='Print model layer summary')
    p.add_argument('--model', choices=sorted(MODELS), required=True,
                   help='Model to summarize')
    p.add_argument('--channels', type=int, default=90,
                   help='Number of input channels (default: 90)')
    p.add_argument('--samples', type=int, default=500,
                   help='Number of time samples (default: 500)')
    p.add_argument('--classes', type=int, default=1,
                   help='Number of output classes (default: 1)')
    p.add_argument('--F1', type=int, default=8)
    p.add_argument('--D', type=int, default=2)
    p.add_argument('--F2', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--batch', type=int, default=1,
                   help='Batch size for shape display (default: 1)')
    return p.parse_args()


def main():
    args = parse_args()
    input_shape = (args.batch, args.channels, args.samples)
    model = MODELS[args.model](args)
    print_summary(model, input_shape, model_name=args.model)


if __name__ == '__main__':
    main()
