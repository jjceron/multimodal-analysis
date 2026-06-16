import sys, torch
sys.path.insert(0, '.')
from src.models.eegnet import EEGNet

model = EEGNet(n_channels=128, n_classes=2, F1=8, D=2, F2=16, aggregate=True, meanmax_alpha=0.5)
shapes = {}
hooks = []
def make_hook(path):
    def hook_fn(m, inp, out):
        in_shape = tuple(inp[0].shape) if inp and inp[0] is not None else ()
        out_shape = tuple(out.shape) if hasattr(out, 'shape') else ()
        p = sum(p.numel() for p in m.parameters())
        shapes[path] = {'in': in_shape, 'out': out_shape, 'params': p}
    return hook_fn
for name, m in model.named_modules():
    if name:
        hooks.append(m.register_forward_hook(make_hook(name)))
x = torch.randn(1, 128, 30000)
logits, logits_time = model(x)
for h in hooks:
    h.remove()
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_mods = dict(model.named_modules())

BLOCK_WIDTH = 25
IN_WIDTH = 32
OUT_WIDTH = 32
PARAM_WIDTH = 8
COL_GAP = 2
TOTAL_WIDTH = BLOCK_WIDTH + COL_GAP + IN_WIDTH + COL_GAP + OUT_WIDTH + COL_GAP + PARAM_WIDTH
LABEL_WIDTH = BLOCK_WIDTH + COL_GAP + IN_WIDTH + COL_GAP + OUT_WIDTH
sep = "=" * TOTAL_WIDTH

lines = [sep]
lines.append(f"{'Module':<{BLOCK_WIDTH}}{'':<{COL_GAP}}{'Input Shape':<{IN_WIDTH}}{'':<{COL_GAP}}{'Output Shape':<{OUT_WIDTH}}{'':<{COL_GAP}}{'Params':<{PARAM_WIDTH}}")
lines.append(sep)
lines.append(f"{'Input':<{BLOCK_WIDTH}}{'':<{COL_GAP}}{'(B, 128, 30000)':<{IN_WIDTH}}{'':<{COL_GAP}}{'-':<{OUT_WIDTH}}{'':<{COL_GAP}}{'0':<{PARAM_WIDTH}}")

skip_containers = {"temporal_block", "spatial_block", "separable_block", "eegnet"}
ordered = [p for p, _ in model.named_modules() if p and p not in skip_containers]
for path in ordered:
    s = shapes.get(path)
    if not s: continue
    m_type = type(model_mods.get(path, None)).__name__
    lines.append(f"{m_type:<{BLOCK_WIDTH}}{'':<{COL_GAP}}{str(s['in']):<{IN_WIDTH}}{'':<{COL_GAP}}{str(s['out']):<{OUT_WIDTH}}{'':<{COL_GAP}}{s['params']:<{PARAM_WIDTH}}")

s_class = shapes.get("classifier", {}) or shapes.get("classifier.0", {})
out_shape = s_class.get("out") if s_class else None
if s_class and out_shape:
    B, C, _, T_last = out_shape
    logits_per_time = (B, T_last, C)
    lines.append(f"{'MeanMax':<{BLOCK_WIDTH}}{'':<{COL_GAP}}{str(logits_per_time):<{IN_WIDTH}}{'':<{COL_GAP}}{str((B, C)):<{OUT_WIDTH}}{'':<{COL_GAP}}{'0':<{PARAM_WIDTH}}")
    lines.append(f"{'Output':<{BLOCK_WIDTH}}{'':<{COL_GAP}}{str((B, C)):<{IN_WIDTH}}{'':<{COL_GAP}}{'-':<{OUT_WIDTH}}{'':<{COL_GAP}}{'0':<{PARAM_WIDTH}}")

lines.append(sep)
lines.append(f"{'Total parameters':<{LABEL_WIDTH}}{'':<{COL_GAP}}{total_p:<{PARAM_WIDTH}}")
lines.append(f"{'Trainable parameters':<{LABEL_WIDTH}}{'':<{COL_GAP}}{train_p:<{PARAM_WIDTH}}")
lines.append(f"{'Non-trainable parameters':<{LABEL_WIDTH}}{'':<{COL_GAP}}{total_p - train_p:<{PARAM_WIDTH}}")
lines.append(sep)

print('\n'.join(lines))
print(f"\nTOTAL_WIDTH={TOTAL_WIDTH}, sep={len(sep)}, data_width=103")
