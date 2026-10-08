"""Adversarial patch attacks on [0, 1] images: universal patch (Brown et al. 2017) and per-image LaVAN
(Karmon et al. 2018).

A patch is a square of side `size` pasted over the image (it replaces pixels, no blending). On 32x32
CIFAR-10, side 5 covers 2.44% of the image and side 8 covers 6.25% (plan, Part 3).

Placement is random. Training a universal patch draws fresh locations every step (EOT over location).
Evaluation uses fixed locations from a seeded generator, so every model sees the same placements.

"physical" EOT (ablation) also draws a random rotation and brightness / contrast change of the patch
each time it is pasted. It makes the patch robust to those changes at the cost of digital strength,
so location-only EOT is the main (stronger digital) attack.

Gradient-free models: every attack takes a `grad_model` (where gradients come from) and an
`eval_model` (whose predictions count). They differ for quantized targets attacked through their
FP32 surrogate (BPDA): forward pass on the quantized model, backward pass on the float one.
"""
import math

import torch
import torch.nn.functional as F

IMG = 32
PHYSICAL = {"max_rot_deg": 20.0, "brightness": (0.8, 1.2), "contrast": (0.8, 1.2)}


def random_locations(n, size, gen, device=None):
    """Top-left (row, col) of n patches, uniform over all positions that keep the patch inside the image."""
    hi = IMG - size + 1
    rows = torch.randint(0, hi, (n,), generator=gen)
    cols = torch.randint(0, hi, (n,), generator=gen)
    return rows.to(device), cols.to(device)


def _uniform(n, lo, hi, gen):
    return lo + (hi - lo) * torch.rand(n, generator=gen)


def physical_transform(patch, n, gen):
    """n randomly rotated / brightness / contrast-changed copies of patch (3, s, s).

    Returns (patches (n, 3, s, s), masks (n, 1, s, s)); the mask is where the rotated patch lands,
    the rest of the s x s square shows the image underneath.
    """
    dev, s = patch.device, patch.size(-1)
    ang = _uniform(n, -1.0, 1.0, gen).to(dev) * math.radians(PHYSICAL["max_rot_deg"])
    cos, sin, zero = ang.cos(), ang.sin(), torch.zeros_like(ang)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid = F.affine_grid(theta, (n, 4, s, s), align_corners=False)
    both = torch.cat([patch, torch.ones_like(patch[:1])]).expand(n, 4, s, s)
    out = F.grid_sample(both, grid, align_corners=False)
    p, mask = out[:, :3], out[:, 3:]
    b = _uniform(n, *PHYSICAL["brightness"], gen).to(dev).view(-1, 1, 1, 1)
    c = _uniform(n, *PHYSICAL["contrast"], gen).to(dev).view(-1, 1, 1, 1)
    mean = p.mean((1, 2, 3), keepdim=True)
    return ((p - mean) * c + mean * b).clamp(0, 1), mask


def apply_patch(x, patch, rows, cols, mask=None):
    """Paste patch (3, s, s) or (B, 3, s, s) into x (B, 3, H, W) at per-image (rows, cols).

    With mask (B, 1, s, s) the square shows patch * mask + image * (1 - mask). Differentiable in patch.
    """
    b, s = x.size(0), patch.size(-1)
    ar = torch.arange(s, device=x.device)
    idx = (torch.arange(b, device=x.device)[:, None, None], slice(None),
           (rows[:, None] + ar)[:, :, None], (cols[:, None] + ar)[:, None, :])
    val = patch.movedim(-3, -1)
    if mask is not None:
        m = mask.movedim(-3, -1)
        val = val * m + x[idx] * (1 - m)
    out = x.clone()
    out[idx] = val
    return out


def paste(x, patch, gen, physical=False, locations=None):
    """Patch at random (or given) locations, with physical transforms if asked."""
    rows, cols = locations or random_locations(x.size(0), patch.size(-1), gen, x.device)
    if physical:
        p, mask = physical_transform(patch, x.size(0), gen)
        return apply_patch(x, p, rows, cols, mask)
    return apply_patch(x, patch, rows, cols)


def _batches(loader):
    while True:
        yield from loader


def train_universal(model, loader, size, target, steps, lr, device, gen, physical=False,
                    monitor=None, monitor_every=250):
    """Targeted universal patch: minimize CE(model(x with patch), target) with Adam, EOT over placement.

    monitor(patch) -> dict is called every monitor_every steps (convergence curve).
    """
    patch = torch.rand(3, size, size, generator=gen).to(device).requires_grad_()
    opt = torch.optim.Adam([patch], lr=lr)
    curve = []
    for step, (x, _) in enumerate(_batches(loader), start=1):
        x = x.to(device, non_blocking=True)
        t = torch.full((x.size(0),), target, device=device)
        loss = F.cross_entropy(model(paste(x, patch, gen, physical)), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
        if monitor and (step % monitor_every == 0 or step == steps):
            curve.append({"step": step, **monitor(patch.detach())})
        if step == steps:
            break
    return patch.detach(), curve


@torch.no_grad()
def eval_universal(model, x_all, y_all, patch, target, seed, physical=False, bs=1000):
    """Patch at fixed seeded placements (one per image, in order). Accuracy and target hit rate."""
    gen = torch.Generator().manual_seed(seed)
    rows, cols = random_locations(len(x_all), patch.size(-1), gen, x_all.device)
    correct = hits = 0
    for i in range(0, len(x_all), bs):
        x, y, j = x_all[i:i + bs], y_all[i:i + bs], slice(i, i + bs)
        pred = model(paste(x, patch, gen, physical, (rows[j], cols[j]))).argmax(1)
        correct += (pred == y).sum().item()
        hits += ((pred == target) & (y != target)).sum().item()
    return {"patched_acc": correct / len(x_all), "target_success": hits / (y_all != target).sum().item()}


def lavan(grad_model, eval_model, x, y, size, steps, step_size, restarts, gen, target=None):
    """Per-image patch (LaVAN), sign-gradient steps on the patch pixels, random location per restart.

    targeted (target is a tensor): drive eval_model's prediction to target; untargeted: away from y.
    An image stops updating once it succeeds on eval_model, so its patch is the first successful one;
    images that fail get a new location and a fresh random patch on the next restart.
    Returns patches (B, 3, s, s), rows, cols, success (B,) bool.
    """
    b, dev = x.size(0), x.device
    patch = torch.zeros(b, 3, size, size, device=dev)
    rows = torch.zeros(b, dtype=torch.long, device=dev)
    cols = torch.zeros_like(rows)
    done = torch.zeros(b, dtype=torch.bool, device=dev)
    for _ in range(restarts):
        idx = (~done).nonzero().squeeze(1)  # each restart only attacks the images not yet fooled
        if len(idx) == 0:
            break
        xs, ys, ts = x[idx], y[idx], None if target is None else target[idx]
        r, c = random_locations(len(idx), size, gen, dev)
        p = torch.rand(len(idx), 3, size, size, generator=gen).to(dev)
        d = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        for i in range(steps + 1):
            p.requires_grad_()
            xp = apply_patch(xs, p, r, c)
            logits = grad_model(xp)
            pred = (logits if eval_model is grad_model else eval_model(xp.detach())).argmax(1)
            d |= (pred == ts) if ts is not None else (pred != ys)
            if d.all() or i == steps:
                break
            # reduction="sum" keeps per-image gradients independent of batch size (BN is in eval mode)
            loss = F.cross_entropy(logits, ts, reduction="sum") if ts is not None \
                else -F.cross_entropy(logits, ys, reduction="sum")
            grad, = torch.autograd.grad(loss, p)
            with torch.no_grad():
                p = (p - step_size * grad.sign() * (~d).view(-1, 1, 1, 1)).clamp(0, 1)
        patch[idx], rows[idx], cols[idx], done[idx] = p.detach(), r, c, d
    return patch, rows, cols, done
