"""Small RF math helpers shared by schedulers and solver code."""


def _as_tensor_like(torch, value, like):
    if hasattr(value, "to"):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


def _broadcast_time(torch, value, like):
    if hasattr(value, "to"):
        out = value.to(device=like.device, dtype=like.dtype)
    else:
        out = torch.tensor(value, device=like.device, dtype=like.dtype)

    while out.ndim < like.ndim:
        out = out[..., None]
    return out


def _clamp01(value: float) -> float:
    if value != value:
        raise ValueError("progress value must not be NaN")
    return min(1.0, max(0.0, float(value)))


def _finite_schedule_terminal(torch, sigmas):
    if int(sigmas.shape[0]) >= 2:
        return sigmas[-2]
    return sigmas[-1]


def _rf_external_sigma(torch, t, *, like, eps: float = 1e-6):
    value = _as_tensor_like(torch, t, like)
    value = torch.clamp(value, min=eps, max=1.0 - eps)
    return value / (1.0 - value)


def _rf_lambda(torch, t, *, eps: float = 1e-6):
    if hasattr(t, "to"):
        value = t
    else:
        value = torch.as_tensor(t)
    value = torch.clamp(value, min=eps, max=1.0 - eps)
    return torch.log1p(-value) - torch.log(value)


def _rms(torch, value):
    return torch.sqrt(torch.mean(value.float() * value.float()))


def _scalar_float(torch, value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(torch.as_tensor(value).detach().cpu())
