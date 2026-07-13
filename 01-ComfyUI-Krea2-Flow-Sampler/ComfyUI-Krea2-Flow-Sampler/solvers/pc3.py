"""RF x0 exponential PC3 predictor/corrector solver."""

from __future__ import annotations

import importlib

from ..flow_math import _as_tensor_like, _rf_lambda, _rms, _scalar_float
from ..solver_types import FlowPC3PredictorResult, FlowPC3State, FlowPC3StepResult


def flow_pc3_predictor_step_result(
    x,
    denoised,
    t,
    t_next,
    *,
    state: FlowPC3State | None = None,
    max_order: int = 3,
    eps: float = 1e-6,
) -> FlowPC3PredictorResult:
    """Predict an RF PC3 endpoint with internal P1/P2/P3 warmup."""

    if float(t_next) <= 0.0 or float(t) <= 0.0:
        return FlowPC3PredictorResult(denoised, 1)

    torch = importlib.import_module("torch")
    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    current_lambda = _rf_lambda(torch, t, eps=eps)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    h_f = _scalar_float(torch, h)
    ratio = t_next_tensor / t_current

    if h_f <= eps:
        return FlowPC3PredictorResult(ratio * x + (1.0 - ratio) * denoised, 1)

    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    x_p1 = ratio * x + c * k0 * denoised

    max_order = max(1, min(3, int(max_order)))
    if max_order <= 1 or state is None or state.previous_denoised is None or state.previous_lambda is None:
        return FlowPC3PredictorResult(x_p1, 1)

    h_previous = current_lambda - state.previous_lambda
    h_previous_f = _scalar_float(torch, h_previous)
    if h_previous_f <= eps:
        return FlowPC3PredictorResult(x_p1, 1)

    x_p2 = ratio * x + c * (
        (k0 + k1 / h_previous) * denoised
        - (k1 / h_previous) * state.previous_denoised
    )
    if (
        max_order <= 2
        or state.previous_previous_denoised is None
        or state.previous_previous_lambda is None
    ):
        return FlowPC3PredictorResult(x_p2, 2)

    h_previous_previous = state.previous_lambda - state.previous_previous_lambda
    h_previous_previous_f = _scalar_float(torch, h_previous_previous)
    if h_previous_previous_f <= eps:
        return FlowPC3PredictorResult(x_p2, 2)

    r1 = h_f / h_previous_f
    r2 = h_previous_f / h_previous_previous_f
    k2 = torch.exp(h) * h * h - 2.0 * k1
    a = h_previous
    b = h_previous + h_previous_previous
    weight_current = k0 + ((a + b) / (a * b)) * k1 + k2 / (a * b)
    weight_previous = -(k2 + b * k1) / (a * (b - a))
    weight_previous_previous = (k2 + a * k1) / (b * (b - a))
    coeff_l1 = _scalar_float(
        torch,
        (
            torch.abs(weight_current)
            + torch.abs(weight_previous)
            + torch.abs(weight_previous_previous)
        )
        / (torch.abs(k0) + eps),
    )
    if not (0.50 <= r1 <= 2.00 and 0.50 <= r2 <= 2.00) or h_f > 0.55 or coeff_l1 > 5.0:
        return FlowPC3PredictorResult(x_p2, 2)

    return FlowPC3PredictorResult(
        ratio * x
        + c
        * (
            weight_current * denoised
            + weight_previous * state.previous_denoised
            + weight_previous_previous * state.previous_previous_denoised
        ),
        3,
    )


def flow_pc3_next_state(torch, state: FlowPC3State | None, denoised, t, *, eps: float = 1e-6):
    """Accept the current x0 prediction into PC3 history."""

    current_lambda = _rf_lambda(torch, t, eps=eps)
    if state is None:
        state = FlowPC3State()
    return FlowPC3State(
        previous_denoised=denoised,
        previous_lambda=current_lambda,
        previous_previous_denoised=state.previous_denoised,
        previous_previous_lambda=state.previous_lambda,
    )


def flow_pc3_predictor_max_order(step_index: int, total_steps: int) -> int:
    """Lower PC3 to second order for the final two integration steps."""

    return 2 if int(total_steps) - int(step_index) <= 2 else 3


def flow_pc3_should_endpoint_correct(
    torch,
    state: FlowPC3State | None,
    predictor_order: int,
    step_index: int,
    total_steps: int,
    t_next,
) -> bool:
    """Only spend the second model call on a fully warmed-up third-order step."""

    if _scalar_float(torch, t_next) <= 0.0:
        return False
    if int(total_steps) - int(step_index) <= 2:
        return False
    if (
        state is None
        or state.previous_denoised is None
        or state.previous_lambda is None
        or state.previous_previous_denoised is None
        or state.previous_previous_lambda is None
    ):
        return False
    return int(predictor_order) >= 3


def _flow_pc3_clamped_gamma(
    torch,
    x,
    x_pred,
    x_corrected,
    gamma,
    lambda_next,
    *,
    predictor_order: int,
    eps: float = 1e-6,
):
    if int(predictor_order) < 3:
        return gamma

    correction_rms = _rms(torch, x_corrected - x_pred)
    if _scalar_float(torch, correction_rms) <= eps:
        return gamma

    predictor_step_rms = _rms(torch, x_pred - x)
    anchor_rms = torch.maximum(_rms(torch, x_pred), _rms(torch, x))
    cap_ratio = 0.65 if _scalar_float(torch, lambda_next) >= 3.5 else 0.90
    correction_cap = torch.maximum(predictor_step_rms * cap_ratio, anchor_rms * 0.015)
    applied_rms = torch.abs(gamma) * correction_rms
    scale = torch.clamp(correction_cap / (applied_rms + eps), min=0.0, max=1.0)
    return gamma * scale


def flow_pc3_damped_step_result(
    x,
    denoised,
    denoised_pred,
    t,
    t_next,
    *,
    state: FlowPC3State | None = None,
    max_gamma: float = 1.0,
    tolerance: float = 0.005,
    x_pred=None,
    predictor_order: int = 1,
    eps: float = 1e-6,
) -> FlowPC3StepResult:
    """Correct a PC3 endpoint prediction with adaptive damping."""

    if state is None:
        state = FlowPC3State()
    torch = importlib.import_module("torch")
    current_lambda = _rf_lambda(torch, t, eps=eps)
    next_state = flow_pc3_next_state(torch, state, denoised, t, eps=eps)
    predictor_order = max(1, int(predictor_order))
    if x_pred is None:
        predictor = flow_pc3_predictor_step_result(
            x, denoised, t, t_next, state=state, eps=eps
        )
        x_pred = predictor.x
        predictor_order = predictor.order

    zero = x.new_tensor(0.0)
    if float(t_next) <= 0.0 or float(t) <= 0.0:
        return FlowPC3StepResult(x_pred, next_state, x_pred, x_pred, zero, zero, predictor_order, 0)
    if state.previous_denoised is None or state.previous_lambda is None:
        return FlowPC3StepResult(x_pred, next_state, x_pred, x_pred, zero, zero, predictor_order, 0)

    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    h_previous = current_lambda - state.previous_lambda
    if _scalar_float(torch, h) <= eps or _scalar_float(torch, h_previous) <= eps:
        return FlowPC3StepResult(x_pred, next_state, x_pred, x_pred, zero, zero, predictor_order, 0)

    gamma_max = max(0.0, min(1.0, float(max_gamma)))
    tol = max(0.0, float(tolerance))
    if gamma_max <= 0.0 or tol <= 0.0:
        return FlowPC3StepResult(x_pred, next_state, x_pred, x_pred, zero, zero, predictor_order, 0)

    ratio = t_next_tensor / t_current
    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    k2 = torch.exp(h) * h * h - 2.0 * k1
    previous_node = state.previous_lambda - current_lambda
    previous_weight = (k2 - h * k1) / (previous_node * (previous_node - h))
    current_weight = (k2 - (previous_node + h) * k1 + previous_node * h * k0) / (previous_node * h)
    endpoint_weight = (k2 - previous_node * k1) / ((h - previous_node) * h)
    x_corrected = ratio * x + c * (
        previous_weight * state.previous_denoised
        + current_weight * denoised
        + endpoint_weight * denoised_pred
    )

    error = _rms(torch, x_corrected - x_pred) / (_rms(torch, x_pred) + eps)
    gamma_error = torch.sqrt(torch.clamp(x.new_tensor(tol) / (error + eps), min=0.0))
    gamma_error = torch.clamp(gamma_error, min=0.0, max=1.0)
    gamma = gamma_max * _flow_pc3_lambda_gate(torch, current_lambda, lambda_next) * gamma_error
    gamma = _flow_pc3_clamped_gamma(
        torch,
        x,
        x_pred,
        x_corrected,
        gamma,
        lambda_next,
        predictor_order=predictor_order,
        eps=eps,
    )
    return FlowPC3StepResult(
        x_pred + gamma * (x_corrected - x_pred),
        next_state,
        x_pred,
        x_corrected,
        gamma,
        error,
        predictor_order,
        3,
    )


def _flow_pc3_lambda_gate(torch, lambda_current, lambda_next):
    high_noise_gate = torch.sigmoid((lambda_current + 2.5) / 0.5)
    low_noise_gate = torch.sigmoid((4.5 - lambda_next) / 0.8)
    return high_noise_gate * low_noise_gate
