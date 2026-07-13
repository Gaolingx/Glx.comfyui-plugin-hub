"""Dataclasses shared by RF solver implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FlowERState:
    """History needed by the RF x0 LMS multistep corrector."""

    previous_denoised: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_lambda: Any | None = None


@dataclass
class FlowPC3State:
    """Accepted-state history for the RF x0 exponential PC3 solver."""

    previous_denoised: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_lambda: Any | None = None


@dataclass
class FlowPC3StepResult:
    """Accepted PC3 step plus diagnostics for logs and tests."""

    x: Any
    state: FlowPC3State
    x_predictor: Any
    x_corrected: Any
    gamma: Any
    error: Any
    predictor_order: int
    corrector_order: int


@dataclass
class FlowPC3PredictorResult:
    """PC3 predictor proposal and the internal order actually used."""

    x: Any
    order: int


@dataclass
class Flow3MStepResult:
    """Accepted one-eval 3M step plus diagnostics."""

    x: Any
    state: FlowERState
    x_2m: Any
    x_3m: Any
    gamma3: Any
    e32: Any
    order: int
    coeff_l1: float


@dataclass
class FlowUniPC2State:
    """History for the RF x0 UniPC predictor/corrector."""

    previous_denoised: Any | None = None
    previous_t: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_t: Any | None = None
    previous_previous_lambda: Any | None = None
    last_sample: Any | None = None
    lower_order_nums: int = 0
    this_order: int = 1
    model_outputs: tuple[Any | None, ...] | None = None
    sigma_history: tuple[Any | None, ...] | None = None
    lambda_history: tuple[Any | None, ...] | None = None


@dataclass
class FlowUniPC2StepResult:
    """Accepted UniPC step plus diagnostics for tests and logs."""

    x: Any
    state: FlowUniPC2State
    x_corrected: Any
    x_predictor: Any
    predictor_order: int
    corrector_order: int
