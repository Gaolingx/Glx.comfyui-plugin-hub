"""Krea2-native Flow-Matching Euler sampler.

This deliberately does not reuse Anima schedules.  Krea2 is registered in
ComfyUI as a FLUX-family model, so its own ModelSamplingFlux sigma mapping is
used to construct the trajectory.
"""

from __future__ import annotations

import math
import torch

from .flow_math import _rf_lambda, _scalar_float
from .solver_types import FlowPC3State, FlowUniPC2State
from .solvers.pc3 import (
    flow_pc3_damped_step_result,
    flow_pc3_next_state,
    flow_pc3_predictor_max_order,
    flow_pc3_predictor_step_result,
    flow_pc3_should_endpoint_correct,
)
from .solvers.unipc import flow_unipc2_x0_step


def _flux_sigma(shift: float, t: torch.Tensor) -> torch.Tensor:
    """The same shifted-flow mapping used by ComfyUI ModelSamplingFlux."""
    e = math.exp(float(shift))
    return e / (e + (1.0 / t - 1.0))


def _krea_sigmas(model, steps: int, denoise: float, shift_override: float) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0.0 < denoise <= 1.0:
        raise ValueError("denoise must be in (0, 1]")

    sampling = model.get_model_object("model_sampling")
    import comfy.model_sampling
    if not isinstance(sampling, comfy.model_sampling.ModelSamplingFlux):
        raise ValueError(
            "Krea2 Flow Euler requires a Krea2/FLUX model "
            f"(got {sampling.__class__.__name__})."
        )

    # Use the model's declared shift unless the user explicitly requests one.
    shift = float(getattr(sampling, "shift", 1.15)) if shift_override <= 0 else float(shift_override)
    total = max(steps, int(math.ceil(steps / denoise)))
    # Never evaluate t=0 in the shift formula; append sigma=0 afterwards.
    t = torch.linspace(1.0, 1.0 / total, total, dtype=torch.float32)
    sigmas = _flux_sigma(shift, t)
    sigmas = torch.cat((sigmas, torch.zeros(1, dtype=sigmas.dtype)))
    return sigmas[-(steps + 1):]


def _pc3_diffusers_tail(t, t_next) -> bool:
    """Identify unstable terminal intervals on a shifted-linear RF grid."""
    t_value = _scalar_float(torch, t)
    t_next_value = _scalar_float(torch, t_next)
    if t_next_value <= 0.0 or t_next_value <= 0.10:
        return True
    lambda_gap = _scalar_float(torch, _rf_lambda(torch, t_next) - _rf_lambda(torch, t))
    return t_value <= 0.25 and lambda_gap >= 0.65


def sample_krea2_flow(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    flow_solver="euler",
    pc3_gamma=1.0,
    pc3_tolerance=0.005,
    **_,
):
    """Stable x0/flow solvers on Krea2's native FLUX sigma trajectory."""
    extra_args = extra_args or {}
    previous_d = None
    previous_step = None
    unipc_state = None
    pc3_state = FlowPC3State()
    total_steps = len(sigmas) - 1
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)
        if callback is not None:
            callback({"i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised, "x": x})

        if flow_solver == "pc3_diffusers_damped":
            # The zero endpoint is already the current model's x0 prediction.
            if float(sigma_next) <= 0.0 or float(sigma) <= 0.0:
                x = denoised
                continue

            tail_interval = _pc3_diffusers_tail(sigma, sigma_next)
            max_order = 1 if tail_interval else flow_pc3_predictor_max_order(i, total_steps)
            predictor = flow_pc3_predictor_step_result(
                x,
                denoised,
                sigma,
                sigma_next,
                state=pc3_state,
                max_order=max_order,
            )
            should_correct = not tail_interval and flow_pc3_should_endpoint_correct(
                torch,
                pc3_state,
                predictor.order,
                i,
                total_steps,
                sigma_next,
            )
            if not should_correct:
                x = predictor.x
                pc3_state = flow_pc3_next_state(torch, pc3_state, denoised, sigma)
                continue

            denoised_pred = model(
                predictor.x,
                sigma_next * predictor.x.new_ones([predictor.x.shape[0]]),
                **extra_args,
            )
            result = flow_pc3_damped_step_result(
                x,
                denoised,
                denoised_pred,
                sigma,
                sigma_next,
                state=pc3_state,
                max_gamma=float(pc3_gamma),
                tolerance=float(pc3_tolerance),
                x_pred=predictor.x,
                predictor_order=predictor.order,
            )
            pc3_state = result.state
            x = result.x
            continue

        if flow_solver == "unipc2_diffusers_x0":
            # UniPC2 x0 with the Diffusers-style BH2 update and lower-order
            # final steps.  Unlike Anima, this stays on Krea's native grid.
            result = flow_unipc2_x0_step(
                x,
                denoised,
                sigma,
                sigma_next,
                state=unipc_state,
                step_index=i,
                total_steps=len(sigmas) - 1,
                solver_order=2,
                solver_type="bh2",
                lower_order_final=True,
            )
            unipc_state = result.state
            x = result.x
            continue

        # The final zero-sigma point is the model's x0 prediction.
        if float(sigma_next) <= 0.0 or float(sigma) <= 0.0:
            x = denoised
            continue

        # d = (x - x0) / sigma is ComfyUI's standard ODE derivative.
        step = sigma_next - sigma
        d = (x - denoised) / sigma
        if flow_solver == "heun":
            predicted = x + d * step
            denoised_next = model(
                predicted,
                sigma_next * predicted.new_ones([predicted.shape[0]]),
                **extra_args,
            )
            d_next = (predicted - denoised_next) / sigma_next
            x = x + (d + d_next) * (step * 0.5)
        elif flow_solver == "ab2" and previous_d is not None and previous_step is not None and float(previous_step) != 0.0:
            # Variable-step Adams-Bashforth 2. The native shifted sigma grid
            # is not uniform, so the usual fixed-step AB2 coefficients are
            # intentionally not used.
            ratio = step / previous_step
            x = x + step * ((1.0 + ratio * 0.5) * d - ratio * 0.5 * previous_d)
        else:
            x = x + d * step

        previous_d = d
        previous_step = step
    return x


class Krea2FlowEulerSampler:
    """Use only with Krea2 models; compatible with Qwen VAE latent IO."""

    CATEGORY = "sampling/krea2"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "add_noise": (["enable", "disable"],),
                "flow_solver": (["euler", "heun", "ab2", "unipc2_diffusers_x0", "pc3_diffusers_damped"], {"default": "euler"}),
                "shift_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pc3_gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "pc3_tolerance": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
            }
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        denoise,
        add_noise,
        flow_solver,
        shift_override,
        pc3_gamma,
        pc3_tolerance,
    ):
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import latent_preview

        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        expected_channels = model.get_model_object("latent_format").latent_channels
        if samples.shape[1] != expected_channels:
            raise ValueError(
                "Krea2 Flow Euler received a latent with "
                f"{samples.shape[1]} channels, but this Krea2 model requires "
                f"{expected_channels}. Use an empty latent (it will be promoted "
                "automatically) or encode the input with the Krea2 Qwen VAE."
            )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index")) if add_noise == "enable" else torch.zeros_like(samples)
        sigmas = _krea_sigmas(model, int(steps), float(denoise), float(shift_override)).to(samples.device)
        sampler = comfy.samplers.KSAMPLER(
            sample_krea2_flow,
            extra_options={
                "flow_solver": str(flow_solver),
                "pc3_gamma": float(pc3_gamma),
                "pc3_tolerance": float(pc3_tolerance),
            },
        )
        callback = latent_preview.prepare_callback(model, int(steps))
        output = comfy.sample.sample_custom(
            model, noise, float(cfg), sampler, sigmas, positive, negative, samples,
            noise_mask=latent.get("noise_mask"), callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )
        latent["samples"] = output
        return (latent,)


NODE_CLASS_MAPPINGS = {"Krea2FlowEulerSampler": Krea2FlowEulerSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2FlowEulerSampler": "Krea2 Flow Sampler (Native Shift)"}
