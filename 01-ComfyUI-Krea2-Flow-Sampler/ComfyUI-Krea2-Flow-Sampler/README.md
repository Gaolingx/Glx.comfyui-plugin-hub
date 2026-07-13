# Krea2 Flow Euler Sampler

Independent Krea2-only sampler. It uses Krea2's FLUX `ModelSamplingFlux`
trajectory and stable Flow-Matching solvers: Euler (default), Heun,
variable-step AB2, and UniPC2 Diffusers x0 (BH2). It intentionally does not
modify or depend on the Anima sampler.

Start with: 20 steps, CFG 1.0, denoise 1.0, `shift_override = 0` (use the
model's own shift). Start with Euler; Heun costs almost twice as many model
evaluations. UniPC2 Diffusers x0 is a single-call-per-step Krea-adapted port
of the user's preferred Anima solver; it retains BH2 and lower-order final
steps but intentionally omits Anima/Cosmos tail policy. Connect a normal
Krea2 model, conditioning, empty latent, and Qwen VAE decoder as usual.
