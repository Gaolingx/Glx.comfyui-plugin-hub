# Krea2 Flow Euler Sampler

Independent Krea2-only sampler. It uses Krea2's FLUX `ModelSamplingFlux`
trajectory and stable Flow-Matching solvers: Euler (default), Heun,
variable-step AB2, UniPC2 Diffusers x0 (BH2), and damped Diffusers-grid PC3.
It intentionally does not
modify or depend on the Anima sampler.

Start with: 20 steps, CFG 1.0, denoise 1.0, `shift_override = 0` (use the
model's own shift). Start with Euler; Heun costs almost twice as many model
evaluations. UniPC2 Diffusers x0 is a single-call-per-step Krea-adapted port
of the user's preferred Anima solver; it retains BH2 and lower-order final
steps but intentionally omits Anima/Cosmos tail policy. Connect a normal
Krea2 model, conditioning, empty latent, and Qwen VAE decoder as usual.

For higher detail, select `pc3_diffusers_damped`. It uses third-order x0
prediction plus adaptively damped endpoint correction, with warmup,
lower-order final steps, and shifted-linear grid tail protection. Recommended
starting values are `pc3_gamma = 1.0` and `pc3_tolerance = 0.005`; PC3 requires
roughly 1.5-1.8x as many model evaluations at typical Turbo step counts.
