import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        """Initialize the diffusion process with given parameters."""
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x, t, noise_pred):
        """Reverse diffusion process (sampling)."""
        # Get alpha values for current timestep (with proper broadcasting)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)
        
        # Calculate mean
        mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred)
        
        # Add noise only if t > 0 (using torch.where for batched operations)
        noise = torch.randn_like(x)
        variance = torch.sqrt(beta)
        return torch.where(t.view(-1, 1, 1, 1) > 0, mean + variance * noise, mean)

    def p_losses(self, denoise_model, x_start, t, noise=None):
        """Calculate the loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        return F.mse_loss(noise, predicted_noise)
        
    def ddim_sample(self, model, x, prev_x, action, timesteps=100, eta=0.0, guidance_scale=4.0):
        """
        DDIM sampler for accelerated and deterministic sampling.
        
        Args:
            model: UNet model
            x: Initial latent [B, C, H, W]
            prev_x: Previous frames for context [B, T, C, H, W]
            action: Action tensor [B, T]
            timesteps: Number of sampling timesteps to use
            eta: Controls the amount of stochasticity (0 = deterministic)
            guidance_scale: Controls the guidance strength for classifier-free guidance
            
        Returns:
            The generated latent
        """
        # Create timestep sequence
        skip = self.timesteps // timesteps
        seq = list(range(0, self.timesteps, skip))
        if seq[-1] != self.timesteps - 1:
            seq.append(self.timesteps - 1)
        seq_next = [-1] + seq[:-1]
        
        # Ensure seq is in descending order (from T to 0)
        seq = seq[::-1]
        seq_next = seq_next[::-1]
        
        # Ready for sampling
        batch_size = x.shape[0]
        device = x.device
        
        # Start with the input latent (typically random noise or slightly noised previous frame)
        latent = x.clone()
        
        # Sample through the timesteps
        for i in range(len(seq) - 1):
            # Get current and next timestep
            t_curr = seq[i]
            t_next = seq_next[i]
            
            # Timestep as tensor
            t = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
            
            # Predict noise with classifier-free guidance
            with torch.no_grad():
                # Unconditional prediction (empty context)
                null_action = torch.zeros_like(action)
                null_context = torch.zeros_like(prev_x)
                noise_pred_uncond = model(latent, null_context, t, null_action)
                
                # Conditional prediction (with real context)
                noise_pred_cond = model(latent, prev_x, t, action)
                
                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Get alphas for current and next timestep
            alpha_cumprod_curr = self.alphas_cumprod[t_curr]
            alpha_cumprod_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            
            # Current x_0 estimate (predicted clean image)
            pred_x0 = (latent - torch.sqrt(1 - alpha_cumprod_curr).view(-1, 1, 1, 1) * noise_pred) / \
                     torch.sqrt(alpha_cumprod_curr).view(-1, 1, 1, 1)
            
            # Direction to prior x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_next).view(-1, 1, 1, 1) * noise_pred
            
            # Random noise component (0 if eta=0 for deterministic sampling)
            if eta > 0:
                sigma = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod_curr) * 
                                      (1 - alpha_cumprod_curr / alpha_cumprod_next))
                noise = torch.randn_like(latent)
                rand_noise = sigma.view(-1, 1, 1, 1) * noise
            else:
                rand_noise = 0
            
            # DDIM update rule
            latent = torch.sqrt(alpha_cumprod_next).view(-1, 1, 1, 1) * pred_x0 + dir_xt + rand_noise
        
        return latent