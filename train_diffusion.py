import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from unet import UNet
from diffusion import Diffusion
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

def train_diffusion_model(
    num_epochs=50,              # Increased training epochs
    batch_size=48,               # Adjusted batch size
    learning_rate=2e-4,          # Slightly increased learning rate
    device=None,
    latents_path="latents.pt",
    buffer_path="buffer.pt",
    next_state_weight=0.5,       # Increased next state prediction weight
    noise_weight=0.7,            # Adjusted noise prediction weight
    temporal_consistency_weight=0.3,  # New temporal consistency loss
    checkpoint_dir="unet_trained_temporal",  # New checkpoint directory
    resume=False,
    save_every=5,                # Save more frequently
    context_window=2             # Increased context window for better temporal modeling
):
    # Force CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available! Using GPU")
    else:
        print("CUDA is not available. Training requires GPU.")
        print("Exiting...")
        return
    
    print(f"Using device: {device}")
    torch.cuda.empty_cache()  # Clear any cached memory
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load latents with weights_only
    print(f"Loading latents from {latents_path}")
    original_latents = torch.load(latents_path, weights_only=True).to(device)
    
    # Load buffer using TensorDictReplayBuffer
    print(f"Loading buffer from {buffer_path}")
    replay_buffer_capacity = 25000
    storage = LazyMemmapStorage(replay_buffer_capacity)
    replay_buffer = TensorDictReplayBuffer(storage=storage)
    replay_buffer.load(buffer_path)
    
    # Get original actions from buffer
    original_actions = torch.tensor(replay_buffer["action"], dtype=torch.long, device=device)
    del replay_buffer  # Free memory
    
    # Augment the dataset by 4x
    num_augmentations = 3
    num_original = len(original_latents)
    
    # Expand latents
    latents = original_latents.repeat(num_augmentations, 1, 1, 1)
    
    # Create new random actions for each augmentation
    actions = torch.empty(num_original * num_augmentations, dtype=torch.long, device=device)
    for i in range(num_augmentations):
        if i == 0:
            # Keep original actions for first copy
            actions[i * num_original:(i + 1) * num_original] = original_actions
        else:
            # Random actions for other copies
            actions[i * num_original:(i + 1) * num_original] = torch.randint(
                0, 7, (num_original,), device=device
            )
    
    print(f"Original dataset size: {num_original}")
    print(f"Augmented dataset size: {len(latents)}")
    print(f"Latents shape: {latents.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Create context windows for previous latents with more frames for better temporal modeling
    prev_latents_list = []
    for i in range(context_window):
        shift = i + 1
        prev = torch.roll(latents, shifts=shift, dims=0)
        # Fill in the first few frames that don't have history
        prev[:shift] = prev[shift:shift+1].repeat(shift, 1, 1, 1)
        prev_latents_list.append(prev)
    
    # Stack previous latents along new dimension [batch, context_window, channels, height, width]
    prev_latents = torch.stack(prev_latents_list, dim=1)
    
    # Create context window for actions
    prev_actions_list = []
    for i in range(context_window):
        shift = i + 1
        prev_action = torch.roll(actions, shifts=shift, dims=0)
        prev_action[:shift] = prev_action[shift]  # Fill in first frames
        prev_actions_list.append(prev_action)
    
    # Stack previous actions along new dimension [batch, context_window]
    prev_actions = torch.stack(prev_actions_list, dim=1)
    
    # Create next frames for prediction targets (not used as inputs)
    next_latents_list = []
    next_actions_list = []
    
    # Create sequence of 3 future frames for temporal consistency targets
    for i in range(1, 4):
        # Get future frames as targets
        next_latent = torch.roll(latents, shifts=-i, dims=0)
        # Fill in the last few frames that don't have future
        next_latent[-i:] = next_latent[-(i+1)]
        next_latents_list.append(next_latent)
        
        # Get future actions (only used for analysis, not for model inputs)
        next_action = torch.roll(actions, shifts=-i, dims=0)
        next_action[-i:] = next_action[-(i+1)]
        next_actions_list.append(next_action)
    
    # Stack future frames and actions
    next_latents = torch.stack(next_latents_list, dim=1)
    next_actions = torch.stack(next_actions_list, dim=1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(latents, prev_latents, prev_actions, actions, next_latents)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and diffusion
    model = UNet(
        in_channels=4, 
        action_dim=7, 
        context_window=context_window
    ).to(device)
    diffusion = Diffusion(timesteps=1000, device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Cosine annealing learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                          T_max=num_epochs * len(dataloader), 
                                                          eta_min=learning_rate/10)
    
    # Initialize tracking variables
    start_epoch = 0
    best_loss = float('inf')
    
    # Try to load latest checkpoint if resume is True
    if resume:
        latest_checkpoint = None
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], 
                                     key=os.path.getctime)
        except FileNotFoundError:
            pass
        
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            print(f"Resuming from epoch {start_epoch}")
    
    # Define temporal consistency loss function
    def temporal_consistency_loss(model, x, prev_x, prev_action, curr_action, next_latents):
        """
        Calculate temporal consistency loss by predicting the next frame
        and comparing with ground truth next frame.
        
        Only uses past and current information.
        """
        # Set up for model to predict next frame at t=0
        t_zero = torch.zeros(x.shape[0], device=device, dtype=torch.long)
        
        # Create action tensor for current step
        current_action_tensor = curr_action.unsqueeze(1)
        
        # First predict the immediate next frame
        predicted_next = model(x, prev_x, t_zero, torch.cat([prev_action[:, 1:], current_action_tensor], dim=1))
        
        # Calculate loss against ground truth next frame
        next_frame_loss = F.mse_loss(predicted_next, next_latents[:, 0])
        
        # The model has now predicted one step into the future
        # For autoregressive prediction (without future information), we would:
        # 1. Use the predicted frame as part of the new context
        # 2. Use the current frame as the oldest context frame
        # 3. Predict again using only known information
        
        # But we don't need multiple steps for training - the single-step loss is sufficient
        # to encourage temporal consistency
        
        return next_frame_loss
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        total_noise_loss = 0
        total_next_state_loss = 0
        total_temporal_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (latent, prev_latent, prev_action, curr_action, next_latent) in enumerate(progress_bar):
            # Sample timestep
            t = torch.randint(0, diffusion.timesteps, (latent.shape[0],), device=device).long()
            t_zero = torch.zeros_like(t)
            
            optimizer.zero_grad()
            
            # Add noise to the latent
            noise = torch.randn_like(latent, device=device)
            noisy_latent = diffusion.q_sample(latent, t, noise)
            
            # Create action context (previous actions + current action)
            action_context = torch.cat([prev_action[:, 1:], curr_action.unsqueeze(1)], dim=1) 
            
            # Predict noise with context
            predicted_noise = model(noisy_latent, prev_latent, t, action_context)
            
            # Calculate standard noise prediction loss
            noise_loss = F.mse_loss(noise, predicted_noise)
            
            # Next state prediction with context
            predicted_next = model(latent, prev_latent, t_zero, action_context)
            next_state_loss = F.mse_loss(predicted_next, next_latent[:, 0])
            
            # Temporal consistency loss - only using past+current information to predict future
            temp_loss = temporal_consistency_loss(
                model, latent, prev_latent, prev_action, curr_action, next_latent
            )
            
            # Combined loss with temporal consistency
            loss = (noise_weight * noise_loss + 
                   next_state_weight * next_state_loss + 
                   temporal_consistency_weight * temp_loss)
            
            # Backprop
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            total_loss += loss.item()
            total_noise_loss += noise_loss.item()
            total_next_state_loss += next_state_loss.item()
            total_temporal_loss += temp_loss.item()
            
            progress_bar.set_postfix({
                "total_loss": total_loss / (batch_idx + 1),
                "noise_loss": total_noise_loss / (batch_idx + 1),
                "next_loss": total_next_state_loss / (batch_idx + 1),
                "temp_loss": total_temporal_loss / (batch_idx + 1)
            })
        
        # Calculate average losses
        avg_total_loss = total_loss / len(dataloader)
        avg_noise_loss = total_noise_loss / len(dataloader)
        avg_next_state_loss = total_next_state_loss / len(dataloader)
        avg_temporal_loss = total_temporal_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Average Total Loss: {avg_total_loss:.6f}")
        print(f"Average Noise Loss: {avg_noise_loss:.6f}")
        print(f"Average Next State Loss: {avg_next_state_loss:.6f}")
        print(f"Average Temporal Loss: {avg_temporal_loss:.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save regular checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'total_loss': avg_total_loss,
                'noise_loss': avg_noise_loss,
                'next_state_loss': avg_next_state_loss,
                'temporal_loss': avg_temporal_loss,
                'best_loss': best_loss,
                'context_window': context_window
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'total_loss': avg_total_loss,
                'noise_loss': avg_noise_loss,
                'next_state_loss': avg_next_state_loss,
                'temporal_loss': avg_temporal_loss,
                'best_loss': best_loss,
                'context_window': context_window
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_diffusion_model(
        num_epochs=20,  # Reduced from 150 to 20 epochs
        batch_size=48,
        learning_rate=2e-4,
        next_state_weight=0.5,
        noise_weight=0.7,
        temporal_consistency_weight=0.3,
        checkpoint_dir="unet_trained_temporal",
        context_window=2,
        resume=False
    ) 