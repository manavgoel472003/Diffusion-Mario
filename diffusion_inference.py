import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from unet import UNet
from diffusion import Diffusion
from Diff import Encoder, Decoder
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import ImageDraw, ImageFilter, ImageEnhance
import torchvision.transforms as T

def image_to_video(frames_dir, output_name, fps=30):
    """
    Create a video from a directory of image frames
    
    Args:
        frames_dir: Directory containing the frames
        output_name: Name of the output video file
        fps: Frames per second for the video
    """
    frames_path = os.path.join(frames_dir, "frame_%04d.png")
    output_path = os.path.join(frames_dir, output_name)
    
    # Use OpenCV or FFMPEG command to create video
    try:
        # Try using opencv first
        img = cv2.imread(os.path.join(frames_dir, "frame_0000.png"))
        if img is not None:
            height, width, _ = img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            i = 0
            while True:
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                if not os.path.exists(frame_path):
                    break
                frame = cv2.imread(frame_path)
                out.write(frame)
                i += 1
                
            out.release()
            print(f"Video saved to {output_path}")
            return
    except Exception as e:
        print(f"OpenCV error: {e}, trying ffmpeg")
    
    # Fallback to ffmpeg if OpenCV fails
    try:
        os.system(f"ffmpeg -y -framerate {fps} -i {frames_path} -c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p {output_path}")
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"FFMPEG error: {e}")
        print("Could not create video. Please manually convert the frames to video.")

def load_image(image_path):
    """Load and preprocess image to correct shape (B, C, H, W)"""
    image = Image.open(image_path)
    image = image.resize((128, 128))
    # Convert to numpy array and normalize to [0, 1]
    image = np.array(image).astype(np.float32) / 255.0
    # Convert from (H, W, C) to (C, H, W)
    image = image.transpose(2, 0, 1)
    # Convert to tensor
    image = torch.from_numpy(image)
    return image

def apply_game_style_filter(img_tensor):
    """Apply post-processing to make the image more game-like"""
    # Convert tensor to PIL image
    # Ensure tensor has right dimensions (C,H,W)
    if img_tensor.dim() == 4:  # If B,C,H,W format
        img_tensor = img_tensor.squeeze(0)  # Remove batch dimension
    
    if img_tensor.dim() > 3:
        # Handle any other unexpected dimensions
        img_tensor = img_tensor.reshape(-1, img_tensor.shape[-2], img_tensor.shape[-1])
    
    img = T.ToPILImage()(img_tensor)
    
    # Enhance contrast and saturation
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Color(img)  # Color instead of Saturation
    img = enhancer.enhance(1.2)
    
    # Apply slight sharpening
    img = img.filter(ImageFilter.SHARPEN)
    
    # Optional: Quantize colors for more game-like appearance
    # img = img.quantize(256).convert('RGB')
    
    # Convert back to tensor
    return T.ToTensor()(img)

def inference(
    image_path="0.png",
    autoencoder_path="autoencoder_final.pt",
    diffusion_path="unet_trained_temporal/best_model.pt",  # Updated path to temporal model
    device="cuda",
    timesteps=1000,
    num_frames=100,
    context_window=2,  # Increased context window for temporal coherence
    output_path="diffusion_video",
    guidance_scale=4.0,  # Adjusted for temporal model
    ddim_sampling_eta=0.0,  # Deterministic sampling
    skip_steps=5,  # Smaller skip for more accurate inference
    denoise_strength=0.5,  # Moderate denoising
    temporal_smooth_window=4,  # For additional temporal consistency
    action_sequence=None,  # Sequence of actions to use for generation
    style_filter_strength=0.3  # Reduced style filter reliance with better model
):
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Set CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
        
    # Adjust context window if needed
    print(f"Context window size: {context_window}")
    
    # Load image and convert to latent
    image = load_image(image_path).to(device)
    
    # Load autoencoder
    sd = torch.load(autoencoder_path, map_location=device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    encoder.load_state_dict(sd['encoder_state_dict'])
    decoder.load_state_dict(sd['decoder_state_dict'])
    
    # Encode starting image
    with torch.no_grad():
        image_latent = encoder(image.unsqueeze(0))
    
    # Load diffusion model
    unet = UNet(in_channels=4, context_window=context_window).to(device)
    # Load with map_location to handle device differences
    model_data = torch.load(diffusion_path, map_location=device)
    unet.load_state_dict(model_data['model_state_dict'])
    unet.eval() # Set to evaluation mode
    
    diffusion = Diffusion(timesteps=timesteps, device=device)
    
    # Determine sampling steps - use more for higher quality
    # Adjusted for better balance of speed vs. quality
    sampling_timesteps = min(1000, max(100, timesteps // skip_steps))
    print(f"Using {sampling_timesteps} sampling steps")
    
    # Create action sequence if none provided
    if action_sequence is None:
        # Default to RIGHT action for all frames
        action_sequence = [3] * num_frames
        print(f"No action sequence provided, using default action (3)")
    
    # Initialize list to store latents for video
    all_latents = []
    all_images = []
    
    # Store previous latents for temporal conditioning
    prev_latents_buffer = []
    for _ in range(context_window):
        prev_latents_buffer.append(image_latent.clone())
    
    # Initialize action history
    action_history = [action_sequence[0]] * context_window
    
    # Generate frames
    for i in tqdm(range(num_frames)):
        current_action = action_sequence[i]
        
        # Update action history
        action_history = action_history[1:] + [current_action]
        
        # Stack previous latents for context
        prev_latents = torch.stack(prev_latents_buffer, dim=1)
        
        # Create action tensor
        action_tensor = torch.tensor([action_history], device=device)
        
        # Generate latent
        with torch.no_grad():
            # Start from previous frame with some noise for continuity
            start_latent = prev_latents_buffer[-1].clone()
            if i > 0:
                # Add some noise to previous frame (adjusted for temporal model)
                noise = torch.randn_like(start_latent) * denoise_strength
                start_latent = start_latent + noise
            
            # Sample using DDIM for faster, deterministic results
            latent = diffusion.ddim_sample(
                model=unet,
                x=start_latent,
                prev_x=prev_latents,
                action=action_tensor,
                timesteps=sampling_timesteps,
                eta=ddim_sampling_eta,
                guidance_scale=guidance_scale
            )
            
            # Apply temporal smoothing (weighted average with previous)
            if i > 0 and temporal_smooth_window > 0:
                # Calculate weights based on window size
                alpha = min(0.7, 1.0 / temporal_smooth_window)
                # Blend current prediction with previous
                latent = (1 - alpha) * prev_latents_buffer[-1] + alpha * latent
            
            # Update previous latents buffer
            prev_latents_buffer.append(latent.clone())
            prev_latents_buffer = prev_latents_buffer[-context_window:]
            
            # Store the latent
            all_latents.append(latent.clone())
            
            # Decode to image
            decoded_image = decoder(latent)
            
            # Ensure decoded_image has the expected dimensions
            if decoded_image.dim() == 4 and decoded_image.size(0) == 1:
                decoded_image = decoded_image.squeeze(0)
            
            # Apply game style filter
            style_img = apply_game_style_filter(decoded_image.cpu())
            
            # Blend original with filtered for controlled stylization
            if style_filter_strength < 1.0:
                orig_tensor = decoded_image.cpu()
                style_img = style_img * style_filter_strength + orig_tensor * (1 - style_filter_strength)
            
            # Save individual frame
            img_path = os.path.join(output_path, f"frame_{i:04d}.png")
            save_image(style_img, img_path)
            
            # Store image for video
            all_images.append(style_img.clone())
    
    # Create video from frames
    image_to_video(output_path, "output.mp4")
    
    return all_latents, all_images

if __name__ == "__main__":
    inference(
        timesteps=1000,
        guidance_scale=4.0,      # Adjusted for temporal model
        ddim_sampling_eta=0.0,   # Deterministic for cleaner results
        skip_steps=5,            # Balanced sampling steps
        denoise_strength=0.5,    # Less post-process denoising needed with better model
        temporal_smooth_window=4, # Additional smoothing for stable sequences
        style_filter_strength=0.3, # Less reliance on style filters
        context_window=2         # Match the context window from training
    ) 