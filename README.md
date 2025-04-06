# Diffusion-Mario

You can also use your own VAE, but we found that it had very condusing outputs, with hard to recognise Mario figures, so we ended fine tunning it
Link for VAE-Mario model : https://drive.google.com/drive/folders/1Y9Xhiz87YsbD9FcL4zObY04G2QvLDY1F?usp=drive_link

# Temporal Diffusion Model for Game-Like Video Generation

A temporally-coherent diffusion model that generates high-quality video sequences with game-like visuals. This model implements advanced temporal modeling techniques to ensure smooth, consistent frame transitions and accurate action-conditioned generation.

## Data Generation
- The `rl.ipynb` uses OpenAI Gym and DQN alogrithm to generate frames and actions, that are then used for VAE to generate latents.
- The latents are then used in UNet for the rest

## Features

- **Temporal Context Awareness**: Uses a context window of multiple frames for coherent video generation
- **Action-Conditioned Generation**: Generates appropriate visual responses to gameplay actions
- **Specialized Temporal Attention**: Custom TemporalAttention module that explicitly models relationships between frames
- **Multi-Loss Training**: Combines standard diffusion noise prediction with temporal consistency losses
- **DDIM Sampling**: Efficient deterministic sampling for faster and more stable inference
- **Game-Style Post-Processing**: Subtle enhancements to match the visual style of games

## Project Structure

- `train_diffusion.py` - Training script with temporal consistency loss
- `diffusion_inference.py` - Video generation with temporal smoothing
- `unet.py` - Enhanced UNet architecture with temporal attention
- `diffusion.py` - Diffusion process and DDIM sampling implementation 
- `Diff.py` - Autoencoder for latent space encoding/decoding


## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- tqdm
- PIL
- numpy
- matplotlib
- opencv-python

## Installation

```bash
git clone https://github.com/manavgoel472003/Diffusion-Mario.git
cd Diffusion-Mario
```

## Usage

### Training

```bash
python train_autoencoder.py
python train_diffusion.py
```

Key training parameters:
- `num_epochs`: Number of training epochs (default: 20)
- `context_window`: Number of previous frames to consider (default: 2)
- `temporal_consistency_weight`: Weight for temporal loss (default: 0.3)

### Video Generation

```bash
python diffusion_inference.py
```

Key inference parameters:
- `num_frames`: Number of frames to generate (default: 100)
- `context_window`: Temporal context size (default: 2)
- `guidance_scale`: Classifier-free guidance strength (default: 4.0)
- `temporal_smooth_window`: Additional smoothing window size (default: 4)

## Technical Details

### Temporal Architecture

The model enhances a standard U-Net with temporal components:

1. **Context Window**: Processes multiple previous frames (default: 2) to inform the current frame generation
2. **TemporalAttention**: Cross-attention between current frame features and temporal context
3. **Context Processing**: Network to combine and process previous frame information
4. **Action Context**: Weighted combination of previous actions for consistent action response

### Training Losses

The model is trained with three complementary losses:

1. **Noise Loss** (weight: 0.7): Standard diffusion denoising objective
2. **Next State Loss** (weight: 0.5): Predicting the next frame given current frame
3. **Temporal Consistency Loss** (weight: 0.3): Ensuring temporal coherence across multiple frames

### DDIM Sampling

The implementation uses Denoising Diffusion Implicit Models (DDIM) for faster, deterministic sampling. This approach:

- Reduces the number of sampling steps
- Ensures more stable transitions between frames
- Supports classifier-free guidance for better quality

## Results

Generated videos show significant improvements in temporal coherence compared to standard frame-by-frame generation approaches. The model maintains visual consistency while responding to input actions.

## Acknowledgements

This project builds upon prior work in diffusion models, particularly extending them with temporal capabilities for video generation.


## NOTE:
- We need long training for both VAE and UNet to get good frame generation, at least 100+ epochs.
