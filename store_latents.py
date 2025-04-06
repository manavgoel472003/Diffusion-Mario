import torch
from Diff import Encoder
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

def store_latents(buffer_path="buffer.pt", model_path="autoencoder_final.pt", batch_size=32):
    # Load buffer
    replay_buffer_capacity = 25000
    storage = LazyMemmapStorage(replay_buffer_capacity)
    replay_buffer = TensorDictReplayBuffer(storage=storage)
    replay_buffer.load(buffer_path)

    # Get images from buffer
    images = torch.tensor(replay_buffer["state"], dtype=torch.float32) / 255.0
    del replay_buffer  # Free memory

    # Load encoder
    encoder = Encoder()
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    # Store latents
    all_latents = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = batch.permute(0, 3, 1, 2)  # NHWC to NCHW
            batch = batch.to(device)
            
            # Encode
            latents = encoder(batch)
            all_latents.append(latents.cpu())
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{len(images)} images")

    # Concatenate and save all latents
    all_latents = torch.cat(all_latents, dim=0)
    torch.save(all_latents, "latents.pt")
    print(f"Saved latents of shape {all_latents.shape} to latents.pt")

if __name__ == "__main__":
    store_latents() 