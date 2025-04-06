import torch
import torch.nn as nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from Diff import Encoder, Decoder

class TrainAutoencoder:
    def __init__(self, path="buffer.pt", batch_size=32):
        # Load buffer
        replay_buffer_capacity = 25000
        storage = LazyMemmapStorage(replay_buffer_capacity)
        replay_buffer = TensorDictReplayBuffer(storage=storage)
        replay_buffer.load(path)

        full_length = len(replay_buffer["state"])
        small_length = full_length // 6
        self.train_images = torch.tensor(replay_buffer["state"][:small_length], dtype=torch.float32) / 255.0
        del replay_buffer

        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs=100, lr=1e-4):
        encoder = Encoder().to(self.device)
        decoder = Decoder().to(self.device)
        
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            for i in range(0, len(self.train_images), self.batch_size):
                batch = self.train_images[i:i + self.batch_size]
                batch = batch.permute(0, 3, 1, 2)  # NHWC to NCHW
                batch = batch.to(self.device)

                optimizer.zero_grad()
                
                # Forward pass
                latents = encoder(batch)
                reconstructed = decoder(latents)
                loss = criterion(reconstructed, batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

                if i % (self.batch_size * 10) == 0:
                    print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss.item():.6f}")

            avg_loss = total_loss / batches
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.6f}")

            # Save checkpoints periodically
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }, f'autoencoder_checkpoint_{epoch+1}.pt')

        # Save final model
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
        }, 'autoencoder_final.pt')

if __name__ == "__main__":
    trainer = TrainAutoencoder(path="buffer.pt", batch_size=32)
    trainer.train(epochs=100, lr=1e-4)