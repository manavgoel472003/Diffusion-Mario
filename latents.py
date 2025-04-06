from diffusers import AutoencoderKL
import torch

class Variational_Encoder_Mario:

    def __init__(self, model_path="vae_mario", device=None):
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        self.model = AutoencoderKL.from_pretrained(model_path, torch_dtype=torch.float32).to(self.device)

    @staticmethod
    def process_images(images, reverse=False):
        if reverse:
            images = images.permute(0, 2, 3, 1)
        else:
            images = images.permute(0, 3, 1, 2)
        return images

    def get_latents(self, images, images_processed=False):
        """"
        Images : Float32 tensor
        Processed : If not in (Batch, channel, Height, Width) format set as False
        Returns Float32 tensor of latent with shape (batch_size, 4, 16, 16)
        """
        if not images_processed:
            images = self.process_images(images)
        images = images.to(self.device)
        return self.model.encode(images).latent_dist.sample()
    
    @torch.no_grad
    def get_images(self, latents, for_display=True):
        """
        Latents: Directly add the encoder latents to get images
        For_Display : Makes sure the tensor is display ready tensor, only needing numpy conversion
        Returns image tensor 
        """
        images = self.model.decode(latents).sample
        if for_display:
            images = self.process_images(images, reverse=True)
            if self.device != "cpu":
                images = images.cpu()
        
        return images


