import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # Single conv for Q,K,V
        self.proj = nn.Conv2d(channels, channels, 1)
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Normalization for improved training stability
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        x = self.norm(x)  # Apply normalization first
        B, C, H, W = x.shape
        
        # Get Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape to (B, num_heads, head_dim, H*W)
        q = q.view(B, self.num_heads, self.head_dim, H*W)
        k = k.view(B, self.num_heads, self.head_dim, H*W)
        v = v.view(B, self.num_heads, self.head_dim, H*W)
        
        # Attention
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Combine heads
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        
        return self.proj(out) + x  # Residual connection

class TemporalAttention(nn.Module):
    """Attention module that explicitly models temporal relationships between frames"""
    def __init__(self, channels, context_window=2, latent_channels=4):
        super().__init__()
        self.channels = channels
        self.context_window = context_window
        self.latent_channels = latent_channels
        
        # Current frame query projector
        self.to_q = nn.Conv2d(channels, channels, 1)
        
        # Context frames need preprocessing to match channel dimensions
        self.context_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_channels, channels // 2, 3, padding=1),
                nn.GroupNorm(8, channels // 2),
                nn.GELU(),
                nn.Conv2d(channels // 2, channels, 1)
            ) for _ in range(context_window)
        ])
        
        # Key and value projectors operate on processed context
        self.to_k = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in range(context_window)
        ])
        self.to_v = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in range(context_window)
        ])
        
        # Final projection
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Layer normalization
        self.norm = nn.GroupNorm(8, channels)
        
        # Scaling factor for attention
        self.scale = channels ** -0.5
        
    def forward(self, x, context_frames):
        """
        x: Current frame [B, C, H, W]
        context_frames: Context frames [B, T, C_latent, H, W] where:
            - T is context_window
            - C_latent is latent_channels (typically 4)
        """
        B, C, H, W = x.shape
        T = context_frames.shape[1]
        
        # Input normalization for stability
        x = self.norm(x)
        
        # Project current frame to query
        q = self.to_q(x)
        # Dynamic reshaping based on actual dimensions
        q_flat = q.flatten(2)  # [B, C, H*W]
        q_heads = q_flat.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Process each context frame
        k_list, v_list = [], []
        for i in range(min(T, len(self.to_k))):
            # Get context frame
            context_i = context_frames[:, i]
            
            # Process context frame to match channel dimensions
            processed_context = self.context_processors[i](context_i)
            
            # Generate keys and values
            k_i = self.to_k[i](processed_context)
            v_i = self.to_v[i](processed_context)
            
            # Dynamic reshaping based on actual dimensions
            k_flat = k_i.flatten(2)  # [B, C, H*W]
            v_flat = v_i.flatten(2)  # [B, C, H*W]
            
            k_heads = k_flat.permute(0, 2, 1)  # [B, H*W, C]
            v_heads = v_flat.permute(0, 2, 1)  # [B, H*W, C]
            
            k_list.append(k_heads)
            v_list.append(v_heads)
        
        # Concatenate keys and values across temporal dimension
        k = torch.cat(k_list, dim=1)  # [B, T*H*W, C]
        v = torch.cat(v_list, dim=1)  # [B, T*H*W, C]
        
        # Compute attention weights
        attn = torch.matmul(q_heads, k.transpose(-2, -1)) * self.scale  # [B, H*W, T*H*W]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H*W, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to spatial dimension
        
        return self.proj(out) + x  # Residual connection

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection if channels don't match
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, emb=None):
        # Main branch
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        
        # Add embedding if provided
        if emb is not None:
            h = h + emb
            
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        
        # Residual connection
        return h + self.residual(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim):
        super().__init__()
        # Simple single-head attention matching checkpoint dimensions
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(query_dim, query_dim)
        self.to_v = nn.Linear(query_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)
        
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).transpose(1, 2)  # [b, h*w, c]
        
        if context is not None:
            context = context.squeeze(-1).squeeze(-1)  # Remove spatial dims from embedding
        else:
            context = x_flat
            
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Simple scaled dot-product attention
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # Project back
        out = self.proj(out)
        
        # Restore spatial dimensions
        out = out.transpose(1, 2).view(b, c, h, w)
        
        return out

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, context_window=2):
        super().__init__()
        
        # Time embedding projection to match out_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.GELU()
        )
        
        # Action embedding projection to match out_ch
        self.action_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.GELU()
        )
        
        # Basic components
        self.cross_attn = CrossAttention(out_ch, out_ch)
        self.residual = ResidualBlock(in_ch, out_ch)
        self.self_attention = SelfAttention(out_ch)
        
        # Temporal attention for processing context frames (with proper channel handling)
        self.temporal_attn = TemporalAttention(out_ch, context_window, latent_channels=4)
        self.use_temporal = context_window > 0
        
        # Normalization layers
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # Sampling layer (up or down)
        if up:
            self.sampling = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.sampling = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t, action_emb, context_frames=None):
        # Project both embeddings to match channel dimensions
        time_emb = self.time_mlp(t)[:, :, None, None]
        action_emb = self.action_mlp(action_emb)[:, :, None, None]
        
        # Combine embeddings
        combined_emb = time_emb + action_emb
        
        # Rest of the forward pass
        h = self.residual(x, combined_emb)
        h = self.norm1(h)
        h = self.self_attention(h)
        
        # Apply temporal attention if context frames are provided
        if self.use_temporal and context_frames is not None:
            h = self.temporal_attn(h, context_frames)
            
        h = self.norm2(h)
        h = self.cross_attn(h, combined_emb)
        return self.sampling(h)

class UNet(nn.Module):
    def __init__(self, in_channels=4, time_emb_dim=256, action_dim=7, context_window=2):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, time_emb_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Store context window
        self.context_window = context_window
        
        # Context processing - ensure 4 channels output
        self.context_processor = nn.Sequential(
            nn.Conv2d(in_channels * context_window, 16, 3, padding=1),  # More initial channels
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GroupNorm(8, 8),
            nn.GELU(),
            nn.Conv2d(8, 4, 3, padding=1)
        )
        
        # Initial projection - input is 8 channels (4 + 4)
        self.conv0 = nn.Conv2d(8, 32, 3, padding=1)  # Fixed input channels to 8
        
        # Downsampling path
        self.downs = nn.ModuleList([
            Block(32, 64, time_emb_dim, context_window=context_window),
            Block(64, 128, time_emb_dim, context_window=context_window),
            Block(128, 256, time_emb_dim, context_window=context_window),
            Block(256, 512, time_emb_dim, context_window=context_window),
        ])
        
        # Middle block for additional processing
        self.mid_block = ResidualBlock(512, 512)
        self.mid_attn = SelfAttention(512)
        
        # Upsampling path
        self.ups = nn.ModuleList([
            Block(512 + 512, 256, time_emb_dim, up=True, context_window=context_window),
            Block(256 + 256, 128, time_emb_dim, up=True, context_window=context_window),
            Block(128 + 128, 64, time_emb_dim, up=True, context_window=context_window),
            Block(64 + 64, 32, time_emb_dim, up=True, context_window=context_window),
        ])
        
        # Final output
        self.output = nn.Conv2d(32, 4, 1)  # Fixed to output 4 channels

    def forward(self, x, prev_x, timestep, action):
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Process action context
        action_embs = []
        if action.shape[1] == 1:  # If context_window is 1
            action_i = action.squeeze(1)
            emb_i = self.action_embedding(action_i)
            emb_i = self.action_mlp(emb_i)
            action_emb = emb_i
        else:  # For context_window >= 2
            # Process each action in the context window
            for i in range(min(self.context_window, action.shape[1])):
                action_i = action[:, i]
                emb_i = self.action_embedding(action_i)
                emb_i = self.action_mlp(emb_i)
                action_embs.append(emb_i)
            
            # Combine action embeddings with weights (more recent = more important)
            action_weights = torch.tensor([0.7 ** i for i in range(len(action_embs))],
                                       device=x.device)[None, :, None]
            action_embs = torch.stack(action_embs, dim=1)
            action_emb = (action_embs * action_weights).sum(dim=1) / action_weights.sum()
        
        # Extract context frames for temporal attention
        context_frames = None
        if len(prev_x.shape) == 5:  # Shape: [B, CW, C, H, W]
            # Already in the right format for the temporal attention
            context_frames = prev_x
            
            # Flatten for the context processor
            B, CW, C, H, W = prev_x.shape
            prev_x_flat = prev_x.reshape(B, -1, H, W)
            
            # Process through context processor
            context = self.context_processor(prev_x_flat)
        else:
            # Process through context processor directly
            context = self.context_processor(prev_x)
        
        # Concatenate current and processed context
        x = torch.cat([x, context], dim=1)  # Combine current frame with processed context
        x = self.conv0(x)  # Initial convolution
        
        # Store residuals for skip connections
        residual_inputs = []
        
        # Downsampling path
        for down in self.downs:
            x = down(x, t, action_emb, context_frames)
            residual_inputs.append(x)
        
        # Middle block
        x = self.mid_block(x)
        x = self.mid_attn(x)
        
        # Upsampling path with skip connections
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t, action_emb, context_frames)
        
        return self.output(x)
