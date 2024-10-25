import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Multi-Head Self-Attention module with scalable heads and embedding dimension
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Ensure embedding dimension is divisible by number of heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        
        # Linear transformations for query, key, and value
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    # Attention mechanism based on scaled dot-product attention
    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = key.size(-1)
        scaled_score = score / torch.sqrt(torch.tensor(dim_key, dtype=torch.float32))
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    # Separate heads for parallel processing in multi-head attention
    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    # Forward pass for multi-head self-attention
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        # Linear projections for query, key, and value
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Separate heads for each projection
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Perform attention calculation and combine results
        attention, weights = self.attention(query, key, value)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention.view(batch_size, -1, self.embed_dim)
        output = self.combine_heads(concat_attention)
        return output

# Transformer block combining multi-head attention and feedforward layers
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # Feedforward network within transformer block
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        
        # Layer normalization and dropout for regularization
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # Forward pass of transformer block
    def forward(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# Vision Transformer (ViT) model for image classification
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_layers,  d_model, num_heads, mlp_dim, patchDim, dropout=0.1):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patchDim
        
        

        self.patch_size = patch_size
        d_model=d_model-d_model%num_heads
        self.d_model = d_model
        self.num_layers = num_layers

        # self.rescale = nn.Identity()  # We assume the input is already normalized
        
        # Positional encoding for patch embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        
        # Transformer layers
        self.enc_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    # Extract image patches for ViT input
    def extract_patches(self, images):
        # This is the PyTorch equivalent of tf.image.extract_patches
        batch_size = images.size(0)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        return patches

    # Forward pass through Vision Transformer
    def forward(self, x):
        x = self.patch_proj(x)
        x = x + self.pos_emb
        for layer in self.enc_layers:
            x = layer(x)
        return x


# Patch merging for downsampling in ViT-based architectures
class PatchMerging(nn.Module):
    def __init__(self, dim, dim_scale=2):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        
        self.reduction = nn.Linear(in_features=4 * dim, out_features=dim * dim_scale, bias=False)
        self.norm = nn.LayerNorm(4 * dim)  # Normalize over the concatenated features (4 * C)

    # Forward pass for patch merging
    def forward(self, x):
        HW = int(np.sqrt(x.shape[1]))
        B, L, C = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(B, HW, HW, C)

        # Perform the patch merging (taking every second element)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # Concatenate the patches along the channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)  # Reshape to (B, L/4, 4 * C)

        # Normalize and apply the linear reduction layer
        x = self.norm(x)
        x = self.reduction(x)

        return x

# Patch expansion for upsampling in ViT-based architectures
class PatchExpansion(nn.Module):
    def __init__(self, dim, dim_scale=2):
        super(PatchExpansion, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // 2, eps=1e-6)

    # Forward pass for patch expansion
    def forward(self, x):
        # Compute height and width based on the input patch size
        HW = int(np.sqrt(x.shape[1]))
        
        # Apply linear layer for expansion
        x = self.expand(x)
        
        # Get batch size (B), sequence length (L), and channels (C)
        B, L, C = x.shape
        
        # Reshape to match the desired output format
        x = x.reshape(B, HW, HW, C)
        
        # Reshape for patch expansion
        x = x.reshape(B, HW, HW, -1, C // 4)
        
        # Permute dimensions to match the desired layout
        x = x.permute(0, 1, 3, 2, 4)
        
        # Reshape back to the final expanded patch shape
        x = x.reshape(B, -1, HW * 2, C // 4)
        
        # Final reshaping
        x = x.reshape(B, -1, C // 4)
        
        # Apply Layer Normalization
        x = self.norm(x)
        
        return x
    
# Example Usage
def main():
    image_size = 224
    patch_size = 16
    num_layers = 12
    num_classes = 10
    d_model = 768
    num_heads = 12
    mlp_dim = 3072

    # Simulated input image
    x = torch.randn(4, 3, image_size, image_size)  # Batch of 4 images with 3 channels
    

    # Initialize and forward through the VisionTransformer
    vit = VisionTransformer(image_size, patch_size, num_layers,  d_model, num_heads, mlp_dim, d_model)
    patches = vit.extract_patches(x)
    output = vit(patches)

    print("Model output shape:", output.shape)  # Expect [batch_size, num_classes]


if __name__ == '__main__':
    main()