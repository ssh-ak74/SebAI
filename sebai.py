import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SebConfig:
    vocab_size = 256
    context_length = 128
    embedding_dim = 256
    num_heads = 8
    num_layers = 4
    dropout = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.qkv = nn.Linear(config.embedding_dim, config.embedding_dim * 3)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length
                )
            ).view(
                1,
                1,
                config.context_length,
                config.context_length
            )
        )

    def forward(self, x):
        batch, seq, channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(
            self.mask[:, :, :seq, :seq] == 0,
            float("-inf")
        )

        attention = F.softmax(attention, dim=-1)

        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch, seq, channels)

        return self.proj(output)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)

        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class SebAI(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.embedding_dim)

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None):
        batch, seq = input_ids.shape

        positions = torch.arange(
            seq,
            device=input_ids.device
        )

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

config = SebConfig()
model = SebAI(config)

parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Parameters: {parameters:,}")
