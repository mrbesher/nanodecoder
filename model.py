import torch
from torch import nn
from torch.nn import functional as F

from config import NanoDecoderConfig


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class MLP(nn.Module):
    def __init__(self, config: NanoDecoderConfig):
        super().__init__()
        self.fc_1 = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.fc_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = SwiGLU()

    def forward(self, x):
        x = self.activation(self.fc_1(x))
        x = self.fc_2(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NanoDecoderConfig):
        super().__init__()
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.input_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_hidden_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout

    def forward(self, x: torch.FloatTensor):
        # batch_size, seq_len, _ = x.size()

        x = self.input_proj(x)

        qkv_states = self.query_key_value(x)
        q, k, v = qkv_states.chunk(3, dim=-1)

        # batch size, seq len, num heads, per head hidden size
        q = q.unflatten(-1, [self.num_attention_heads, self.attention_hidden_size])
        k = k.unflatten(-1, [self.num_attention_heads, self.attention_hidden_size])
        v = v.unflatten(-1, [self.num_attention_heads, self.attention_hidden_size])

        # batch size, num heads, seq len, per head head size
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )

        # batch size, seq len, hidden size
        context = context.transpose(1, 2).flatten(-2).contiguous()

        context = self.o_proj(context)

        return context


class Layer(nn.Module):
    def __init__(self, config: NanoDecoderConfig):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.FloatTensor):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class NanoDecoder(nn.Module):
    def __init__(self, config: NanoDecoderConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.LongTensor):
        # input_ids.shape: batch size, seq len

        # batch size, seq len, hidden size
        embeddings = self.embedding(input_ids)

        for layer in self.layers:
            embeddings = layer(embeddings)

        return embeddings


class NanoDecoderForCausalLM(nn.Module):
    def __init__(self, config: NanoDecoderConfig):
        super().__init__()
        self.model = NanoDecoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def post_init(self):
        # tie embeddings and lm_head
        self.lm_head.weight = self.model.embedding.weight

    def forward(self, input_ids: torch.LongTensor):
        # input_ids.shape: batch size, seq len

        # batch size, seq len, hidden size
        embeddings = self.model(input_ids)
        # batch size, seq len, vocab size
        logits = self.lm_head(embeddings)

        return logits



if __name__ == "__main__":
    config = NanoDecoderConfig()

    swiglu = SwiGLU()
    x = torch.randn(1, 10)
    x_chunked = x.chunk(2, dim=-1)
    assert torch.allclose(swiglu(x), F.silu(x_chunked[1]) * x_chunked[0])

    mlp = MLP(config)
    x = torch.randn(1, config.hidden_size)
    out = mlp(x)
    assert out.shape == torch.Size([1, config.hidden_size])

    att = CausalSelfAttention(config)
    x = torch.randn(1, 10, config.hidden_size)
    out = att(x)
    assert out.shape == torch.Size([1, 10, config.hidden_size])

    layer = Layer(config)
    x = torch.randn(1, 10, config.hidden_size)
    out = layer(x)
    assert out.shape == torch.Size([1, 10, config.hidden_size])

    decoder = NanoDecoder(config)
    input_ids = torch.LongTensor([[2, 5, 7], [8, 1, 5]])
    out = decoder(input_ids)
    assert out.shape == torch.Size([input_ids.shape[0], input_ids.shape[1], config.hidden_size])

    model = NanoDecoderForCausalLM(config)
    x = torch.LongTensor([[2, 5, 7], [8, 1, 5]])
    out = model(x)
    assert out.shape == torch.Size([x.shape[0], x.shape[1], config.vocab_size])