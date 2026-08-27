import math
import torch

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # 初始化参数 weight = [out_features, in_features]
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # 使用Xavier初始化权重
        std = math.sqrt(2 / (in_features + out_features))
        # trunc_normal_初始化权重，截断正态分布，均值为0，标准差为std，截断范围为[a,b]
        torch.nn.init.trunc_normal_(self.weight, mean=0, a=-3 * std, b=3 * std, std=std)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先检查x的维度,一般来说接受[..., in_features]的输入
        assert x.shape[-1] == self.in_features, f"Expected input with last dimension {self.in_features}, but got {x.shape[-1]}"
        # 使用einsum进行线性变换，输出[..., out_features]
        return torch.einsum('...i,oi->...o', x, self.weight)

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 取token_ids的最后一维作为索引，返回对应的嵌入向量
        # token_ids的形状为[..., seq_len]，返回的嵌入向量形状为[..., seq_len, embedding_dim]
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先将原始的dtype转换为float32，计算均方根归一化
        original_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x_norm = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_norm * self.weight).to(original_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_ff, d_model, device, dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(silu(self.w1(x)) * self.w2(x))   


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor: 
    # values和indices是max的返回值，其中，values返回沿dim的最大值
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # q, k, v的形状为[..., seq_len, d_k]
    d_k = q.shape[-1]
    # 计算注意力分数
    scores = torch.einsum('...qd,...kd->...qk', q, k) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    return torch.einsum('...qk,...kd->...qd', attn_weights, v)

class RoPE(torch.nn.Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k:{d_k} must be even for RoPE.")
        inverse_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        positions = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)

        rotate_theta = torch.einsum('i,j->ij', positions, inverse_freq)

        # 后续可以复用sin、cos的值，避免重复计算
        self.register_buffer("cos_cache", torch.cos(rotate_theta), persistent=False)
        self.register_buffer("sin_cache", torch.sin(rotate_theta), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # x的形状为[..., seq_len, d_k]
        # token_positions的形状为[..., seq_len]
        cos = self.cos_cache[token_positions]  # [..., seq_len, d_k//2]
        sin = self.sin_cache[token_positions]  # [..., seq_len, d_k//2]

        x_even = x[..., ::2]  # [..., seq_len, d_k//2]
        x_odd = x[..., 1::2]  # [..., seq_len, d_k//2]

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos

        return x_rotated



# MHA实现
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError(f"d_model:{d_model} must be divisible by num_heads:{num_heads}.")
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)
        elif theta is None and max_seq_len is None:
            self.rope = None
        else:
            raise ValueError("Both theta and max_seq_len must be provided for RoPE, or neither.")


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 符号：... = 任意 batch 维，T = seq_len，D = d_model，H = num_heads，K = d_head
        # 输入 x: [..., T, D]
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # Q/K/V 投影：[..., T, D]
        # 拆分最后的 D = H * K：[..., T, D] -> [..., T, H, K]
        # 拆头后转置，使得一个头内接受[..., T, K]的输入，方便后续计算注意力分数
        q = torch.reshape(self.q_proj(x), (*batch_shape, seq_len, self.num_heads, self.d_head)).transpose(-2, -3)  # [..., H, T, K]
        k = torch.reshape(self.k_proj(x), (*batch_shape, seq_len, self.num_heads, self.d_head)).transpose(-2, -3)  # [..., H, T, K]
        v = torch.reshape(self.v_proj(x), (*batch_shape, seq_len, self.num_heads, self.d_head)).transpose(-2, -3)  # [..., H, T, K]
        
        # RoPE 应当作用于拆头并交换轴后的 Q和K: [..., H, T, K]
        # token_positions 在 MHA 内应可广播为 [..., 1, T]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            rope_positions = token_positions.unsqueeze(-2)  # [..., 1, T]
            # 
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        # causal mask: [T, T]；广播到 attention scores [..., H, T, T]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))  # [1, 1, seq_len, seq_len]
        # 每个 head 独立对 T 个 token 做 attention: [..., H, T, K]
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)  # [..., num_heads, seq_len, d_head]

        # 合并 heads 的正确过程：[..., H, T, K] -> [..., T, H, K] -> [..., T, D]
        # 这里为什么不做concat？
        attn_output = attn_output.transpose(-2, -3).reshape(*batch_shape, seq_len, self.d_model)  # [..., T, D]

        # output projection: [..., T, D] -> [..., T, D]
        return self.output_proj(attn_output)

# Transformer Block
class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pre_norm
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError(f"Input sequence length {token_ids.shape[-1]} exceeds model context length {self.context_length}.")
        token_positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
