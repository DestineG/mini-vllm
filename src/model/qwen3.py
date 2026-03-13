# src/model/qwen3.py

import math
import torch
import torch.distributed as dist

from src.model.linear import *
from src.model.layernorm import *
from src.model.rotary_embedding import *
from src.model.attention import *
from src.model.common import init_parallel
from src.model.activation import *
from src.utils.context import get_context, set_context
from src.model.embedding_head import *

class Qwen3Attention(torch.nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        block_size: int = 256,
        tp: tuple[int, int] = (1, 0),
        tp_group = None,
    ):
        super().__init__()
        (self.tp_world_size, self.tp_rank), self.tp_group = tp, tp_group
        self.head_dim = head_dim or (hidden_size // num_heads)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_heads_per_partition = self.num_heads // self.tp_world_size
        self.num_kv_heads_per_partition = self.num_kv_heads // self.tp_world_size

        self.scale = scale

        self.q_size = head_dim * self.num_heads_per_partition
        self.kv_size = head_dim * self.num_kv_heads_per_partition
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            bias=qkv_bias,
            tp=(self.tp_world_size, self.tp_rank)
        )

        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position,
        )

        self.attention = Attention(
            num_heads=self.num_heads_per_partition,
            head_dim=self.head_dim,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads_per_partition,
            block_size=block_size
        )

        self.o_proj = RowParallelLinear(
            in_features=head_dim * self.num_heads_per_partition,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor,      # (B * N, hidden_size) or (B, N, hidden_size)
                positions: torch.Tensor
    ) -> torch.Tensor:
        # (B * N, hidden_size)->
        # q: (B * N, num_heads_per_partition * head_dim)
        # k: (B * N, num_kv_heads_per_partition * head_dim)
        # v: (B * N, num_kv_heads_per_partition * head_dim)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q: (B * N, num_heads_per_partition * head_dim) -> (B * N, num_heads_per_partition, head_dim)
        # kv: (B * N, num_kv_heads_per_partition * head_dim) -> (B * N, num_kv_heads_per_partition, head_dim)
        if q.dim() == 2:
            q = q.view(-1, self.num_heads_per_partition, self.head_dim)
            k = k.view(-1, self.num_kv_heads_per_partition, self.head_dim)
            v = v.view(-1, self.num_kv_heads_per_partition, self.head_dim)
        # q: (B, N, num_heads_per_partition * head_dim) -> (B, N, num_heads_per_partition, head_dim)
        # kv: (B, N, num_kv_heads_per_partition * head_dim) -> (B, N, num_kv_heads_per_partition, head_dim)
        else:
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads_per_partition, self.head_dim)
            k = k.view(B, N, self.num_kv_heads_per_partition, self.head_dim)
            v = v.view(B, N, self.num_kv_heads_per_partition, self.head_dim)
        
        if self.qkv_bias is False:
            q, _ = self.q_norm(q)
            k, _ = self.k_norm(k)
        
        q, k = self.rotary_emb(positions, q, k)

        o = self.attention(q, k, v)

        o = self.o_proj(o)

        return o
    
    @classmethod
    def test(cls):
        tp_group, dp_group = init_parallel(tp_size=2)

        tp_rank = dist.get_rank(tp_group)
        tp_world_size = dist.get_world_size(tp_group)

        ddp_rank = dist.get_rank(dp_group)
        ddp_world_size = dist.get_world_size(dp_group)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # 获取全局 rank
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        B, N, H, D = 2, 256, 8, 16
        hidden_size = H * D
        num_heads = H
        head_dim = D

        attention = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=math.sqrt(head_dim),
            qkv_bias=False,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group,
        ).to(device)
        for name, param in attention.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

        x = torch.randn(B*N, hidden_size).to(device)
        # (N, ) -> (N*B,)
        positions = torch.arange(N, device=device).repeat(B)
        o = attention(x, positions)
        if tp_rank == 0:
            print(f"Rank {dist.get_rank()}, TP_Rank {tp_rank}, Output Sum: {o.sum().item():.4f}")
        
        # dist.barrier()
        dist.destroy_process_group()

class Qwen3MLP(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 bias: bool = True,
                 tp: tuple[int, int] = (1, 0),
                 tp_group = None
    ):
        super().__init__()
        scale = 2
        self.gate_up = MergedColumnParallelLinear(
            in_features=hidden_size,
            out_features_list=[intermediate_size] * scale,
            bias=bias,
            tp=tp
        )
        self.activation = SiluAndMul()
        assert intermediate_size * scale % 2 == 0, "intermediate_size * scale must be divisible by 2 for SiluAndMul activation"
        self.down = RowParallelLinear(
            in_features=intermediate_size * scale // 2,
            out_features=hidden_size,
            bias=False,
            tp=tp,
            tp_group=tp_group
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up: (B * N, hidden_size) -> (B * N, intermediate_size * scale / tp_world_size)
        gate_up = self.gate_up(x)
        # activation: (B * N, intermediate_size * scale / tp_world_size) -> (B * N, intermediate_size * scale / tp_world_size // 2)
        activated = self.activation(gate_up)
        # down: (B * N, intermediate_size * scale / tp_world_size // 2) -> (B * N, intermediate_size * scale // 2)
        o = self.down(activated)
        return o

    @classmethod
    def test(cls):
        tp_group, dp_group = init_parallel(tp_size=2, backend="gloo")

        tp_rank = dist.get_rank(tp_group)
        tp_world_size = dist.get_world_size(tp_group)

        ddp_rank = dist.get_rank(dp_group)
        ddp_world_size = dist.get_world_size(dp_group)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # 获取全局 rank
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        B, N, H, D = 2, 256, 8, 16
        hidden_size = H * D
        intermediate_size = hidden_size * 4

        mlp = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group,
        ).to(device)
        for name, param in mlp.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

        x = torch.randn(B*N, hidden_size).to(device)
        o = mlp(x)
        if tp_rank == 0:
            print(f"Rank {dist.get_rank()}, TP_Rank {tp_rank}, Output: {o.shape}, Output Sum: {o.sum().item():.4f}")
        
        # dist.barrier()
        dist.destroy_process_group()

class Qwen3DecoderLayer(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 head_dim: int,
                 scale: float = 1.0,
                 num_kv_heads: int | None = None,
                 rms_norm_eps: float = 1e-6,
                 qkv_bias: bool = False,
                 base: int = 10000,
                 max_position: int = 16384,
                 intermediate_size: int = 4 * 1024,
                 ffn_bias: bool = True,
                 block_size: int = 256,
                 tp: tuple[int, int] = (1, 0),
                 tp_group = None
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
            tp=tp,
            tp_group=tp_group
        )
        self.post_attention_layernorm = LayerNorm(gamma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
            tp=tp,
            tp_group=tp_group
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入归一化
        x, residual = self.input_layernorm(x, residual)
        context = get_context()

        # 计算位置编码
        if context.is_prefill and context.cu_seqlens_q is not None:
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            positions = torch.arange(x.size(0), device=x.device)
        else:
            positions = context.context_lens - 1

        # 注意力模块计算
        x = self.self_attn(x, positions)
        # 注意力结果归一化
        x, residual = self.post_attention_layernorm(x, residual)
        # ffn 模块
        x = self.mlp(x)
        return x, residual
    
    @classmethod
    def test(cls):
        tp_group, _ = init_parallel(tp_size=2, backend="gloo")
        tp_rank = dist.get_rank(tp_group)
        tp_world_size = dist.get_world_size(tp_group)

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B, N = 2, 512  # Batch=2, SeqLen=512
        hidden_size = 256
        num_heads = 8
        head_dim = hidden_size // num_heads
        
        # 实例化 DecoderLayer
        layer = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=hidden_size * 4,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group
        ).to(device)

        # 假设有两个句子，长度均为 N
        cu_seqlens = torch.tensor([0, N, 2*N], device=device, dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens)

        # 准备输入数据
        x = torch.randn(B * N, hidden_size, device=device)
        residual = torch.randn(B * N, hidden_size, device=device)

        # 正向传播
        out_x, out_res = layer(x, residual)

        if tp_rank == 0:
            print(f"Rank {dist.get_rank()}, TP_Rank {tp_rank}, Output X Shape: {out_x.shape}, Output Residual Shape: {out_res.shape}")
        
        dist.destroy_process_group()

class Qwen3Model(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        block_size: int = 256,
        tp: tuple[int, int] = (1, 0),
        tp_group = None
    ):
        super().__init__()
        self.token_embedding = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            tp=tp,
            tp_group=tp_group
        )
        self.layers = torch.nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_eps=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
                tp=tp,
                tp_group=tp_group
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x

class Qwen3ForCausalLM(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        block_size: int = 256,
        tp: tuple[int, int] = (1, 0),
        tp_group = None
    ):
        super().__init__()
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )

        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def test(cls):
        tp_group, _ = init_parallel(tp_size=2, backend="gloo")
        tp_rank = dist.get_rank(tp_group)
        tp_world_size = dist.get_world_size(tp_group)

        torch.manual_seed(42)
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # 超参数
        vocab_size = 1000
        hidden_size = 128
        num_heads = 8
        num_layers = 2
        B, N = 2, 128  # Batch=2, SeqLen=128

        # 实例化
        model = cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group
        ).to(device)

        for name, param in model.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

        # 模拟输入数据 (Prefill)
        input_ids = torch.randint(0, vocab_size, (B * N,), device=device)
        
        # 设置 Prefill Context
        cu_seqlens = torch.tensor([0, N, 2*N], device=device, dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens)

        # 正向传播
        logits = model(input_ids)

        # 验证结果
        if tp_rank == 0:
            print(f"--- Qwen3ForCausalLM Prefill Test ---")
            print(f"Input IDs Shape: {input_ids.shape}")
            print(f"Logits Shape: {logits.shape}") # 预期: (B*N, vocab_size / tp_world_size)
            print(f"Logits Sum: {logits.sum().item():.4f}")

        dist.destroy_process_group()

if __name__ == "__main__":
    # NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 -m src.model.qwen3
    # Qwen3Attention.test()
    # Qwen3MLP.test()
    # Qwen3DecoderLayer.test()
    Qwen3ForCausalLM.test()