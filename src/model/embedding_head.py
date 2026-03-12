# src/model/embedding_head.py

import torch
import torch.nn.functional as F
import torch.distributed as dist

from src.model.common import init_parallel
from src.utils.context import get_context, set_context, reset_context

class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, tp=(1, 0), tp_group=None):
        super(VocabParallelEmbedding, self).__init__()

        (self.tp_world_size, self.tp_rank), self.tp_group = tp, tp_group
        assert self.tp_world_size == 1 or (self.tp_world_size > 1 and tp_group is not None), "RowParallelLinear requires tp_world_size > 1 and a valid tp_group"

        self.num_embeddings = num_embeddings
        self.padded_num_embeddings = int((num_embeddings + self.tp_world_size - 1) // self.tp_world_size) * self.tp_world_size

        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_world_size
        self.embedding_dim = embedding_dim

        self.weight = torch.nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        setattr(self.weight, 'weight_loader', self.weight_loader)
    
    def weight_loader(self, param: torch.nn.Parameter, loaded_weights: torch.Tensor):

        # 计算加载 loaded_weights 的开始索引和结束索引
        loaded_weights_start_point = min(self.tp_rank * self.num_embeddings_per_partition, loaded_weights.size(0))
        loaded_weights_end_point = min((self.tp_rank + 1) * self.num_embeddings_per_partition, loaded_weights.size(0))
        shard_size = loaded_weights_end_point - loaded_weights_start_point

        with torch.no_grad():
            sharded_weights = loaded_weights.narrow(0, loaded_weights_start_point, shard_size)
            # 加载有效权重
            param.data[:shard_size].copy_(sharded_weights)
            # 填充 0 权重
            param.data[shard_size:].zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 筛选属于当前 tp_rank 的 token_id
        mask = (
            (x >= self.tp_rank * self.num_embeddings_per_partition) &
            (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) &
            (x < self.num_embeddings)
        )
        # 将非法索引 token_id 置 0
        idx = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        out = F.embedding(idx, self.weight)

        if self.tp_world_size > 1:
            # 清除掉非法索引对应的 embedding 输出: (B, T, 1) * (B, T, C)
            out = mask.unsqueeze(-1) * out
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        return out

    @classmethod
    def test(cls):
        # 初始化并行环境
        tp_group, _ = init_parallel(tp_size=2, backend="gloo")
        tp_rank = dist.get_rank(group=tp_group)
        global_rank = dist.get_rank()
        tp_world_size = dist.get_world_size(group=tp_group)

        vocab_size = 10
        embedding_dim = 4

        # 构造完整 embedding weight
        full_weight = torch.arange(vocab_size * embedding_dim).reshape(vocab_size, embedding_dim).float()

        # 输入 token
        tokens = torch.tensor([
            [0, 3, 5, 8],
            [1, 4, 6, 9]
        ])

        # 初始化 TP embedding
        emb = cls(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group
        )

        # 加载 shard 权重
        emb.weight_loader(emb.weight, full_weight)

        # forward
        output = emb(tokens)

        # 参考结果
        expected = torch.nn.functional.embedding(tokens, full_weight)

        # 验证
        if torch.allclose(output, expected, atol=1e-5):
            if tp_rank == 0:
                print(f"[Rank {global_rank}] VocabParallelEmbedding 测试通过！输出形状: {output.shape}")
        else:
            if tp_rank == 0:
                print(f"[Rank {global_rank}] 结果不一致！")
                print("实际结果:")
                print(output)
                print("预期结果:")
                print(expected)
            raise AssertionError(f"Rank {global_rank} 计算错误")

        dist.destroy_process_group()

class ParallelLMHead(VocabParallelEmbedding):
    '''ColumnParallelLinear
    '''
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 tp=(1, 0),
                 tp_group=None
    ):
        super().__init__(num_embeddings, embedding_dim, tp=tp, tp_group=tp_group)

    # x: (B, T, C)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            last_token = context.cu_seqlens_q[1:] - 1
            x = x[last_token].contiguous()

        # (B, T, embedding_dim) * (embedding_dim, self.num_embeddings_per_partition) = (B, T, self.num_embeddings_per_partition)
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_world_size > 1:
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_world_size)] if self.tp_rank == 0 else None
            # (B, T, self.num_embeddings_per_partition) * self.tp_world_size
            # = (B, T, self.num_embeddings_per_partition*self.tp_world_size)
            # = (B, T, self.padded_num_embeddings)
            dist.gather(logits, gather_list=all_logits, group=self.tp_group, group_dst=0)
            # concatenate
            if self.tp_rank == 0:
                # (B, T, self.padded_num_embeddings) -> (B, T, self.num_embeddings)
                logits = torch.cat(all_logits, dim=-1)
                logits = logits[..., :self.num_embeddings]

        return logits

    @classmethod
    def test(cls):
        # 初始化并行环境
        torch.manual_seed(42)
        tp_group, _ = init_parallel(tp_size=2, backend="gloo")
        tp_rank = dist.get_rank(group=tp_group)
        tp_world_size = dist.get_world_size(group=tp_group)
        global_rank = dist.get_rank()

        vocab_size = 12
        embedding_dim = 8
        
        # 设置 Context (Prefill 阶段)
        cu_seqlens = torch.tensor([0, 3, 8], device="cpu") 
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens)

        # 构造数据
        full_weight = torch.randn(vocab_size, embedding_dim)
        hidden_states = torch.randn(8, embedding_dim)

        # 初始化
        head = cls(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group
        )

        # 加载分片权重
        head.weight_loader(head.weight, full_weight)

        # 执行 Forward
        output = head(hidden_states)

        # 验证逻辑
        if tp_rank == 0:
            last_token_indices = cu_seqlens[1:] - 1
            expected_x = hidden_states[last_token_indices]
            expected_logits = torch.nn.functional.linear(expected_x, full_weight)

            if torch.allclose(output, expected_logits, atol=1e-5):
                print(f"[Rank {global_rank}] ParallelLMHead 测试通过！")
            else:
                print(f"[Rank {global_rank}] 验证失败！误差: {(output - expected_logits).abs().max()}")
                raise AssertionError
        
        # 清理 context
        reset_context()
        # 确保所有进程都完成测试后再销毁进程组
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    # VocabParallelEmbedding.test()
    ParallelLMHead.test()