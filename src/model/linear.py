# src/model/linear.py

import torch
import torch.distributed as dist
from torch.nn import Parameter

from src.model.common import init_parallel

class LinearBase(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=True,
                 tp: tuple[int, int]=(1, 0)
        ):
        super().__init__()
        self.tp_world_size, self.tp_rank = tp

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

class ReplicatedLinear(LinearBase):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
    
    def weight_loader(self, param: torch.nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=True,
                 tp: tuple[int, int]=(1, 0)
        ):
        assert out_features % tp[0] == 0, f"out_features must be divisible by tp_world_size ({tp[0]}), but got {out_features}"
        super().__init__(in_features, int(out_features // tp[0]), bias, tp)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weights: torch.Tensor):
        tp_world_size, tp_rank = self.tp_world_size, self.tp_rank

        shard_size = loaded_weights.shape[0] // tp_world_size

        start = tp_rank * shard_size
        end = start + shard_size

        with torch.no_grad():
            param.data.copy_(loaded_weights[start:end])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform linear transformation using the partitioned weights
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    @classmethod
    def test(cls):
        tp_group, dp_group = init_parallel(tp_size=2)

        tp_rank = dist.get_rank(group=tp_group)
        tp_world_size = dist.get_world_size(group=tp_group)

        ddp_rank = dist.get_rank(group=dp_group)
        ddp_world_size = dist.get_world_size(group=dp_group)

        weight = torch.randn(2048, 512)
        bias = torch.randn(2048)
        l1 = cls(in_features=512, out_features=2048, bias=True, tp=(tp_world_size, tp_rank))
        l1.weight.weight_loader(l1.weight, weight)
        l1.bias.weight_loader(l1.bias, bias)
        input_tensor = torch.randn(4, 512)
        output = l1(input_tensor)
        print(f"TP Rank: {tp_rank}, DDP Rank: {ddp_rank}, Output Shape: {output.shape}")

        dist.destroy_process_group()

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self,
                 in_features: int,
                 out_features_list: list[int],
                 bias: bool=True,
                 tp: tuple[int, int]=(1, 0)):
        assert all(x % tp[0] == 0 for x in out_features_list), f"All out_features in out_features_list must be divisible by tp_world_size ({tp[0]}), but got {out_features_list}"
        super().__init__(in_features, sum(out_features_list), bias, tp)
        self.out_features_list = out_features_list

    def weight_loader(self, param: Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        tp_world_size, tp_rank = self.tp_world_size, self.tp_rank

        assert 0 <= loaded_weight_id < len(self.out_features_list), f"loaded_weight_id out of bounds: {loaded_weight_id}"
        assert self.out_features_list[loaded_weight_id] == loaded_weights.shape[0], f"Shape mismatch for weight {loaded_weight_id}: {self.out_features_list[loaded_weight_id]} != {loaded_weights.shape[0]}"
        assert all(x % tp_world_size == 0 for x in self.out_features_list), "All segments must be divisible by tp_world_size."

        out_features = self.out_features_list[loaded_weight_id]
        shard_size = out_features // tp_world_size
        loaded_weights_start_point = tp_rank * shard_size
        param_start_point = sum(self.out_features_list[:loaded_weight_id]) // tp_world_size

        with torch.no_grad():
            param.data[param_start_point:param_start_point + shard_size].copy_(
                loaded_weights[loaded_weights_start_point:loaded_weights_start_point + shard_size]
            )

    def weight_loader_all(self, param: Parameter, loaded_weights: torch.Tensor):
        tp_world_size, tp_rank = self.tp_world_size, self.tp_rank

        assert sum(self.out_features_list) == loaded_weights.shape[0], f"Shape mismatch: {sum(self.out_features_list)} != {loaded_weights.shape[0]}"
        assert all(x % tp_world_size == 0 for x in self.out_features_list), "All segments must be divisible by tp_world_size."

        offset_out_features = 0
        for i in range(len(self.out_features_list)):
            out_features = self.out_features_list[i]
            shard_size = out_features // tp_world_size

            loaded_weights_start_point = offset_out_features + tp_rank * shard_size
            param_start_point = offset_out_features // tp_world_size

            with torch.no_grad():
                param.data[param_start_point:param_start_point + shard_size].copy_(
                    loaded_weights[loaded_weights_start_point:loaded_weights_start_point + shard_size]
                )

            offset_out_features += out_features

    @classmethod
    def test(cls):
        tp_group, _ = init_parallel(tp_size=2)
        tp_rank = dist.get_rank(group=tp_group)
        tp_world_size = dist.get_world_size(group=tp_group)

        # 模拟 GQA 维度: Q(1024), K(256), V(256)
        in_features = 512
        out_features_list = [1024, 256, 256]
        
        # 准备原始权重
        q_w = torch.randn(1024, in_features)
        k_w = torch.randn(256, in_features)
        v_w = torch.randn(256, in_features)
        qkv_all_w = torch.cat([q_w, k_w, v_w], dim=0) # 模拟一次性加载的 Tensor

        # 测试分批加载 (weight_loader)
        l_sep = cls(in_features, out_features_list, bias=False, tp=(tp_world_size, tp_rank))
        l_sep.weight.weight_loader(l_sep.weight, q_w, loaded_weight_id=0)
        l_sep.weight.weight_loader(l_sep.weight, k_w, loaded_weight_id=1)
        l_sep.weight.weight_loader(l_sep.weight, v_w, loaded_weight_id=2)

        # 测试一次性加载 (weight_loader_all)
        l_all = cls(in_features, out_features_list, bias=False, tp=(tp_world_size, tp_rank))
        l_all.weight_loader_all(l_all.weight, qkv_all_w)

        # 验证：两种加载方式的结果必须完全相同
        assert torch.allclose(l_sep.weight, l_all.weight), "Loader 逻辑不一致！"
        
        # 验证：本地分片的数据是否真的是原始数据的正确分片
        # 以 K 为例：K 在 list 索引 1，起始位置应该是 sum([1024]) // 2 = 512
        # K 的分片大小应该是 256 // 2 = 128
        k_shard_expected = k_w[tp_rank*128 : (tp_rank+1)*128]
        k_shard_actual = l_sep.weight[512 : 512+128]
        assert torch.allclose(k_shard_expected, k_shard_actual), f"Rank {tp_rank} 的 K 分片加载错误"

        # 前向计算测试
        input_tensor = torch.randn(2, in_features)
        output = l_sep(input_tensor)
        
        # 预期输出形状: sum(out_features_list) // tp_world_size = 1536 // 2 = 768
        expected_shape = (2, 768)
        assert output.shape == expected_shape, f"输出形状错误: {output.shape} != {expected_shape}"

        if tp_rank == 0:
            print(f"MergedColumnParallelLinear 测试通过！")
            print(f"   本地权重形状: {l_sep.weight.shape}")
            print(f"   输出 Tensor 形状: {output.shape}")

        dist.destroy_process_group()

class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
        tp: tuple[int, int] = (1, 0)
    ):
        tp_world_size = tp[0]
        num_kv_heads = num_kv_heads or num_heads
        assert num_heads % tp_world_size == 0, f"num_heads ({num_heads}) is not divisible by tp_world_size ({tp_world_size})"
        assert num_kv_heads % tp_world_size == 0, f"num_kv_heads ({num_kv_heads}) is not divisible by tp_world_size ({tp_world_size})"
        self.head_size = head_size
        self.num_heads = num_heads // tp_world_size
        self.num_kv_heads = num_kv_heads // tp_world_size
        total_out_features = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_out_features, bias=bias, tp=tp)
    
    def weight_loader(self, param: torch.nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        assert load_weight_id in ["q", "k", "v"], f"Invalid load_weight_id: {load_weight_id}"
        # loaded_weights: [head_size*(num_heads|num_kv_heads), input_size]
        if load_weight_id == "q":
            # assert
            param_start_point = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == "k":
            param_start_point = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            param_start_point = self.head_size * (self.num_heads + self.num_kv_heads)
            shard_size = self.head_size * self.num_kv_heads
        
        _, tp_rank = self.tp_world_size, self.tp_rank
        loaded_weights_start_point = tp_rank * shard_size

        with torch.no_grad():
            param.data[param_start_point:param_start_point + shard_size].copy_(
                loaded_weights[loaded_weights_start_point:loaded_weights_start_point + shard_size]
            )

    @classmethod
    def test(cls):
        tp_group, _ = init_parallel(tp_size=2)
        tp_rank = dist.get_rank(group=tp_group)
        tp_world_size = dist.get_world_size(group=tp_group)

        # 模拟配置: 8个Q头, 2个KV头 (GQA), head_size=64
        input_size = 256
        head_size = 64
        num_heads = 8
        num_kv_heads = 2
        
        # 准备全量原始权重
        q_w_all = torch.randn(num_heads * head_size, input_size)
        k_w_all = torch.randn(num_kv_heads * head_size, input_size)
        v_w_all = torch.randn(num_kv_heads * head_size, input_size)

        # 初始化算子
        l_qkv = cls(
            input_size=input_size,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
            tp=(tp_world_size, tp_rank)
        )

        # 分三次加载权重
        l_qkv.weight_loader(l_qkv.weight, q_w_all, load_weight_id="q")
        l_qkv.weight_loader(l_qkv.weight, k_w_all, load_weight_id="k")
        l_qkv.weight_loader(l_qkv.weight, v_w_all, load_weight_id="v")

        # 验证数据正确性
        # 每张卡分到的 Q 头数 = 8 / 2 = 4, 长度 = 4 * 64 = 256
        # 每张卡分到的 KV 头数 = 2 / 2 = 1, 长度 = 1 * 64 = 64
        q_shard_size = (num_heads // tp_world_size) * head_size
        kv_shard_size = (num_kv_heads // tp_world_size) * head_size
        
        # 验证 Q 部分
        expected_q = q_w_all[tp_rank * q_shard_size : (tp_rank + 1) * q_shard_size]
        actual_q = l_qkv.weight[0 : q_shard_size]
        assert torch.allclose(expected_q, actual_q), f"Rank {tp_rank} Q 权重切分错误"

        # 验证 K 部分 (偏移量为 q_shard_size)
        expected_k = k_w_all[tp_rank * kv_shard_size : (tp_rank + 1) * kv_shard_size]
        actual_k = l_qkv.weight[q_shard_size : q_shard_size + kv_shard_size]
        assert torch.allclose(expected_k, actual_k), f"Rank {tp_rank} K 权重切分错误"

        # 5. 前向计算验证
        input_tensor = torch.randn(1, 16, input_size) # [batch, seq, hidden]
        output = l_qkv(input_tensor)
        
        # 预期输出宽度: (4 + 1 + 1) * 64 = 6 = 384
        expected_out_dim = (num_heads // tp_world_size + 2 * (num_kv_heads // tp_world_size)) * head_size
        assert output.shape == (1, 16, expected_out_dim), f"输出形状错误: {output.shape}"

        if tp_rank == 0:
            print(f"QKVColumnParallelLinear 测试通过！")
            print(f"   单卡 Q 头数: {l_qkv.num_heads}, KV 头数: {l_qkv.num_kv_heads}")
            print(f"   本地权重形状: {l_qkv.weight.shape}")
            print(f"   输出 Tensor 形状: {output.shape}")

        dist.destroy_process_group()

class RowParallelLinear(LinearBase):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=True,
                 tp: tuple[int, int]=(1, 0),
                 tp_group=None
        ):
        tp_world_size, tp_rank = tp
        assert tp_world_size == 1 or (tp_world_size > 1 and tp_group is not None), "RowParallelLinear requires tp_world_size > 1 and a valid tp_group"
        assert in_features % tp_world_size == 0, f"in_features must be divisible by tp_world_size ({tp_world_size}), but got {in_features}"
        super().__init__(int(in_features // tp_world_size), out_features, bias, tp)
        self.tp_group = tp_group
        if bias and tp_rank != 0:
            with torch.no_grad():
                self.bias.zero_()
    
    def weight_loader(self, param: torch.nn.Parameter, loaded_weights: torch.Tensor):
        # 处理 Bias 的特殊逻辑: 保证一个 TP 组内只有一个进程有 bias，否则 all_reduce 会把 bias 加两遍
        if param is self.bias:
            with torch.no_grad():
                if self.tp_rank == 0:
                    param.copy_(loaded_weights)
                else:
                    param.zero_()
            return

        # 处理 Weight 的分片逻辑
        tp_world_size, tp_rank = self.tp_world_size, self.tp_rank
        
        shard_size = loaded_weights.shape[1] // tp_world_size
        start = tp_rank * shard_size
        end = start + shard_size

        with torch.no_grad():
            param.copy_(loaded_weights.narrow(1, start, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.tp_world_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.tp_group)
            
        return result

    @classmethod
    def test(cls):
        # 初始化并行环境
        tp_group, _ = init_parallel(tp_size=2, backend="gloo")
        tp_rank = dist.get_rank(group=tp_group)
        global_rank = dist.get_rank()
        tp_world_size = dist.get_world_size(group=tp_group)

        in_features, out_features = 8, 4
        
        # 确定性数值填充
        full_weight = torch.arange(out_features * in_features).reshape(out_features, in_features).float()
        # 偏置用 1 填充
        full_bias = torch.ones(out_features)
        # 输入用 1 填充
        input_full = torch.ones(2, in_features)

        # 初始化算子
        l_row = cls(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp=(tp_world_size, tp_rank),
            tp_group=tp_group
        )

        # 加载权重
        l_row.weight_loader(l_row.weight, full_weight)
        l_row.weight_loader(l_row.bias, full_bias)

        # 前向计算
        shard_size = in_features // tp_world_size
        input_shard = input_full[:, tp_rank*shard_size : (tp_rank+1)*shard_size]
        output = l_row(input_shard)
        
        # 验证结果
        expected_output = torch.nn.functional.linear(input_full, full_weight, full_bias)
        
        if torch.allclose(output, expected_output, atol=1e-5):
            if tp_rank == 0:
                print(f"[Rank {global_rank}] 测试通过！结果维度: {output.shape}")
        else:
            if tp_rank == 0:
                print(f"[Rank {global_rank}] 结果不一致！")
                print(f"实际结果:\n{output}")
                print(f"预期结果:\n{expected_output}")
            raise AssertionError(f"Rank {global_rank} 计算错误")

        dist.destroy_process_group()

if __name__ == "__main__":
    # ColumnLinear.test()
    # MergedColumnParallelLinear.test()
    # QKVColumnParallelLinear.test()
    RowParallelLinear.test()