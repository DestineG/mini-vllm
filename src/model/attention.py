# src/model/attention.py

import triton 
import triton.language as tl
from src.utils.context import get_context
import torch

@triton.jit
def store_kvcache_kernel(
    key_ptr,            # (num_tokens, num_kv_heads, head_dim)
    value_ptr,          # (num_tokens, num_kv_heads, head_dim)
    k_cache_ptr,        # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,        # (num_slots, num_kv_heads, head_dim)
    slot_mapping_ptr,   # kv 中第一个 token 的 slot_mapping 索引
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    '''kernel 函数，启动方式为 <num_tokens, num_kv_heads>，每个线程处理一个 token 的一个 head 的数据，将其存储到对应的 slot 中
    '''
    # 获取当前线程负责处理的 token 起始索引
    token_idx = tl.program_id(0)
    # 一个 slot 可以存储一个 token 的所有 head 的数据
    # 根据 token 索引从 slot_mapping 中获取对应的 slot 索引
    # slot_mapping 可以理解为是一个 virtual memory 地址转换表
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return
    # 获取当前线程负责处理的 head 起始索引
    head_idx = tl.program_id(1)

    # 计算当前线程负责处理的 head 的维度索引范围
    head_offsets = tl.arange(0, head_dim)

    # 计算输入偏移地址
    input_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim + head_offsets
    # 计算输出偏移地址
    cache_offset = slot_idx * num_kv_heads * head_dim + head_idx * head_dim + head_offsets

    # 从输入中加载 key 和 value 数据
    key_data = tl.load(key_ptr + input_offset)
    value_data = tl.load(value_ptr + input_offset)

    # 将 key 和 value 数据存储到对应的 slot 中
    tl.store(k_cache_ptr + cache_offset, key_data)
    tl.store(v_cache_ptr + cache_offset, value_data)

def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    num_tokens, num_kv_heads, head_dim = key.shape

    # 确保输入张量是 row 连续的
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert value.shape == key.shape, "K and V must have the same shape"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match the number of tokens"

    # 启动 triton kernel
    store_kvcache_kernel[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        k_cache_ptr=k_cache,
        v_cache_ptr=v_cache,
        slot_mapping_ptr=slot_mapping,
        num_kv_heads=num_kv_heads,  # type: ignore
        head_dim=head_dim,          # type: ignore
        block_size=block_size       # type: ignore
    )

@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    '''kernel 函数，启动方式为 <max_seq_len/BLOCK_M, num_heads, num_seqs>

    其中 max_seq_len 是当前 batch 中序列的最大长度，num_seqs 是 batch 中序列的数量

    tl.program_id(2) 用于定位当前线程处理的是哪个序列的数据，tl.program_id(0) 和 tl.program_id(1) 则用于定位当前线程处理的序列的哪个 block 和哪个 head 的数据
    '''
    # (T*B, H, D)
    Q_block_idx = tl.program_id(0)  # Q_block_idx, block_size = BLOCK_M
    Q_head_idx = tl.program_id(1)   # Q_head_idx
    seq_idx = tl.program_id(2)      # sequence_idx

    # 计算 KV_head_idx，这种计算方式隐含了 tl.program_id(1) 维度的 size = num_heads
    KV_head_idx = Q_head_idx // (num_heads // num_kv_heads)

    # 计算当前线程负责处理的 Q 的起始索引
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    # 如果当前 Q_block_idx 超过了序列长度，则不进行计算，直接返回
    if Q_block_idx * BLOCK_M >= seq_len:
        return
    
    # 计算当前线程负责处理的 Q 的维度索引范围
    block_range = Q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    head_dim_range = tl.arange(0, head_dim)

    # 计算当前线程处理的 Q 的偏移地址，size=(BLOCK_M, head_dim)
    Q_ptr = Q + (seq_start + block_range[:, None]) * num_heads * head_dim + Q_head_idx * head_dim + head_dim_range[None, :]

    # 处理序列长度不足 BLOCK_M 的情况
    Q_mask = block_range < seq_len
    Q = tl.load(Q_ptr, mask=Q_mask[:, None], other=0.0)

    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    row_max = tl.full([BLOCK_M], dtype=tl.float32) - 1e10
    block_result = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # 计算 KV block 数量
    num_KV_blocks = tl.cdiv(seq_len, BLOCK_N)

    # 遍历 KV block
    # Todo: 此处会遍历整个句子的 KV block，可以添加边界条件(例如: all(qk_mask) is False)，只遍历当前 Q block 相关的 KV block 来减少计算量
    for block_idx in range(num_KV_blocks):
        # 计算当前线程负责处理的 K 和 V 的起始索引
        KV_block_start = block_idx * BLOCK_N
        KV_block_range = KV_block_start + tl.arange(0, BLOCK_N)

        # 处理序列长度不足 BLOCK_N 的情况
        KV_mask = KV_block_range < seq_len  # (BLOCK_N,)
        # KV_block_range 作为列， head_dim_range 作为行，读取 K, shape=(head_dim, BLOCK_N)
        K_ptr = K + (seq_start + KV_block_range[None, :]) * num_kv_heads * head_dim + KV_head_idx * head_dim + head_dim_range[:, None]
        K = tl.load(K_ptr, mask=KV_mask[None, :], other=0.0)

        # Q@K^T, size=(BLOCK_M, head_dim)@(head_dim, BLOCK_N)=>(BLOCK_M, BLOCK_N)
        # Q@K^T, size=(BLOCK_M, head_dim)@(head_dim, valid_BLOCK_N)=>(BLOCK_M, valid_BLOCK_N)
        qk = tl.dot(Q, K)
        qk = qk * scale

        # casual mask
        qk_mask = (block_range[:, None] + seq_start) >= (KV_block_range[None, :] + seq_start)   # (BLOCK_M, BLOCK_N)
        # casual mask 和 KV_mask 共同作用，屏蔽掉无效的 qk 计算结果 (BLOCK_M, valid_BLOCK_N)
        qk = tl.where(qk_mask & KV_mask[None, :], qk, -1e10)

        # online softmax
        row_local_max = tl.max(qk, axis=1)  # (BLOCK_M,)
        row_max_new = tl.maximum(row_max, row_local_max)  # (BLOCK_M,)
        alpha = tl.exp(row_max - row_max_new)  # (BLOCK_M,)

        # 修正历史 softmat 结果 exp(x-row_max_new)，该修正是作用在 qkv 的结果上，效果等同于作用在 qk 上
        block_result = block_result * alpha[:, None]
        # 当前 block 的 softmax 结果 exp(qk-row_max_new)
        p = tl.exp(qk - row_max_new[:, None])

        # V block 读取方式同 K
        V_ptr = V + (seq_start + KV_block_range[None, :]) * num_kv_heads * head_dim + KV_head_idx * head_dim + head_dim_range[:, None]
        V = tl.load(V_ptr, mask=KV_mask[None, :], other=0.0)

        # 计算 qkv
        block_result = block_result + tl.dot(p.to(V.dtype), V)

        # 更新 onlinesoftmax 参数
        row_sum = row_sum * alpha + tl.sum(p, axis=1)   # 历史 row_sum 修正 + 当前 的 p.sum
        row_max = row_max_new
    
    # 应用 row_sum
    block_result = block_result / row_sum[:, None]

    # 将结果写回输出张量 O
    O_ptr = O + (seq_start + block_range[:, None]) * num_heads * head_dim + Q_head_idx * head_dim + head_dim_range[None, :]
    tl.store(O_ptr, block_result.to(O.dtype.element_ty) , mask=Q_mask[:, None])

def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    
    # 确保输入张量是 row 连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # 分配输出张量
    output = torch.empty_like(q)

    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs
    # 根据 head_dim 选择合适的 BLOCK_M 和 BLOCK_N，充分利用 GPU 的缓存，减少内存访问延迟
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # 计算 num_seqs
    num_seqs = cu_seqlens.shape[0] - 1

    # 计算 max_seq_len
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # 启动 triton kernel
    flash_attention_varlen_kernel[triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs](
        Q=q, K=k, V=v, O=output,
        cu_seqlens_q_ptr=cu_seqlens,
        scale=scale,
        num_heads=num_heads,            # type: ignore
        num_kv_heads=num_kv_heads,      # type: ignore
        head_dim=head_dim,              # type: ignore
        BLOCK_M=BLOCK_M,                # type: ignore
        BLOCK_N=BLOCK_N                 # type: ignore
    )

    return output

@triton.jit
def paged_attention_decode_kernel(
    output_ptr,                     # (num_tokens, num_heads, head_dim)
    query_ptr,                      # (num_tokens, num_heads, head_dim)
    k_cache_ptr,                    # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,                    # (num_slots, num_kv_heads, head_dim)
    block_tables_ptr,               # (num_tokens, max_num_blocks) -> list[slot_idx]
    context_lens_ptr,               # (num_tokens,) -> context_lens
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    '''kernel 函数，启动方式为 <num_tokens, num_heads>，每个线程处理一个 token 的一个 head 的数据
    '''
    # 获取当前线程负责处理的 token 和 head 的索引
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # 计算 KV_head_idx，这种计算方式隐含了 tl.program_id(1) 维度的 size = num_heads
    KV_head_idx = head_idx // (num_heads // num_kv_heads)

    # 从 context_lens 中获取当前 token 的 context 长度
    context_len = tl.load(context_lens_ptr + token_idx)

    # 加载 query
    Q_range = tl.arange(0, head_dim)
    Q_ptr = query_ptr + token_idx * num_heads * head_dim + head_idx * head_dim + Q_range
    Q = tl.load(Q_ptr)

    result = tl.zeros([head_dim], dtype=tl.float32)
    qk_sum = 0.0
    qk_max = -1e10

    # 根据 context_len 计算当前 token 相关的 KV block 数量
    num_chunks = tl.cdiv(context_len, BLOCK_N)

    # 遍历 KV block
    for chunk_idx in range(num_chunks):
        # 计算 KV_block 位置
        KV_block_start = chunk_idx * BLOCK_N
        KV_block_range = KV_block_start + tl.arange(0, BLOCK_N)

        # 处理序列填不满当前 BLOCK_N 的情况
        KV_mask = KV_block_range < context_len  # (BLOCK_N,)

        qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
        # 遍历 K_block 每一个 token
        for i in range(BLOCK_N):
            K_token_idx = KV_block_start + i

            # 如果 token_idx 超过了 context_len，则不进行计算，直接跳过
            if K_token_idx < context_len:
                block_idx = K_token_idx // block_size
                block_offset = K_token_idx % block_size

                # 计算当前 K 在 block_table 中的 block 索引位置
                block_table_offset = token_idx * max_num_blocks + block_idx
                # 从 block_table 中获取当前 block 的物理块编号
                block_physical_idx = tl.load(block_tables_ptr + block_table_offset)

                if block_physical_idx != -1:
                    # 此种取 K 的方式，隐含着 block 中有 block_size 个连续存储的 token
                    K_offset = (
                        block_physical_idx * block_size * num_kv_heads * head_dim
                                        + block_offset * num_kv_heads * head_dim
                                                        + KV_head_idx * head_dim
                                                                        + Q_range
                    )
                    K = tl.load(k_cache_ptr + K_offset)
                    # (1, head_dim)@(head_dim,) -> (1,)
                    score = tl.sum(Q * K) * scale

                    # 使用 mask 将计算结果放到 qk 的索引 i 处
                    mask_i = tl.arange(0, BLOCK_N) == i
                    qk = tl.where(mask_i, score, qk)
        
        # online softmax 计算
        qk_local_max = tl.max(qk)
        qk_max_new = tl.maximum(qk_max, qk_local_max)
        alpha = tl.exp(qk_max - qk_max_new)
    
        # 修正历史 softmat 结果
        result = result * alpha
        qk_sum = qk_sum * alpha

        p = tl.exp(qk - qk_max_new)

        # 遍历 V_block 每一个 token，计算加权和
        for i in range(BLOCK_N):
            V_token_idx = KV_block_start + i

            # 如果 token_idx 超过了 context_len，则不进行计算，直接跳过
            if V_token_idx < context_len:
                block_idx = V_token_idx // block_size
                block_offset = V_token_idx % block_size

                # 计算当前 V 在 block_table 中的 block 索引位置
                block_table_offset = token_idx * max_num_blocks + block_idx
                # 从 block_table 中获取当前 block 的物理块编号
                block_physical_idx = tl.load(block_tables_ptr + block_table_offset)

                if block_physical_idx != -1:
                    # block 中有 block_size 个连续存储的 token
                    V_offset = (
                        block_physical_idx * block_size * num_kv_heads * head_dim
                                        + block_offset * num_kv_heads * head_dim
                                                        + KV_head_idx * head_dim
                                                                        + Q_range
                    )
                    V = tl.load(v_cache_ptr + V_offset)
                    mask_i = tl.arange(0, BLOCK_N) == i
                    qk_i = tl.sum(tl.where(mask_i, p, 0.0))

                    result = result + qk_i * V
                    qk_sum = qk_sum + qk_i
        qk_max = qk_max_new
    
    # 应用 qk_sum
    result = result / qk_sum

    # result 存放到 output_ptr
    result_ptr = output_ptr + token_idx * num_heads * head_dim + head_idx * head_dim + Q_range
    tl.store(result_ptr, result.to(output_ptr.dtype.element_ty))

def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    num_tokens = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    query = query.contiguous()
    output = torch.empty_like(query)

    BLOCK_N = 64 if head_dim <= 128 else 32

    paged_attention_decode_kernel[(num_tokens, num_heads)](
        output_ptr=output,
        query_ptr=query,
        k_cache_ptr=k_cache,
        v_cache_ptr=v_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        scale=scale,                        # type: ignore
        num_heads=num_heads,                # type: ignore
        num_kv_heads=num_kv_heads,          # type: ignore
        head_dim=head_dim,                  # type: ignore
        block_size=block_size,              # type: ignore
        max_num_blocks=max_num_blocks,      # type: ignore
        BLOCK_N=BLOCK_N                     # type: ignore
    )

    return output

class Attntion(torch.nn.Module):
    def __init__(self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,   # type: ignore
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache = self.v_cache

        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if k.dim() == 4:
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            o = paged_attention_decode(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
