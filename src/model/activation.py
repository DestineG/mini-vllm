# src/model/activation.py

import torch
import torch.nn.functional as F

class SiluAndMul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.compile
    def forward_compile(self, x):
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
    
    def forward(self, x):
        # 语义越明确，优化越好，所以此处使用 chunk 而非索引切分
        # (..., 2 * dim) -> (.., dim), (.., dim)
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
    
    @classmethod
    def compare_compile(cls):
        # 定义测试数据
        configs = {
            "small": torch.randn(4, 1024, 2048).cuda(),
            "mid": torch.randn(64, 1024, 2048).cuda(),
            "large": torch.randn(1024, 1024, 2048).cuda()
        }
        
        model = cls().cuda()

        # 针对每个尺寸预热，确保编译器生成对应的优化 Kernel
        print("Warming up...")
        for name, tensor in configs.items():
            for _ in range(10):
                _ = model(tensor)
                _ = model.forward_compile(tensor)

        # 正式测试
        results = {}
        print("Testing...")
        for name, tensor in configs.items():
            forward_times = []
            compile_times = []
            
            for _ in range(100):
                # 测量原生 forward
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(tensor)
                end.record()
                torch.cuda.synchronize()
                forward_times.append(start.elapsed_time(end))

                # 测量编译 forward_compile
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model.forward_compile(tensor)
                end.record()
                torch.cuda.synchronize()
                compile_times.append(start.elapsed_time(end))
            
            results[name] = {
                "forward": sum(forward_times) / len(forward_times),
                "compile": sum(compile_times) / len(compile_times),
                "shape": list(tensor.shape)
            }

        # 打印结果
        print("\n" + "="*60)
        print(f"{'Size':<10} | {'Shape':<20} | {'Native (ms)':<12} | {'Compile (ms)':<12} | {'Speedup'}")
        print("-" * 60)
        for name, data in results.items():
            speedup = data['forward'] / data['compile']
            shape_str = str(data['shape'])
            print(f"{name:<10} | {shape_str:<20} | {data['forward']:>11.4f} | {data['compile']:>11.4f} | {speedup:.2f}x")
        print("="*60)

if __name__ == "__main__":
    SiluAndMul.compare_compile()