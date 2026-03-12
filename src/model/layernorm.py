# src/model/layrernorm.py

import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        self.eps = eps

    @property
    def gamma(self):
        return self.weight


    @torch.compile
    def rms_forward_compile(self, x):
        # out = (x / sqrt(mean(x^2) + eps)) ⊙ weight
        _rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # 计算 1/RMS
        return x * _rms_inv * self.weight

    def rms_forward_with_residual_compile(self, x, residual):
        return self.rms_forward_compile(x) + residual
    
    def forward_compile(self, x, residual=None):
        if residual is not None:
            return self.rms_forward_with_residual_compile(x, residual)
        else:
            return self.rms_forward_compile(x)


    def rms_forward(self, x):
        # out = (x / sqrt(mean(x^2) + eps)) ⊙ weight
        _rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # 计算 1/RMS
        return x * _rms_inv * self.weight

    def rms_forward_with_residual(self, x, residual):
        return self.rms_forward(x) + residual
    
    def forward(self, x, residual=None):
        if residual is not None:
            return self.rms_forward_with_residual(x, residual)
        else:
            return self.rms_forward(x)

    @classmethod
    def compare_compile(cls):
        # 定义测试数据
        model_dim = 2048
        configs = {
            "small": torch.randn(1, 1024, model_dim).cuda(),
            "mid": torch.randn(16, 1024, model_dim).cuda(),
            "large": torch.randn(256, 1024, model_dim).cuda()
        }

        model = cls(gamma=torch.ones(model_dim)).cuda()

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
    LayerNorm.compare_compile()