import torch
import os

def print_environment_info():
    print("\n=== 环境变量信息 ===")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not Set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not Set')}")
    print(f"PATH: {os.environ.get('PATH', 'Not Set')}")

def test_pytorch_gpu():
    print("\n=== PyTorch GPU 测试 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        
        # 测试 GPU 计算
        print("\n执行矩阵乘法测试...")
        # CPU 测试
        cpu_start = torch.cuda.Event(enable_timing=True)
        cpu_end = torch.cuda.Event(enable_timing=True)
        cpu_start.record()
        A = torch.randn(1000, 1000)
        B = torch.randn(1000, 1000)
        C = torch.matmul(A, B)
        cpu_end.record()
        torch.cuda.synchronize()
        print(f"CPU 耗时: {cpu_start.elapsed_time(cpu_end)/1000:.4f} 秒")
        
        # GPU 测试
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        A = torch.randn(1000, 1000, device='cuda')
        B = torch.randn(1000, 1000, device='cuda')
        C = torch.matmul(A, B)
        gpu_end.record()
        torch.cuda.synchronize()
        print(f"GPU 耗时: {gpu_start.elapsed_time(gpu_end)/1000:.4f} 秒")

if __name__ == "__main__":
    print_environment_info()
    test_pytorch_gpu() 