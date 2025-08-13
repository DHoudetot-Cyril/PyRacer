import torch

def test_rocm():
    print("PyTorch version:", torch.__version__)
    print("Is ROCm available:", torch.version.hip is not None and torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
        x = torch.randn(3, 3).to('cuda')
        print("Tensor on ROCm GPU:", x)
    else:
        print("ROCm GPU non détecté")

if __name__ == "__main__":
    test_rocm()
