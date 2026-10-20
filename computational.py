import torch
# from net.model import Restormer,Restormer_orgin
from net.model_zd_zc import Restormer
from net.PromptIR import PromptIR
import time
from fvcore.nn import FlopCountAnalysis

def compute_gflops(model, input_tensor):
    # Compute the number of GFLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    gflops = flops.total() / 1e9  # Convert to GFLOPs
    return gflops

def compute_inference_time(model, input_tensor, num_runs=100):
    # Compute the average inference time over multiple runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()
    avg_inference_time_seconds = (end_time - start_time) / num_runs
    avg_inference_time_ms = avg_inference_time_seconds * 1000
    return avg_inference_time_ms

def compute_num_params(model):
    # Compute the total number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

if __name__ == '__main__':
    # Define input tensor with shape (3, 128, 128)
    input_tensor = torch.randn(1, 3, 128, 128).cuda()  # Batch size of 1, with 3 channels (RGB), 128x128 image
    #
    # Initialize the Restormer model
    model = Restormer().cuda()
    # model = Restormer_orgin().cuda()
    # model = PromptIR(decoder=True).cuda()
    # Compute GFLOPs
    gflops = compute_gflops(model, input_tensor)
    print(f"GFLOPs: {gflops:.2f} GFLOPs")

    # Compute Inference Time
    avg_inference_time = compute_inference_time(model, input_tensor)
    print(f"Average Inference Time: {avg_inference_time:.6f} ms")

    # Compute Number of Parameters
    num_params = compute_num_params(model)
    print(f"Number of Parameters: {num_params / 1e6:.2f} million")
