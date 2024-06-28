# see integration/BitNet/utils_quant.py MatmulConfig
import sys
sys.path.insert(0, './build/lib')
import bitblas
import torch

# enabling debug output

matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=1024,  # N dimension
    K=1024,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="int2",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)

matmul = bitblas.Matmul(config=matmul_config)

# Create input matrices
input_tensor = torch.rand((1, 1024), dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, 7, (1024, 1024), dtype=torch.int8).cuda()

weight_tensor_quant = matmul.transform_weight(weight_tensor)

# Perform mixed-precision matrix multiplication
output_tensor = matmul(input_tensor, weight_tensor_quant)

# Reference result using PyTorch matmul for comparison
ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))

print("Ref output:", ref_result)
print("BitBLAS output:", output_tensor)
