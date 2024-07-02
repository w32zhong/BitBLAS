import sys
sys.path.insert(0, './build/lib')
sys.path.insert(0, './build/lib/bitblas/3rdparty/tvm/python')

import tvm
from tvm import te
from tvm import tir

K=N=1024
group_size=1024
M=1
in_dtype='float16'
storage_dtype='int8'
storage_type='int'
storage_nbit=8
bit=2
accum_dtype="float16"
out_dtype="float16"
fast_decoding=True
source_format='int'
with_scaling=False
with_zeros=False
zeros_mode='original'

A = te.placeholder((M, K), name="A", dtype=in_dtype)
B = te.placeholder((N, K // storage_nbit * bit), name="B", dtype=storage_dtype)
LUT = te.placeholder((1 << bit,), name="LUT", dtype=in_dtype)
Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
Zeros = te.placeholder((N, K // group_size), name="Zeros", dtype=in_dtype)
QZeros = te.placeholder(((K // group_size), N // storage_nbit * bit),
                        name="QZeros",
                        dtype=storage_dtype)
Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

n_float_per_elem = storage_nbit // bit

def _tir_packed_to_signed_convert(*kwargs):
    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        max_int_value = (1 << (nbit - 1))
        return ((val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & tir.const(
            (1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)

    return f_convert

def decode_func(n, k):
    w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
        bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
    return w

B_decode = te.compute((N, K), decode_func, name="B_decode")
args = [A, B, B_decode]

#k = te.reduce_axis((0, K), name="k")
#C = te.compute(
#    (M, N),
#    lambda i, j: te.sum(
#        A[i, k].astype(accum_dtype) * B_decode[j, k].astype(accum_dtype), axis=k),
#    name="C",
#)
#D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
#args = [A, B, D]

func = te.create_prim_func(args)
"""
.with_attr(
    "dequantize_info",
    {
        "B_decode": {
            "decode_block": "B_decode",
            "fast_decoding": fast_decoding,
            "source_format": {
                "bits": bit,
                "format": source_format,
            },
            "storage_dtype": storage_dtype,
            "target_format": in_dtype,
            "with_scaling": with_scaling,
            "with_zeros": with_zeros,
            "zeros_mode": zeros_mode,
            "group_size": group_size,
        }
    },
)
"""
mod = tvm.IRModule.from_expr(func)
print(mod)
