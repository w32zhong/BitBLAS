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
# (K * bit // storage_nbit) is how many bytes (dtype=int8) the compressed values take
B = te.placeholder((N, K // storage_nbit * bit), name="B", dtype=storage_dtype)

def _tir_packed_to_signed_convert(*args):
    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        max_int_value = (1 << (nbit - 1)) # 2
        return (
                (val >> (pos.astype("uint32") * tir.const(nbit, "uint32")))
                &
                tir.const((1 << nbit) - 1, "uint32")
            ).astype(dtype) - tir.const(max_int_value, dtype)

    # B_decode[v_n, v_k] =
    #   T.Cast("float16", T.bitwise_and(
    #       T.shift_right(
    #           T.Cast("uint32", B[v_n, v_k // 4]),
    #           T.Cast("uint32", v_k % 4) * T.uint32(2)
    #       ),
    #       T.uint32(3)
    #   ))
    #   -
    #   T.float16(2)

    return f_convert

def decode_func(n, k):
    n_float_per_elem = storage_nbit // bit # 8 // 2 = 4
    w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
        bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
    return w

B_decode = te.compute((N, K), decode_func, name="B_decode")

k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(
        A[i, k].astype(accum_dtype) * B_decode[j, k].astype(accum_dtype), axis=k),
    name="C",
)
D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
args = [A, B, D]

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
