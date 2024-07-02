import sys
sys.path.insert(0, './build/lib')
sys.path.insert(0, './build/lib/bitblas/3rdparty/tvm/python')

import tvm
import numpy as np
from tvm import relax
from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1, 1024), "float16"), B: T.Buffer((1024, 256), "int8"), B_decode: T.Buffer((1024, 1024), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n, k in T.grid(1024, 1024):
            with T.block("B_decode"):
                v_n, v_k = T.axis.remap("SS", [n, k])
                print(v_n, v_k)
                T.reads(B[v_n, v_k // 4])
                T.writes(B_decode[v_n, v_k])
                B_decode[v_n, v_k] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", B[v_n, v_k // 4]), T.Cast("uint32", v_k % 4) * T.uint32(2)), T.uint32(3))) - T.float16(2)

mod = tvm.build(Module, target="c")
with open('/dev/stdout', 'w') as fh:
    fh.write(mod.get_source())
mod = tvm.build(Module, target="llvm")

a = tvm.nd.array(np.zeros((1, 1024)).astype("float16"))
b = tvm.nd.array(np.zeros((1024, 256)).astype("int8"))
c = tvm.nd.array(np.zeros((1024, 1024)).astype("float16"))
mod(a, b, c)
print(c)
