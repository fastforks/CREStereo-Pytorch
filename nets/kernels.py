"""Custom CUDA kernels of nets/cre_kernels.cu for PyTorch (loader adapted from RAFT-Stereo exp 002 cuda_kernels.py):
compiled once to a cubin (shipped in core/prebuilt/, else nvcc of CUDA_HOME, cached in ~/.cache/crestereo-torch),
loaded through the driver API, launched on torch's current stream. Size-independent: no per-shape compilation.
"""

import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cre_kernels.cu")
_PREBUILT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "crestereo-torch")

# p: pointer (tensor or int address), i: int32, l: int64, f: float32
SIGNATURES = {
    "agcl_corr": "pppppiiiii",
    "bias_act": "pplii",
    "gru_zr": "ppppili",
    "gru_update": "pppppili",
}
_CTYPE = {"i": ctypes.c_int32, "l": ctypes.c_int64, "f": ctypes.c_float, "u": ctypes.c_uint32}


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src + f"sm_{major}{minor}".encode()).hexdigest()[:16]
        cubin = os.path.join(_PREBUILT, f"cre_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            cubin = os.path.join(_CACHE, f"cre_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            os.makedirs(_CACHE, exist_ok=True)
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            nvcc = os.path.join(cuda_home, "bin", "nvcc")
            if not os.path.isfile(nvcc):
                raise RuntimeError(
                    f"No prebuilt CREStereo kernel for sm_{major}{minor}; set CUDA_HOME to a toolkit with nvcc, or use the portable model"
                )
            subprocess.run(
                [
                    nvcc,
                    "-cubin",
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    f"-arch=sm_{major}{minor}",
                    f"-I{cuda_home}/include",
                    "-o",
                    cubin + ".tmp",
                    _SRC,
                ],
                check=True,
            )
            os.replace(cubin + ".tmp", cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(cubin, "rb").read()
        _check(
            self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData"
        )
        self.funcs = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(
                self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()),
                f"cuModuleGetFunction {name}",
            )
            self.funcs[name] = f

    @staticmethod
    def linear(n):
        """Rust's Launch::linear: 256 threads, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args):
        grid, block = cfg
        sig = SIGNATURES[name]
        assert len(sig) == len(args), f"{name}: expected {len(sig)} args, got {len(args)}"
        holders = []
        for t, a in zip(sig, args):
            if t == "p":
                holders.append(
                    ctypes.c_void_p(
                        a.data_ptr() if isinstance(a, torch.Tensor) else (int(a) or None)
                    )
                )
            else:
                holders.append(_CTYPE[t](a))
        arr = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
        )
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, 0, stream, arr, None),
            f"launch {name}",
        )
