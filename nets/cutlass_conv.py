"""Origin: RAFT-Stereo exp 002 attempt/general-fused @ ea62638 (core/cutlass_conv.py); exp 003: non-square filters
(fused_conv_rs), EpiArgs.h2s, CUTLASS headers from CUTLASS_DIR.

Fused-epilogue convolutions from core/cutlass_conv.cu (CUTLASS implicit GEMM, fp16 accumulation), compiled once
ahead of time into a shared library (core/prebuilt/libcutlass_conv_<hash>.so, sm_80/86/89/90/120 SASS + PTX); the
problem size is a runtime argument, so nothing is compiled or tuned per input size. Without the shipped library it
is built once with the active CUDA_HOME's nvcc into ~/.cache (needs third_party/cutlass, pinned tag in the README).

    fc = FusedConvs.get()
    fc.gru_zr(cfg, A, w, h, w_, C, czr, zr, B)     # see GeneralRAFT.gru
"""

import ctypes
import hashlib
import os
import subprocess

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "cutlass_conv.cu")
_PREBUILT = os.path.join(_HERE, "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "stereo-cutlass-conv")


def _cutlass_dir():
    directory = os.environ.get("CUTLASS_DIR", "")
    if not os.path.isfile(os.path.join(directory, "include", "cutlass", "cutlass.h")):
        raise FileNotFoundError(
            "CUTLASS headers not found: set CUTLASS_DIR to a CUTLASS v4.7.1 checkout"
        )
    return directory


class EpiArgs(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("act", ctypes.c_int),
        ("c", ctypes.c_void_p),
        ("cs", ctypes.c_int),
        ("z", ctypes.c_void_p),
        ("zs", ctypes.c_int),
        ("h", ctypes.c_void_p),
        ("hs", ctypes.c_int),
        ("rh", ctypes.c_void_p),
        ("h2", ctypes.c_void_p),
        ("s", ctypes.c_void_p),
        ("t", ctypes.c_void_p),
        ("coords", ctypes.c_void_p),
        ("W", ctypes.c_int),
        ("h2s", ctypes.c_int),
    ]


def _tag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build(out, archs=(80, 86, 89, 90, 120)):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + [
        "-gencode=arch=compute_80,code=compute_80"
    ]
    # nvcc.profile embeds its absolute library-search paths in the fatbin. Supply the toolkit explicitly.
    build_env = dict(
        os.environ,
        PATH=os.path.join(cuda_home, "nvvm", "bin") + os.pathsep + os.environ.get("PATH", ""),
    )
    subprocess.run(
        [
            os.path.join(cuda_home, "bin", "nvcc"),
            "--dont-use-profile",
            "-ldir",
            os.path.join(cuda_home, "nvvm", "libdevice"),
            f"-I{cuda_home}/include",
            "-isystem",
            os.path.join(cuda_home, "include", "cccl"),
            "-cudadevrt",
            "none",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            f"-ffile-prefix-map={_cutlass_dir()}=cutlass",
            "-Xcompiler",
            f"-ffile-prefix-map={_HERE}=.",
            "-cudart",
            "shared",
            "-Xlinker",
            f"-L{cuda_home}/lib",
            *gencode,
            f"-I{_cutlass_dir()}/include",
            "-o",
            out + ".tmp",
            _SRC,
        ],
        check=True,
        env=build_env,
    )
    os.replace(out + ".tmp", out)
    return out


def build_prebuilt():
    """Build the shipped library for the current source (run after editing cutlass_conv.cu)."""
    os.makedirs(_PREBUILT, exist_ok=True)
    out = build(os.path.join(_PREBUILT, f"libcutlass_conv_{_tag()}.so"))
    for f in os.listdir(_PREBUILT):
        if f.startswith("libcutlass_conv_") and os.path.join(_PREBUILT, f) != out:
            os.remove(os.path.join(_PREBUILT, f))
    return out


class FusedConvs:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()  # torch has loaded libcudart.so.13, which the library links against
        tag = _tag()
        lib = os.path.join(_PREBUILT, f"libcutlass_conv_{tag}.so")
        if not os.path.exists(lib):
            major, minor = torch.cuda.get_device_capability()
            lib = os.path.join(_CACHE, f"libcutlass_conv_{tag}_sm{major}{minor}.so")
            if not os.path.exists(lib):
                os.makedirs(_CACHE, exist_ok=True)
                build(lib, archs=(major * 10 + minor,))
        self.lib = ctypes.CDLL(lib)
        self.lib.fused_conv_rs.argtypes = (
            [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            + [ctypes.c_int] * 9
            + [ctypes.POINTER(EpiArgs), ctypes.c_void_p]
        )

    def run(self, cfg, x, w, H, W, C, K, e: EpiArgs, padding=None, stride=1):
        """One launch; returns the status (0 ok, 1 geometry not supported, else error)."""
        R, S = (
            (w.shape[1], w.shape[2])
            if w.dim() == 4 and w.shape[3] == C
            else (w.shape[2], w.shape[3])
        )
        ph, pw = (R // 2, S // 2) if padding is None else padding
        return self.lib.fused_conv_rs(
            cfg,
            x.data_ptr(),
            w.data_ptr(),
            H,
            W,
            C,
            K,
            R,
            S,
            ph,
            pw,
            stride,
            ctypes.byref(e),
            torch.cuda.current_stream().cuda_stream,
        )


if __name__ == "__main__":
    print(build_prebuilt())
