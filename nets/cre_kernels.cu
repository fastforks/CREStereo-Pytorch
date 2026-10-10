// CREStereo inference kernels (size-independent, compiled once to a cubin; loaded by nets/kernels.py).
#include <cuda_fp16.h>
#include <stdint.h>

// Fused AGCL correlation: warp the right features by the flow (bilinear, zero outside, == grid_sample
// align_corners=True) and correlate each of the 4 channel groups (64 ch) with 9 neighbours (3x3 or 1x9).
//   iter mode   (extra == nullptr): corr(p, o) = <L[p], warp(R)[clamp(p + o)]>/64, warp(R)[q] = R(q + flow(q))
//                                  (replicate padding of the warped map, upstream corr_iter)
//   offset mode (extra != nullptr): corr(p, o) = <L[p], R(p + flow(p) + o + extra(p, o))>/64 (corr_att_offset)
// L, R: fp16 NHWC [N, H, W, 256]; flow fp32 NCHW [N, 2, H, W]; extra fp32 NCHW [N, 18, H, W] (o-major, x/y);
// out fp16 NHWC [N, H, W, ocs] with channel g*9 + o, channels 36..ocs-1 set to 0.
// One warp per pixel, lane = 8 channels (group = lane / 8).
extern "C" __global__ void agcl_corr(const __half* __restrict__ L, const __half* __restrict__ R,
                                     const float* __restrict__ flow, const float* __restrict__ extra,
                                     __half* __restrict__ out, int N, int H, int W, int ocs, int small_patch) {
    const int lane = threadIdx.x & 31;
    const long long nwarps = (long long)gridDim.x * (blockDim.x >> 5);
    const long long P = (long long)N * H * W;
    const long long HW = (long long)H * W;
    for (long long p = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); p < P; p += nwarps) {
        const int n = (int)(p / HW);
        const int rem = (int)(p - (long long)n * HW);
        const int y = rem / W, x = rem - y * W;
        float lf[8];
        {
            uint4 v = *reinterpret_cast<const uint4*>(L + p * 256 + lane * 8);
            const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; k++) { float2 f = __half22float2(h2[k]); lf[2 * k] = f.x; lf[2 * k + 1] = f.y; }
        }
        const float* fx = flow + (long long)n * 2 * HW;
        const float* fy = fx + HW;
        const __half* Rn = R + (long long)n * HW * 256 + lane * 8;
        float res = 0.f;  // lane (g*8 + o) of group 0..3 keeps result o of group g (o < 9 needs 2 rounds)
        float keep[9];
#pragma unroll
        for (int o = 0; o < 9; o++) {
            int dy, dx;
            if (small_patch) { dy = o / 3 - 1; dx = o % 3 - 1; } else { dy = 0; dx = o - 4; }
            float sx, sy;
            if (extra == nullptr) {
                int qx = min(max(x + dx, 0), W - 1), qy = min(max(y + dy, 0), H - 1);
                long long q = (long long)qy * W + qx;
                sx = qx + fx[q];
                sy = qy + fy[q];
            } else {
                const float* e = extra + (long long)n * 18 * HW + rem;
                sx = x + fx[rem] + dx + e[(2 * o) * HW];
                sy = y + fy[rem] + dy + e[(2 * o + 1) * HW];
            }
            float x0f = floorf(sx), y0f = floorf(sy);
            int x0 = (int)x0f, y0 = (int)y0f;
            float ax = sx - x0f, ay = sy - y0f;
            float acc[8];
#pragma unroll
            for (int k = 0; k < 8; k++) acc[k] = 0.f;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                int xx = x0 + (t & 1), yy = y0 + (t >> 1);
                float wgt = ((t & 1) ? ax : 1.f - ax) * ((t >> 1) ? ay : 1.f - ay);
                if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
                uint4 v = *reinterpret_cast<const uint4*>(Rn + ((long long)yy * W + xx) * 256);
                const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    float2 f = __half22float2(h2[k]);
                    acc[2 * k] += wgt * f.x;
                    acc[2 * k + 1] += wgt * f.y;
                }
            }
            float d = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) d += lf[k] * acc[k];
            d += __shfl_xor_sync(0xffffffff, d, 4);
            d += __shfl_xor_sync(0xffffffff, d, 2);
            d += __shfl_xor_sync(0xffffffff, d, 1);
            keep[o] = d * (1.f / 64.f);
        }
        // lane l in group g holds all 9 results of group g; write them out: lane (g*8 + j) writes o = j, and j = 0 also o = 8
        __half* op = out + p * ocs;
        const int g = lane >> 3, j = lane & 7;
#pragma unroll
        for (int o = 0; o < 8; o++)
            if (o == j) op[g * 9 + o] = __float2half(keep[o]);
        if (j == 0) op[g * 9 + 8] = __float2half(keep[8]);
        for (int c = 36 + lane; c < ocs; c += 32) op[c] = __float2half(0.f);
        (void)res;
    }
}

__device__ __forceinline__ void ld8(const __half* p, float* f) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; k++) { float2 t = __half22float2(h2[k]); f[2 * k] = t.x; f[2 * k + 1] = t.y; }
}
__device__ __forceinline__ void st8(__half* p, const float* f) {
    uint4 v;
    __half2* h2 = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; k++) h2[k] = __floats2half2_rn(f[2 * k], f[2 * k + 1]);
    *reinterpret_cast<uint4*>(p) = v;
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// y[p, c] = act(y[p, c] + b[c]) in place, y dense [P, C] fp16 (C % 8 == 0), b fp16 [C]; act 0 none, 1 relu
extern "C" __global__ void bias_act(__half* __restrict__ y, const __half* __restrict__ b, long long P, int C, int act) {
    const int C8 = C >> 3;
    long long n = P * C8;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)(i % C8) * 8;
        float v[8], bb[8];
        ld8(y + i * 8, v);
        ld8(b + c, bb);
#pragma unroll
        for (int k = 0; k < 8; k++) { v[k] += bb[k]; if (act == 1) v[k] = fmaxf(v[k], 0.f); }
        st8(y + i * 8, v);
    }
}

// ConvGRU gates, step 1: zr = sigmoid(zr_conv + czr) (z kept in zr[:, :Hd]); writes r*h into hx[:, :Hd] (row stride hxs).
// All tensors dense channels-last fp16, Hd % 8 == 0.
extern "C" __global__ void gru_zr(__half* __restrict__ zr, const __half* __restrict__ czr, const __half* __restrict__ h,
                                  __half* __restrict__ hx, int hxs, long long P, int Hd) {
    const int H8 = Hd >> 3;
    long long n = P * H8;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        long long p = i / H8;
        int c = (int)(i - p * H8) * 8;
        long long iz = p * 2 * Hd + c, ir = iz + Hd;
        float z[8], cz[8], r[8], cr[8], hh[8];
        ld8(zr + iz, z); ld8(czr + iz, cz); ld8(zr + ir, r); ld8(czr + ir, cr); ld8(h + p * Hd + c, hh);
#pragma unroll
        for (int k = 0; k < 8; k++) { z[k] = sigm(z[k] + cz[k]); r[k] = sigm(r[k] + cr[k]) * hh[k]; }
        st8(zr + iz, z);
        st8(hx + p * hxs + c, r);
    }
}

// step 2: h = (1-z) h + z tanh(q_conv + cq), h dense [P, Hd]; also refreshes hx[:, :Hd] = h (for the next GRU)
extern "C" __global__ void gru_update(const __half* __restrict__ q, const __half* __restrict__ cq, const __half* __restrict__ zr,
                                      __half* __restrict__ h, __half* __restrict__ hx, int hxs, long long P, int Hd) {
    const int H8 = Hd >> 3;
    long long n = P * H8;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        long long p = i / H8;
        int c = (int)(i - p * H8) * 8;
        float qq[8], cc[8], z[8], hh[8];
        ld8(q + p * Hd + c, qq); ld8(cq + p * Hd + c, cc); ld8(zr + p * 2 * Hd + c, z); ld8(h + p * Hd + c, hh);
#pragma unroll
        for (int k = 0; k < 8; k++) hh[k] = (1.f - z[k]) * hh[k] + z[k] * tanhf(qq[k] + cc[k]);
        st8(h + p * Hd + c, hh);
        st8(hx + p * hxs + c, hh);
    }
}
