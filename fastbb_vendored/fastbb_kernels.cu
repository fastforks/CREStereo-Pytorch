#include <cuda_fp16.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------------------ instance norm NHWC
// x [N, HW, C] (channels-last), C % 8 == 0. Stats in one read: per block, sums of (x - a) and (x - a)^2 around the block's
// first value a (no cancellation) -> (mean, M2); blocks merged in a fixed order with Chan's formula (in_stats_final)
// -> st[N, C, 2] = (mean, rstd). Deterministic, no atomics. Apply: y = (x - mean) * rstd (* w + b) (relu) -> same layout, in place allowed.
#include <cuda_bf16.h>
template <typename T> __device__ __forceinline__ float2 to_f2(uint32_t v);
template <> __device__ __forceinline__ float2 to_f2<__half>(uint32_t v) { return __half22float2(*reinterpret_cast<__half2*>(&v)); }
template <> __device__ __forceinline__ float2 to_f2<__nv_bfloat16>(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v)); }
template <typename T> __device__ __forceinline__ uint32_t from_f2(float a, float b);
template <> __device__ __forceinline__ uint32_t from_f2<__half>(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&h); }
template <> __device__ __forceinline__ uint32_t from_f2<__nv_bfloat16>(float a, float b) { __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *reinterpret_cast<uint32_t*>(&h); }

// merge (nb, mb, qb) into (na, ma, qa): Chan et al. (count, mean, sum of squared deviations)
__device__ __forceinline__ void chan_merge(float& na, float& ma, float& qa, float nb, float mb, float qb) {
    const float n = na + nb;
    if (nb == 0.f) return;
    const float d = mb - ma, f = nb / n;
    ma += d * f;
    qa += qb + d * d * na * f;
    na = n;
}

// grid (blocks over HW, N), 256 threads; thread t: channel group g = t % G (G = C/8), row r0 = t / G, rows step 256/G.
// Sums of (x - a) and (x - a)^2 around a = the block's first row (a value of the same channel, so no cancellation), reduced
// in shared memory in a fixed order -> partial [N, gridDim.x, C, 2] = per block (mean, M2); the block's count is
// min(rpb, HW - blockIdx.x * rpb).
template <typename T>
__device__ void in_stats_impl(const T* __restrict__ x, float* __restrict__ partial, int HW, int C, int rows_per_block) {
    extern __shared__ float sh[];  // [256][16] partial sums
    const int G = C >> 3, t = threadIdx.x, n = blockIdx.y;
    const int R = 256 / G;  // rows in flight per iteration (G <= 256)
    const int g = t % G, r0 = t / G;
    const int beg = blockIdx.x * rows_per_block, end = min(HW, beg + rows_per_block);
    if (beg >= end) return;  // empty tail block (in_stats_final skips it)
    const T* xn = x + (size_t)n * HW * C + g * 8;
    float a[8], s[8], q[8];
    {
        uint4 v = *reinterpret_cast<const uint4*>(xn + (size_t)beg * C);
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) { float2 f = to_f2<T>(w[i]); a[2 * i] = f.x; a[2 * i + 1] = f.y; }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = q[i] = 0.f;
    if (r0 < R) {
        for (int r = beg + r0; r < end; r += R) {
            uint4 v = *reinterpret_cast<const uint4*>(xn + (size_t)r * C);
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = to_f2<T>(w[i]);
                const float dx = f.x - a[2 * i], dy = f.y - a[2 * i + 1];
                s[2 * i] += dx; s[2 * i + 1] += dy;
                q[2 * i] = fmaf(dx, dx, q[2 * i]); q[2 * i + 1] = fmaf(dy, dy, q[2 * i + 1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) { sh[t * 16 + i] = s[i]; sh[t * 16 + 8 + i] = q[i]; }
    __syncthreads();
    if (t < G) {  // thread g sums the R partials of its group
        float sa[16];
#pragma unroll
        for (int i = 0; i < 16; i++) sa[i] = 0.f;
        for (int r = 0; r < R; r++) {
            const float* p = sh + (r * G + t) * 16;
#pragma unroll
            for (int i = 0; i < 16; i++) sa[i] += p[i];
        }
        const float inv = 1.f / (float)(end - beg);
        float* o = partial + (((size_t)n * gridDim.x + blockIdx.x) * C + t * 8) * 2;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float m = sa[i] * inv;
            o[2 * i] = a[i] + m;
            o[2 * i + 1] = fmaxf(sa[8 + i] - sa[i] * m, 0.f);
        }
    }
}

// one warp per (n, c): lane l merges blocks l, l + 32, ... in order, then a fixed shuffle tree -> st[n, c] = (mean, rstd)
// nc32 / nc16 (optional, batch 1): the same statistics in fastbb.normconv's layout, fp32 [2, C] (rstd; mean) and fp16 [2, C]
// (rstd; -mean * rstd)
extern "C" __global__ void __launch_bounds__(256) in_stats_final(const float* __restrict__ partial, float* __restrict__ st, int N,
                                                                 int nb, int HW, int C, int rpb, float eps,
                                                                 float* __restrict__ nc32, __half* __restrict__ nc16) {
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= N * C) return;
    const int n = i / C, c = i % C;
    float na = 0.f, ma = 0.f, qa = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int cnt = min(rpb, HW - b * rpb);
        if (cnt <= 0) break;
        const float* p = partial + (((size_t)n * nb + b) * C + c) * 2;
        chan_merge(na, ma, qa, (float)cnt, p[0], p[1]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float nb2 = __shfl_down_sync(0xffffffffu, na, o), mb = __shfl_down_sync(0xffffffffu, ma, o),
                    qb = __shfl_down_sync(0xffffffffu, qa, o);
        if (lane < o) chan_merge(na, ma, qa, nb2, mb, qb);
    }
    if (lane == 0) {
        const float rstd = rsqrtf(qa / (float)HW + eps);
        st[2 * i] = ma;
        st[2 * i + 1] = rstd;
        if (nc32) {
            nc32[i] = rstd; nc32[C + i] = ma;
            nc16[i] = __float2half(rstd); nc16[C + i] = __float2half(-ma * rstd);
        }
    }
}

// grid-stride over N*HW*C/8 vectors
template <typename T>
__device__ void in_apply_impl(const T* __restrict__ x, T* __restrict__ y, const float* __restrict__ st, const float* __restrict__ w,
                              const float* __restrict__ b, const T* __restrict__ res, int N, int HW, int C, float eps, int relu) {
    const int G = C >> 3;
    const size_t total = (size_t)N * HW * G;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int g = (int)(i % G);
        const int n = (int)(i / ((size_t)HW * G));
        uint4 v = reinterpret_cast<const uint4*>(x)[i];
        uint32_t u[4] = {v.x, v.y, v.z, v.w};
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (relu == 2) rv = reinterpret_cast<const uint4*>(res)[i];
        const uint32_t ru[4] = {rv.x, rv.y, rv.z, rv.w};
        const float* sp = st + ((size_t)n * C + g * 8) * 2;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 f = to_f2<T>(u[k]);
            float2 fr = to_f2<T>(ru[k]);
            float r[2] = {f.x, f.y}, rr[2] = {fr.x, fr.y};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int c = 2 * k + j;
                r[j] = (r[j] - sp[2 * c]) * sp[2 * c + 1];
                if (w) r[j] = r[j] * w[g * 8 + c] + b[g * 8 + c];
                if (relu) r[j] = fmaxf(r[j], 0.f);
                if (relu == 2) r[j] = fmaxf(r[j] + rr[j], 0.f);  // RAFT ResidualBlock tail: relu(res + relu(norm(y)))
            }
            u[k] = from_f2<T>(r[0], r[1]);
        }
        reinterpret_cast<uint4*>(y)[i] = make_uint4(u[0], u[1], u[2], u[3]);
    }
}
extern "C" __global__ void __launch_bounds__(256) in_stats_h(const __half* x, float* partial, int HW, int C, int rpb) { in_stats_impl<__half>(x, partial, HW, C, rpb); }
extern "C" __global__ void __launch_bounds__(256) in_stats_b(const __nv_bfloat16* x, float* partial, int HW, int C, int rpb) { in_stats_impl<__nv_bfloat16>(x, partial, HW, C, rpb); }
extern "C" __global__ void __launch_bounds__(256) in_apply_h(const __half* x, __half* y, const float* st, const float* w, const float* b, const __half* res, int N, int HW, int C, float eps, int relu) { in_apply_impl<__half>(x, y, st, w, b, res, N, HW, C, eps, relu); }
extern "C" __global__ void __launch_bounds__(256) in_apply_b(const __nv_bfloat16* x, __nv_bfloat16* y, const float* st, const float* w, const float* b, const __nv_bfloat16* res, int N, int HW, int C, float eps, int relu) { in_apply_impl<__nv_bfloat16>(x, y, st, w, b, res, N, HW, C, eps, relu); }
