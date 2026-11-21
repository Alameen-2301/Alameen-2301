// Kernels written to make the compiler emit as many different SASS
// instructions as possible. They are never run; they only feed the corpus.
#include <cuda_fp16.h>

extern "C" __global__ void k_float(float* o, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i];
    float r = x * y + x;
    r += __fdividef(x, y) + sqrtf(fabsf(y)) + rsqrtf(x);
    r += __expf(x) + __logf(y) + __sinf(x) + __cosf(y) + exp2f(x);
    r = fminf(r, y) + fmaxf(r, x);
    r += floorf(x) + ceilf(y) + truncf(x) + rintf(y);
    r += x / y;  // full-precision divide -> slow path call
    o[i] = r > 0.f ? r : -r;
}

extern "C" __global__ void k_double(double* o, const double* a, const double* b) {
    int i = threadIdx.x;
    double x = a[i], y = b[i];
    o[i] = fma(x, y, x) + x * y - y + sqrt(x) + (x / y) + (double)(float)x;
}

extern "C" __global__ void k_int(int* o, const int* a, const unsigned* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[i]; unsigned y = b[i];
    int r = x * n + (x >> 3) + (y >> (x & 31)) + (x << 5);
    r ^= (x & y) | (~y);
    r += __popc(y) + __clz(x) + __ffs(x) + __brev(y);
    r += __byte_perm(x, y, 0x3210) + __funnelshift_l(x, y, 7);
    r += min(x, n) + max(x, (int)y) + abs(x);
    r += x / n + x % n + (int)(y / 7u);
    r += __mulhi(x, n) + __umulhi(y, 3u);
    long long w = (long long)x * y + ((long long)n << 40);
    o[i] = r + (int)(w >> 33) + (r > n ? 1 : 2);
}

extern "C" __global__ void k_conv(int* oi, float* of, double* od, const float* a,
                                  const int* b, const double* c) {
    int i = threadIdx.x;
    float f = a[i]; int k = b[i]; double d = c[i];
    oi[i] = (int)f + __float2int_rn(f) + (int)(unsigned)f + (int)d;
    of[i] = (float)k + (float)(unsigned)k + (float)d + __int_as_float(k);
    od[i] = (double)k + (double)f + (double)(long long)k;
}

extern "C" __global__ void k_half(__half* o, const __half* a, const __half2* b) {
    int i = threadIdx.x;
    __half2 v = b[i];
    v = __hfma2(v, v, v);
    v = __hmul2(v, __hadd2(v, v));
    o[i] = __hadd(a[i], __low2half(v)) * __high2half(v);
}

extern "C" __global__ void k_mem(float* o, const float* __restrict__ a, int n) {
    __shared__ float s[256];
    __shared__ double sd[64];
    int t = threadIdx.x;
    s[t] = a[blockIdx.x * 256 + t];
    if (t < 64) sd[t] = s[t];
    __syncthreads();
    float acc = 0.f;
    #pragma unroll 4
    for (int k = 0; k < n; ++k) acc += s[(t + k) & 255] * __ldg(a + k);
    float4 v = reinterpret_cast<const float4*>(a)[t];
    acc += v.x + v.y + v.z + v.w + (float)sd[t & 63];
    o[t] = acc;
    reinterpret_cast<float2*>(o)[t + 512] = make_float2(acc, acc);
}

extern "C" __global__ void k_atomic(int* c, float* f, unsigned long long* u, int* lock) {
    int i = threadIdx.x;
    atomicAdd(c, 1);
    atomicAdd(f + i, 1.5f);
    atomicMax(c + 1, i);
    atomicMin(c + 2, i);
    atomicOr(c + 3, i);
    atomicXor(c + 4, i);
    atomicExch(c + 5, i);
    atomicCAS(lock, 0, 1);
    atomicAdd(u, 1ull);
    int old = atomicAdd(c + 6, i);
    c[7 + i] = old;
    __shared__ int sh[32];
    atomicAdd(&sh[i & 31], 1);
    __syncthreads();
    c[100 + i] = sh[i & 31];
    __threadfence();
}

extern "C" __global__ void k_warp(int* o, const float* a) {
    int i = threadIdx.x;
    float v = a[i];
    v += __shfl_sync(0xffffffff, v, 0);
    v += __shfl_down_sync(0xffffffff, v, 1);
    v += __shfl_up_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    unsigned b = __ballot_sync(0xffffffff, v > 0.f);
    int all = __all_sync(0xffffffff, v > 1.f);
    int any = __any_sync(0xffffffff, v > 2.f);
    unsigned m = __match_any_sync(0xffffffff, (int)v);
    int red = __reduce_add_sync(0xffffffff, (int)v);
    __syncwarp();
    o[i] = (int)v + b + all + any + m + red + __activemask() + (int)clock();
    unsigned lm; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lm)); o[i + 64] = lm;
}

extern "C" __global__ void k_control(int* o, const int* a, int n) {
    int i = threadIdx.x;
    int r = 0;
    for (int k = 0; k < n; ++k) {
        int x = a[k];
        if (x & 1) r += x; else r -= x;
        if (x == 42) break;
        switch (x & 3) { case 0: r *= 3; break; case 1: r ^= 7; break; default: r += 1; }
    }
    if (i == 0) asm volatile("trap;");
    o[i] = r + blockDim.y + gridDim.z + blockIdx.y;
}

__device__ __noinline__ int helper(int x, int y) { return x * y + (x ^ y); }

extern "C" __global__ void k_call(int* o, const int* a) {
    int i = threadIdx.x;
    int local[16];
    for (int k = 0; k < 16; ++k) local[k] = a[k * i];
    o[i] = helper(a[i], i) + local[a[i] & 15];
}

extern "C" __global__ void k_mma(float* d, const __half* a, const __half* b) {
    // tensor core: one m16n8k16 HMMA via inline PTX
    unsigned ra[4], rb[2];
    float c[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) ra[k] = reinterpret_cast<const unsigned*>(a)[threadIdx.x * 4 + k];
    for (int k = 0; k < 2; ++k) rb[k] = reinterpret_cast<const unsigned*>(b)[threadIdx.x * 2 + k];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
    for (int k = 0; k < 4; ++k) d[threadIdx.x * 4 + k] = c[k];
}

extern "C" __global__ void k_async(float* o, const float* a) {
    __shared__ float s[256];
    unsigned sa = (unsigned)__cvta_generic_to_shared(&s[threadIdx.x]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sa), "l"(a + threadIdx.x));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(sa));
    o[threadIdx.x] = s[threadIdx.x ^ 1] + __uint_as_float(x);
}

extern "C" __global__ void k_tex(float* o, cudaTextureObject_t t, cudaSurfaceObject_t s) {
    int i = threadIdx.x;
    float v = tex1Dfetch<float>(t, i) + tex2D<float>(t, i * 0.5f, 0.25f);
    surf2Dwrite(v, s, i * 4, 0);
    o[i] = v;
}
