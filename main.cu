#include <iostream>
#include "opencv2/opencv.hpp"


#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <chrono>

inline std::shared_ptr<void> MyCudaMalloc(int size)
{
    void* tmp = nullptr;
    if( cudaError_t::cudaSuccess != cudaMalloc(&tmp, size))
        throw std::bad_alloc();
    return {tmp, [](void* p){ cudaFree(p); }};
}

#define TILE_K 16
#define TILE_M 128
#define TILE_N 128
#define TILE_M_4 32
#define TILE_N_4 32

// A B C are all row-major
__global__ void MatrixMul_T_T(int cTileK, int m, int n, int k, float alpha,  const float* A, int lda, const float* B, int ldb, float beta,  float* C, int ldc)
{
    float4 f4_zero = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 result[4][4] = {{f4_zero } };
    float4 reg_a[2];
    float4 reg_b[2];

    __shared__ float4 smemA[2][TILE_K][TILE_M_4];
    __shared__ float4 smemB[2][TILE_K][TILE_N_4];

    int tx_4, ty_4, tx_32, ty_32;
    {
        int ltid = threadIdx.y * blockDim.x + threadIdx.x;
        tx_4 = ltid % 4;
        ty_4 = ltid / 4;
        tx_32 = ltid % 32;
        ty_32 = ltid / 32;
    }

    int tileIdx=0;
    float4 ldA[2], ldB[2];
    {
        const float* pTileA = A + blockIdx.y * TILE_M * lda + tileIdx * TILE_K;
        const float* pTileB = B + tileIdx * TILE_K * ldb + blockIdx.x * TILE_N;
        ldA[0] = *(float4*)( pTileA + ty_4 * lda + tx_4 * 4 );
        ldA[1] = *(float4*)( pTileA + (ty_4 + 64) * lda + tx_4 * 4 );
        ldB[0] = *(float4*)( pTileB + ty_32 * ldb + tx_32 * 4 );
        ldB[1] = *(float4*)( pTileB + (ty_32 + 8) * ldb + tx_32 * 4 );

        *((float*)&smemA[0][0][0] + (tx_4*4    ) * TILE_M + ty_4 ) = ldA[0].x;
        *((float*)&smemA[0][0][0] + (tx_4*4 + 1) * TILE_M + ty_4 ) = ldA[0].y;
        *((float*)&smemA[0][0][0] + (tx_4*4 + 2) * TILE_M + ty_4 ) = ldA[0].z;
        *((float*)&smemA[0][0][0] + (tx_4*4 + 3) * TILE_M + ty_4 ) = ldA[0].w;

        *((float*)&smemA[0][0][0] + (tx_4*4    ) * TILE_M + ty_4 + 64) = ldA[1].x;
        *((float*)&smemA[0][0][0] + (tx_4*4 + 1) * TILE_M + ty_4 + 64) = ldA[1].y;
        *((float*)&smemA[0][0][0] + (tx_4*4 + 2) * TILE_M + ty_4 + 64) = ldA[1].z;
        *((float*)&smemA[0][0][0] + (tx_4*4 + 3) * TILE_M + ty_4 + 64) = ldA[1].w;

        smemB[0][ty_32  ][tx_32] = ldB[0];
        smemB[0][ty_32+8][tx_32] = ldB[1];
    }

    __syncthreads();
    int write_stage_idx = 1;
    do
    {
        int load_stage_idx = write_stage_idx ^ 1;
        ++tileIdx;
        if( tileIdx < cTileK )
        {
            const float* pTileA = A + blockIdx.y * TILE_M * lda + tileIdx * TILE_K;
            const float* pTileB = B + tileIdx * TILE_K * ldb + blockIdx.x * TILE_N;
            ldA[0] = *(float4*)( pTileA + ty_4 * lda + tx_4 * 4 );
            ldA[1] = *(float4*)( pTileA + (ty_4 + 64) * lda + tx_4 * 4 );
            ldB[0] = *(float4*)( pTileB + ty_32 * ldb + tx_32 * 4 );
            ldB[1] = *(float4*)( pTileB + (ty_32 + 8) * ldb + tx_32 * 4 );
        }

#pragma unroll
        for(int subTileIdx=0; subTileIdx < TILE_K; ++subTileIdx)
        {
            reg_a[0] = smemA[load_stage_idx][subTileIdx][threadIdx.y];
            reg_a[1] = smemA[load_stage_idx][subTileIdx][threadIdx.y+16];

            reg_b[0] = smemB[load_stage_idx][subTileIdx][threadIdx.x];
            reg_b[1] = smemB[load_stage_idx][subTileIdx][threadIdx.x+16];

            result[0][0].x += reg_a[0].x * reg_b[0].x;
            result[0][0].y += reg_a[0].x * reg_b[0].y;
            result[0][0].z += reg_a[0].x * reg_b[0].z;
            result[0][0].w += reg_a[0].x * reg_b[0].w;
            result[0][1].x += reg_a[0].y * reg_b[0].x;
            result[0][1].y += reg_a[0].y * reg_b[0].y;
            result[0][1].z += reg_a[0].y * reg_b[0].z;
            result[0][1].w += reg_a[0].y * reg_b[0].w;
            result[0][2].x += reg_a[0].z * reg_b[0].x;
            result[0][2].y += reg_a[0].z * reg_b[0].y;
            result[0][2].z += reg_a[0].z * reg_b[0].z;
            result[0][2].w += reg_a[0].z * reg_b[0].w;
            result[0][3].x += reg_a[0].w * reg_b[0].x;
            result[0][3].y += reg_a[0].w * reg_b[0].y;
            result[0][3].z += reg_a[0].w * reg_b[0].z;
            result[0][3].w += reg_a[0].w * reg_b[0].w;

            // ----------------------

            result[1][0].x += reg_a[0].x * reg_b[1].x;
            result[1][0].y += reg_a[0].x * reg_b[1].y;
            result[1][0].z += reg_a[0].x * reg_b[1].z;
            result[1][0].w += reg_a[0].x * reg_b[1].w;
            result[1][1].x += reg_a[0].y * reg_b[1].x;
            result[1][1].y += reg_a[0].y * reg_b[1].y;
            result[1][1].z += reg_a[0].y * reg_b[1].z;
            result[1][1].w += reg_a[0].y * reg_b[1].w;
            result[1][2].x += reg_a[0].z * reg_b[1].x;
            result[1][2].y += reg_a[0].z * reg_b[1].y;
            result[1][2].z += reg_a[0].z * reg_b[1].z;
            result[1][2].w += reg_a[0].z * reg_b[1].w;
            result[1][3].x += reg_a[0].w * reg_b[1].x;
            result[1][3].y += reg_a[0].w * reg_b[1].y;
            result[1][3].z += reg_a[0].w * reg_b[1].z;
            result[1][3].w += reg_a[0].w * reg_b[1].w;

            // ----------------------

            result[2][0].x += reg_a[1].x * reg_b[0].x;
            result[2][0].y += reg_a[1].x * reg_b[0].y;
            result[2][0].z += reg_a[1].x * reg_b[0].z;
            result[2][0].w += reg_a[1].x * reg_b[0].w;
            result[2][1].x += reg_a[1].y * reg_b[0].x;
            result[2][1].y += reg_a[1].y * reg_b[0].y;
            result[2][1].z += reg_a[1].y * reg_b[0].z;
            result[2][1].w += reg_a[1].y * reg_b[0].w;
            result[2][2].x += reg_a[1].z * reg_b[0].x;
            result[2][2].y += reg_a[1].z * reg_b[0].y;
            result[2][2].z += reg_a[1].z * reg_b[0].z;
            result[2][2].w += reg_a[1].z * reg_b[0].w;
            result[2][3].x += reg_a[1].w * reg_b[0].x;
            result[2][3].y += reg_a[1].w * reg_b[0].y;
            result[2][3].z += reg_a[1].w * reg_b[0].z;
            result[2][3].w += reg_a[1].w * reg_b[0].w;

            // ----------------------

            result[3][0].x += reg_a[1].x * reg_b[1].x;
            result[3][0].y += reg_a[1].x * reg_b[1].y;
            result[3][0].z += reg_a[1].x * reg_b[1].z;
            result[3][0].w += reg_a[1].x * reg_b[1].w;
            result[3][1].x += reg_a[1].y * reg_b[1].x;
            result[3][1].y += reg_a[1].y * reg_b[1].y;
            result[3][1].z += reg_a[1].y * reg_b[1].z;
            result[3][1].w += reg_a[1].y * reg_b[1].w;
            result[3][2].x += reg_a[1].z * reg_b[1].x;
            result[3][2].y += reg_a[1].z * reg_b[1].y;
            result[3][2].z += reg_a[1].z * reg_b[1].z;
            result[3][2].w += reg_a[1].z * reg_b[1].w;
            result[3][3].x += reg_a[1].w * reg_b[1].x;
            result[3][3].y += reg_a[1].w * reg_b[1].y;
            result[3][3].z += reg_a[1].w * reg_b[1].z;
            result[3][3].w += reg_a[1].w * reg_b[1].w;
        }

        if( tileIdx < cTileK )
        {
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4    ) * TILE_M + ty_4 ) = ldA[0].x;
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4 + 1) * TILE_M + ty_4 ) = ldA[0].y;
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4 + 2) * TILE_M + ty_4 ) = ldA[0].z;
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4 + 3) * TILE_M + ty_4 ) = ldA[0].w;

            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4    ) * TILE_M + ty_4 + 64) = ldA[1].x;
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4 + 1) * TILE_M + ty_4 + 64) = ldA[1].y;
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4 + 2) * TILE_M + ty_4 + 64) = ldA[1].z;
            *((float*)&smemA[write_stage_idx][0][0] + (tx_4*4 + 3) * TILE_M + ty_4 + 64) = ldA[1].w;

            smemB[write_stage_idx][ty_32  ][tx_32] = ldB[0];
            smemB[write_stage_idx][ty_32+8][tx_32] = ldB[1];
            write_stage_idx ^= 1;
            __syncthreads();
        }

    }while(tileIdx < cTileK);

#pragma unroll
    for(int i=0; i<4; ++i)
    {
#pragma unroll
        for(int j=0; j<4; ++j)
        {
            result[i][j].x *= alpha;
            result[i][j].y *= alpha;
            result[i][j].z *= alpha;
            result[i][j].w *= alpha;
        }
    }
    float *startPointer[4] = {
            C + (blockIdx.y * TILE_M + threadIdx.y * 4     ) * ldc + blockIdx.x * TILE_N + threadIdx.x * 4,
            C + (blockIdx.y * TILE_M + threadIdx.y * 4     ) * ldc + blockIdx.x * TILE_N + threadIdx.x * 4 + 64,
            C + (blockIdx.y * TILE_M + threadIdx.y * 4 + 64) * ldc + blockIdx.x * TILE_N + threadIdx.x * 4,
            C + (blockIdx.y * TILE_M + threadIdx.y * 4 + 64) * ldc + blockIdx.x * TILE_N + threadIdx.x * 4 + 64
    };
#pragma unroll
    for( int i=0; i<4; ++i)
    {
#pragma unroll
        for( int j=0; j<4; ++j)
        {
            if( beta == 0 ){
                *(float4*)( startPointer[i] + j * ldc ) = result[i][j];
            }else{
                float4 value = *(float4*)( startPointer[i] + j * ldc );
                value.x = beta * value.x + result[i][j].x;
                value.y = beta * value.y + result[i][j].y;
                value.z = beta * value.z + result[i][j].z;
                value.w = beta * value.w + result[i][j].w;
                *(float4*)( startPointer[i] + j * ldc ) = value;
            }
        }
    }
}

void MySgemm( cublasOperation_t transa,
              cublasOperation_t transb,
              int m,
              int n,
              int k,
              float alpha, /* host or device pointer */
              const float* A,
              int lda,
              const float* B,
              int ldb,
              float beta, /* host or device pointer */
              float* C,
              int ldc)
{
    // currently only support row major input
    assert(transa == CUBLAS_OP_T);
    assert(transb == CUBLAS_OP_T);
    assert(m % 128 == 0);
    assert(n % 128 == 0);
    assert(k % 128 == 0);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(n/TILE_N, m/TILE_M);
    if( n % TILE_N != 0 ) dimGrid.x += 1;
    if( m % TILE_M != 0 ) dimGrid.y += 1;
    int cTileK = k / TILE_K;
    if( k % TILE_K != 0 ) cTileK++;
    MatrixMul_T_T<<<dimGrid, dimBlock>>>(cTileK, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

int main() {
    const int m = 2048;
    const int k = 1024;
    const int n = 2048;
    const int iter_count = 500;
    cv::Mat matA(m, k, CV_32FC1);
    cv::randn(matA, 0.5, 1);
    cv::Mat matB(k, n, CV_32FC1);
    cv::randn(matB, 0.5, 1);
    auto cpuT1 = std::chrono::high_resolution_clock::now();
    cv::Mat matC = matA * matB;
    auto cpuT2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(cpuT2 - cpuT1).count() / 1000000.0 << std::endl;
    cv::Mat matD(n, m, CV_32FC1);
    cv::Mat matE(m, n, CV_32FC1);

    auto pMatA = MyCudaMalloc(m*k*sizeof(float));
    cudaMemcpy(pMatA.get(), (void*)matA.data, matA.rows * matA.cols * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    auto pMatB = MyCudaMalloc(k*n*sizeof(float));
    cudaMemcpy(pMatB.get(), (void*)matB.data, matB.rows * matB.cols * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    auto pMatC = MyCudaMalloc(m*n*sizeof(float));
    auto pMatD = MyCudaMalloc(m*n*sizeof(float));
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    float alpha = 1, beta = 0;
    double total = 0;
    for( int i=0; i<iter_count; ++i)
    {
        cudaEvent_t evtStart, evtFinish;
        cudaEventCreate(&evtStart);
        cudaEventCreate(&evtFinish);
        cudaMemset(pMatC.get(), 0, m*n*sizeof(float));
        cudaEventRecord(evtStart, cudaStreamDefault);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, (float*)pMatA.get(), k, (float*)pMatB.get(), n, &beta, (float*)pMatC.get(), m );
        cudaEventRecord(evtFinish, cudaStreamDefault);
        cudaEventSynchronize(evtFinish);
        float ms = 0;
        cudaEventElapsedTime(&ms, evtStart, evtFinish);
        total += ms;
        cudaEventDestroy(evtStart);
        cudaEventDestroy(evtFinish);
    }
    cudaMemcpy(matD.data, pMatC.get(), m*n*sizeof(float), cudaMemcpyDeviceToHost);
    matD = matD.t();
    std::cout << total / 1000.0 / iter_count  << std::endl;

    total = 0;
    for( int i=0; i<iter_count; ++i)
    {
        cudaEvent_t evtStart, evtFinish;
        cudaEventCreate(&evtStart);
        cudaEventCreate(&evtFinish);
        cudaMemset(pMatD.get(), 0, m*n*sizeof(float));
        cudaEventRecord(evtStart, cudaStreamDefault);
        MySgemm(CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, (float*)pMatA.get(), k, (float*)pMatB.get(), n, beta, (float*)pMatD.get(), n );
        cudaEventRecord(evtFinish, cudaStreamDefault);
        cudaEventSynchronize(evtFinish);
        float ms = 0;
        cudaEventElapsedTime(&ms, evtStart, evtFinish);
        total += ms;
        cudaEventDestroy(evtStart);
        cudaEventDestroy(evtFinish);
    }
    std::cout << total / 1000.0 / iter_count << std::endl;

    cudaMemcpy(matE.data, pMatD.get(), m*n*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<m; ++i)
    {
        std::cout << matD.at<float>(0, i) << "\t" << matE.at<float>(0, i) << "\t" << matC.at<float>(0, i) << std::endl;
    }

//    cv::imshow("test", (matE - matC)*500);
//    cv::waitKey(0);

    return 0;
}
