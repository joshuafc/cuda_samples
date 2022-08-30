#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _fe384d24_7_main_cu_aced82d4
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z13MatrixMul_T_TiiiifPKfiS0_ifPfi(int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z13MatrixMul_T_TiiiifPKfiS0_ifPfi(int __par0, int __par1, int __par2, int __par3, float __par4, const float *__par5, int __par6, const float *__par7, int __par8, float __par9, float *__par10, int __par11){__cudaLaunchPrologue(12);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__cudaSetupArgSimple(__par2, 8UL);__cudaSetupArgSimple(__par3, 12UL);__cudaSetupArgSimple(__par4, 16UL);__cudaSetupArgSimple(__par5, 24UL);__cudaSetupArgSimple(__par6, 32UL);__cudaSetupArgSimple(__par7, 40UL);__cudaSetupArgSimple(__par8, 48UL);__cudaSetupArgSimple(__par9, 52UL);__cudaSetupArgSimple(__par10, 56UL);__cudaSetupArgSimple(__par11, 64UL);__cudaLaunch(((char *)((void ( *)(int, int, int, int, float, const float *, int, const float *, int, float, float *, int))MatrixMul_T_T)));}
# 24 "/home/zhoub/CLionProjects/cudaTest/main.cu"
void MatrixMul_T_T( int __cuda_0,int __cuda_1,int __cuda_2,int __cuda_3,float __cuda_4,const float *__cuda_5,int __cuda_6,const float *__cuda_7,int __cuda_8,float __cuda_9,float *__cuda_10,int __cuda_11)
# 25 "/home/zhoub/CLionProjects/cudaTest/main.cu"
{__device_stub__Z13MatrixMul_T_TiiiifPKfiS0_ifPfi( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11);
# 223 "/home/zhoub/CLionProjects/cudaTest/main.cu"
}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T6) {  __nv_dummy_param_ref(__T6); __nv_save_fatbinhandle_for_managed_rt(__T6); __cudaRegisterEntry(__T6, ((void ( *)(int, int, int, int, float, const float *, int, const float *, int, float, float *, int))MatrixMul_T_T), _Z13MatrixMul_T_TiiiifPKfiS0_ifPfi, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
