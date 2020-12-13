#ifndef GPU_H
#define GPU_H
// CUDA runtimenv
#include "node.cuh"
#include "sharedmem.cuh"
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// reduction操作所使用的数组空间，其中以最大的T类型(node<double>)为准，而空间大小由max(n*m, m*Kcap)决定
extern void* temp_reduction;

__global__ void subInit(int* subgrad, int* lbsol, int n);

// 将解数组转为解矩阵的req值
__global__ void vectorToMatrixReq(int* sol, int* temp_req, int* req, int n,  int m);

// 使用归约方法计算，op为0时求最小值。op为1时求最大值，op为2时求和
// ans使用指针传入
// 每隔m个block计算同一个数组的归约，最终将结果分别归约在前m个block中
template<typename T> __global__ void vectorReduction(T* V, int m, int n, int bnum, T* ans, int op);

extern template __global__ void vectorReduction<int>(int* V, int m, int n, int bnum,int* ans, int op);
extern template __global__ void vectorReduction<double>(double* V, int m, int n, int bnum,double* ans, int op);


extern template __global__ void vectorReduction<node<int> >(node<int>* V, int m, int n, int bnum,node<int>* ans, int op);
extern template __global__ void vectorReduction<node<double> >(node<double>* V, int m, int n, int bnum,node<double>* ans, int op);



/*******************
* 当n超过NUM_THREADS，需要进行多次归约
*    举例：当每个block最大1024个线程，有1600个工作。
*          并行2个block，第i个计算V[0...799][i]，并归约到V[i]
*          下一次计算时，使用1个block，归约这两个数到0号位置
* 又有m个并行的归约操作时，数组需要以V[n][m]的顺序存储
*    举例：当每个block最大1024个线程，有1600个工作，80个工厂并行
*          并行2*80个block，第k个block的第i块，计算V[0...799][i][k]，并归约到V[i][k]
*          下一次计算时，使用80个block，分别归约V[0][k]、V[1][k]到V[k]
* 这样间隔计算的优点是：每个block进行计算时直接归约到自己计算的第一个位置上
*      没有读写冲突，且该位置在整体上为起始的连续内存；省去了内存复制或block间同步的时间
*/
// 数组归约操作，m个长度为n的数组并行，分别归约
// op：0：最小值、1：最大值、2：求和

template<typename T> void doReduction(T* V, int m, int n, T* ans, int op);
extern template void doReduction<int>(int* V, int m, int n, int* ans, int op);
extern template void doReduction<double>(double* V, int m, int n, double* ans, int op);
extern template void doReduction<node<int> >(node<int>* V, int m, int n, node<int>* ans, int op);
extern template void doReduction<node<double> >(node<double>* V, int m, int n, node<double>* ans, int op);

__global__ void randStatesInit(curandState* state, unsigned long seed, int n);

__global__ void subInit(int* subgrad, int* lbsol, int n);

__global__ void checkInit(int* temp_cost, int* sol, int* c, int n, int m);

__global__ void kdynInit(double* val, int* ksol, int* c, double* lambda, int m, int n);

__global__ void kdynDP(double* f, int m, int Kcap, int n, int* req, double* val, int* cap, int i);

__global__ void kdynDecode(int m, int i, node<double>* imax, int* req, double* val, int n, double* f, int* Ksol);

__global__ void lagrInit(int* subgrad, int* temp, int n);

__global__ void lagrUpdate(double* lambda, int* subgrad, double step, int n);

__global__ void opt10Init(int* temp_c, int* c, int* sol, int n);

__global__ void opt10Main(int* c, int* capleft, int* req, int* sol, int z, int m, int n, node<int>* ans);

__global__ void opt10Update(int* sol, int* capleft, int* req, int i, int j, int n);

#endif // GPU_H