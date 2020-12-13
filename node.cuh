#ifndef NODE_H
#define NODE_H
#include <cuda_runtime.h>
#include <iostream>
// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
// block sync
//#include <helper_cuda.h>
#include <cooperative_groups.h>

#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

using namespace std;

#ifndef NUM_GPU_H
#define NUM_GPU_H
#define NUM_THREADS 1024
#define BUM_BLOCKS 64
#endif

// __device__ extern int temp_int;
// __device__ extern double temp_double_1;
// __device__ extern double temp_double_2;

extern void* temp_var;

template<typename T> __device__ __host__  T maxx(T a, T b);
extern template __device__ __host__ int maxx(int a, int b);
extern template __device__ __host__  double maxx(double a, double b);

template<typename T> __device__ __host__  T minn(T a, T b);
extern template __device__ __host__  int minn(int a, int b);
extern template __device__ __host__  double minn(double a, double b);


template<typename T> class node{
public:
    int pos;
    T data;
public:
    bool __device__ __host__ operator < (const node<T> &B);
    bool __device__ __host__ operator > (const node<T> &B);
    node<T> __device__ __host__ operator + (const node<T> &B);
    node<T> __device__ __host__ operator - (const node<T> &B);
    // node<T> & __device__ __host__ operator = (const node<T> &B);
};
extern template class node<int>;
extern template class node<double>;

extern node<double> NDinit;

template<typename T>  bool __device__ __host__ operator < (const node<T> &A, const node<T> &B);
extern template __device__ __host__ bool operator < (const node<int> &A, const node<int> &B);
extern template __device__ __host__ bool operator < (const node<double> &A, const node<double> &B);


__device__ __host__ void copy( node<int> *A, const node<int> B);
__device__ __host__ void copy( node<double> *A, const node<double> B);
__device__ __host__ void copy( int *A, const int B);
__device__ __host__ void copy( double *A, const double B);



extern template __device__ __host__  node<int> maxx(node<int> a, node<int> b);
extern template __device__ __host__  node<double> maxx(node<double> a, node<double> b);


extern template __device__ __host__  node<int> minn(node<int> a, node<int> b);
extern template __device__ __host__  node<double> minn(node<double> a, node<double> b);

int __device__ __host__ get_data(node<int> A);
double __device__ __host__ get_data(node<double> A);
int __device__ __host__ get_data(int A);
double __device__ __host__ get_data(double A);
int __device__ __host__ get_pos(node<int> A);
int __device__ __host__ get_pos(node<double> A);
int __device__ __host__ get_pos(int A);
int __device__ __host__ get_pos(double A);


// 将数组封装为node(nata,pos)格式，以找到最大值/最小值/可行解的位置
template<typename T> __global__ void vectorToNode(T* S, node<T>* V, int n, int p);
extern template __global__ void vectorToNode<int>(int* S, node<int>* V, int n, int p);
extern template __global__ void vectorToNode<double>(double* S, node<double>* V, int n, int p);


// 将node数组提取pos
template<typename T> __global__ void nodeToPos(node<T>* S, int* V, int n);
extern template __global__ void nodeToPos<int>(node<int>* S,int* V, int n);
extern template __global__ void nodeToPos<double>(node<double>* S,int* V, int n);


// 将node数组提起data
template<typename T> __global__ void nodeToData(node<T>* S, T* V, int n);
extern template __global__ void nodeToData<int>(node<int>* S,int* V, int n);
extern template __global__ void nodeToData<double>(node<double>* S,double* V, int n);



__device__ __host__ int NumBlocks(int n);
__device__ __host__ int ID2D(int i, int j, int n);

// 将线程id、块id转换为数组id，二维/三维
__device__ __host__ void convertIDto2D(int bid, int tid, int n, int& x, int& y);
__device__ __host__ void convertIDto3D(int bid, int tid, int m, int n, int& x, int& y, int& z);



// 数组初始化为s
template<typename T> __global__ void vectorInit(T* V, int n, T s);
extern template __global__ void vectorInit<int>(int* V, int n, int s);
extern template __global__ void vectorInit<double>(double* V, int n, double s);


// 数组初始化为序号
template<typename T> __global__ void vectorInitIndex(T* V, int n);
extern template __global__ void vectorInitIndex<int>(int* V, int n);
extern template __global__ void vectorInitIndex<double>(double* V, int n);


// 数组拷贝
template<typename T> __global__ void vectorCopy(T* S, T* V, int n);
extern template __global__ void vectorCopy<int>(int* S,int* V, int n);
extern template __global__ void vectorCopy<double>(double* S,double* V, int n);
extern template __global__ void vectorCopy<node<int>>(node<int>* S,node<int>* V, int n);
extern template __global__ void vectorCopy<node<double>>(node<double>* S,node<double>* V, int n);


// 数组相加
template<typename T> __global__ void vectorAdd(T* S1, T* S2, T* V, int n);
extern template __global__ void vectorAdd<int>(int* S1, int* S2, int* V, int n);
extern template __global__ void vectorAdd<double>(double* S1, double* S2, double* V, int n);
extern template __global__ void vectorAdd<node<int>>(node<int>* S1, node<int>* S2, node<int>* V, int n);
extern template __global__ void vectorAdd<node<double>>(node<double>* S1, node<double>* S2, node<double>* V, int n);


// 数组相减
template<typename T> __global__ void vectorSub(T* S1, T* S2, T* V, int n); 
extern template __global__ void vectorSub<int>(int* S1, int* S2, int* V, int n);
extern template __global__ void vectorSub<double>(double* S1, double* S2, double* V, int n);
extern template __global__ void vectorSub<node<int>>(node<int>* S1, node<int>* S2, node<int>* V, int n);
extern template __global__ void vectorSub<node<double>>(node<double>* S1, node<double>* S2, node<double>* V, int n);

// 数组比较，若S均<=T，则res=0，否则res非零
template<typename T> __global__ void vectorCmpLess(T* S, T* V, int n, int* res);
extern template __global__ void vectorCmpLess<int>(int* S, int* V, int n, int* res);
extern template __global__ void vectorCmpLess<double>(double* S, double* V, int n, int* res);

template<typename T>  __global__ void debug_vector(int id, T* V, int m, int n);
extern template  __global__ void debug_vector<int>(int id, int* V, int m, int n);
extern template  __global__ void debug_vector<double>( int id, double* V, int m, int n);
extern template  __global__ void debug_vector<node<int>>(int id, node<int>* V, int m, int n);
extern template  __global__ void debug_vector<node<double>>( int id, node<double>* V, int m, int n);

template<typename T>  __host__  T* get_temp(T a);
extern template  __host__ int* get_temp(int a);
extern template  __host__ double* get_temp(double a);
extern template  __host__ node<int>* get_temp(node<int> a);
extern template  __host__ node<double>* get_temp(node<double> a);

template<typename T>  __host__  T delete_temp(T* a);
extern template  __host__ int delete_temp(int* a);
extern template  __host__ double delete_temp(double* a);
extern template  __host__ node<int> delete_temp(node<int>* a);
extern template  __host__ node<double> delete_temp(node<double>* a);


#endif
