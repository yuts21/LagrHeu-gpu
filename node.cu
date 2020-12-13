#include "node.cuh"
#include "helper.cuh"

template class node<int>;
template class node<double>;

node<double> NDinit;
void *temp_var;

// __device__ int temp_int;
// __device__ double temp_double_1;
// __device__ double temp_double_2;

template<typename T> __device__ __host__  T maxx(T a, T b){
    return a > b ? a : b;
}

template<typename T> __device__ __host__  T minn(T a, T b){
    return a < b ? a : b;
}

template __device__ __host__ int maxx(int a, int b);
template __device__ __host__  double maxx(double a, double b);
template __device__ __host__  int minn(int a, int b);
template __device__ __host__  double minn(double a, double b);

template<typename T> bool __device__ __host__ node<T>::operator < (const node<T> &B){
    if(data < B.data) return true;
    else if(data == B.data && pos < B.pos) return true;
    else return false;
};
// template __device__ __host__ bool node<int>::operator < (const node<int> &B);
// template __device__ __host__ bool node<double>::operator < (const node<double> &B);

template<typename T> bool __device__ __host__ node<T>::operator > (const node<T> &B){
    if(data > B.data) return true;
    else if(data == B.data && pos < B.pos) return true;
    else return false;
};
// template __device__ __host__ bool node<int>::operator > (const node<int> &B);
// template __device__ __host__ bool node<double>::operator > (const node<double> &B);

template<typename T> node<T> __device__ __host__ node<T>::operator + (const node<T> &B){
    node<T> res;
    res.pos = minn(pos, B.pos);
    if(data == INT_MAX || B.data == INT_MAX)
        res.data = INT_MAX;
    else
        res.data = data + B.data;
    return res;
};
// template __device__ __host__ node<int> node<int>::operator + (const node<int> &B);
// template __device__ __host__ node<double> node<double>::operator + (const node<double> &B);

template<typename T> node<T> __device__ __host__ node<T>::operator - (const node<T> &B){
    node<T> res;
    res.pos = minn(pos, B.pos);
    res.data = data - B.data;
    return res;
};
// template __device__ __host__ node<int> node<int>::operator - (const node<int> &B);
// template __device__ __host__ node<double> node<double>::operator - (const node<double> &B);

/*
template<typename T> node<T> & __device__ __host__ node<T>::operator = (const node<T> &B){
    pos = B.pos;
    data = B.data;
    return *this;
};


// template __device__ __host__ node<int>& node<int>::operator = (const node<int> &B);
// template __device__ __host__ node<double>& node<double>::operator = (const node<double> &B);
*/


void __device__ __host__ copy(node<int> *A, const node<int> B)
{
    A->data = B.data;
    A->pos = B.pos;
}
void __device__ __host__ copy(node<double> *A, const node<double> B)
{
    A->data = B.data;
    A->pos = B.pos;
}
void __device__ __host__ copy(int *A, const int B)
{
    *A=B;
}
void __device__ __host__ copy(double *A, const double B)
{
    *A=B;
}

int __device__ __host__ get_data(node<int> A)
{
    return A.data;
}
double __device__ __host__ get_data(node<double> A)
{
    return A.data;
}
int __device__ __host__ get_data(int A)
{
    return A;
}
double __device__ __host__ get_data(double A)
{
    return A;
}

int __device__ __host__ get_pos(node<int> A)
{
    return A.pos;
}
int __device__ __host__ get_pos(node<double> A)
{
    return A.pos;
}
int __device__ __host__ get_pos(int A)
{
    return 0;
}
int __device__ __host__ get_pos(double A)
{
    return 0;
}

template<typename T>  __host__  T* get_temp(T a){
    T* temp = (T*)temp_var;
    vectorInit<<<1,1>>>(temp, 1, a);
    return temp;
}
template  __host__ int* get_temp(int a);
template  __host__ double* get_temp(double a);
template  __host__ node<int>* get_temp(node<int> a);
template  __host__ node<double>* get_temp(node<double> a);

template<typename T>  __host__  T delete_temp(T* a){
    T ans;
    checkCudaErrors(cudaMemcpy(&ans, a, sizeof(T), cudaMemcpyDeviceToHost));
    return ans;
}
template  __host__ int delete_temp(int* a);
template  __host__ double delete_temp(double* a);
template  __host__ node<int> delete_temp(node<int>* a);
template  __host__ node<double> delete_temp(node<double>* a);


template<typename T>  bool __device__ __host__ operator < (const node<T> &A, const node<T> &B){
    if(A.data < B.data) return true;
    else if(A.data == B.data && A.pos < B.pos) return true;
    else return false;
};

template __device__ __host__ bool operator < (const node<int> &A, const node<int> &B);
template __device__ __host__ bool operator < (const node<double> &A, const node<double> &B);
template __device__ __host__  node<int> maxx(node<int> a, node<int> b);
template __device__ __host__  node<double> maxx(node<double> a, node<double> b);
template __device__ __host__  node<int> minn(node<int> a, node<int> b);
template __device__ __host__  node<double> minn(node<double> a, node<double> b);


template<typename T> __global__ void vectorToNode(T* S, node<T>* V, int n, int p)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n) {
        V[index].data = S[index];
        V[index].pos = index / p;
    }
}


// 将node数组提取pos
template<typename T> __global__ void nodeToPos(node<T>* S, int* V, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n) {
        if(S[index].data == 0) V[index] = -1;
        else V[index] = S[index].pos;
    }
}

// 将node数组提起data
template<typename T> __global__ void nodeToData(node<T>* S, T* V, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        V[index] = S[index].data;
}

template __global__ void vectorToNode<int>(int* S, node<int>* V, int n, int p);
template __global__ void vectorToNode<double>(double* S, node<double>* V, int n, int p);
template __global__ void nodeToPos<int>(node<int>* S,int* V, int n);
template __global__ void nodeToPos<double>(node<double>* S,int* V, int n);
template __global__ void nodeToData<int>(node<int>* S,int* V, int n);
template __global__ void nodeToData<double>(node<double>* S,double* V, int n);


__device__ __host__ int NumBlocks(int n){return (n+NUM_THREADS-1)/NUM_THREADS;};
__device__ __host__ int ID2D(int i, int j, int n){return (i*n+j);};



// 数组初始化为s
template<typename T> __global__ void vectorInit(T* V, int n, T s)
{
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        V[index] = s;
    
}



// 数组初始化为序号
template<typename T> __global__ void vectorInitIndex(T* V, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        V[index] = (T)index;
}



// 数组拷贝
template<typename T> __global__ void vectorCopy(T* S, T* V, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        copy(V + index, S[index]);
}



// 数组相加
template<typename T> __global__ void vectorAdd(T* S1, T* S2, T* V, int n)
{ 
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        copy(&V[index], S1[index] + S2[index]);
}




// 数组相减
template<typename T> __global__ void vectorSub(T* S1, T* S2, T* V, int n)
{ 
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        copy(&V[index], S1[index] - S2[index]);
}


// 数组比较，若S均<=T，则res=0，否则res非零
template<typename T> __global__ void vectorCmpLess(T* S, T* V, int n, int* res)
{ 
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    int pd = 0;
    if(index < n)
        pd = (S[index] - V[index] > 0) ? 1 : 0;
    *res = __syncthreads_or(pd);
}


template<typename T> __global__ void debug_vector(int id, T* V, int m, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index == 0){
        printf("【DEBUG】 %d\n", id);
        for(int i = 0; i < m; ++i){
            printf("\t");
            for(int j = 0; j < n; ++j)
                printf("(%lf,%d)\t", (double)get_data(V[i*n+j]), get_pos(V[i*n+j]));
            printf("\n");
        }
    }
}
template __global__ void debug_vector<int>(int id, int* V, int m, int n);
template __global__ void debug_vector<double>( int id, double* V, int m, int n);
template __global__ void debug_vector<node<int> >(int id, node<int>* V, int m, int n);
template __global__ void debug_vector<node<double> >( int id, node<double>* V, int m, int n);


template __global__ void vectorInit<int>(int* V, int n, int s);
template __global__ void vectorInit<double>(double* V, int n, double s);

template __global__ void vectorInitIndex<int>(int* V, int n);
template __global__ void vectorInitIndex<double>(double* V, int n);
template __global__ void vectorCopy<int>(int* S,int* V, int n);
template __global__ void vectorCopy<double>(double* S,double* V, int n);
template __global__ void vectorCopy<node<int> >(node<int>* S,node<int>* V, int n);
template __global__ void vectorCopy<node<double> >(node<double>* S,node<double>* V, int n);
template __global__ void vectorAdd<int>(int* S1, int* S2, int* V, int n);
template __global__ void vectorAdd<double>(double* S1, double* S2, double* V, int n);
template __global__ void vectorAdd<node<int> >(node<int>* S1, node<int>* S2, node<int>* V, int n);
template __global__ void vectorAdd<node<double> >(node<double>* S1, node<double>* S2, node<double>* V, int n);
template __global__ void vectorSub<int>(int* S1, int* S2, int* V, int n);
template __global__ void vectorSub<double>(double* S1, double* S2, double* V, int n);
template __global__ void vectorSub<node<int> >(node<int>* S1, node<int>* S2, node<int>* V, int n);
template __global__ void vectorSub<node<double> >(node<double>* S1, node<double>* S2, node<double>* V, int n);
template __global__ void vectorCmpLess<int>(int* S, int* V, int n, int* res);
template __global__ void vectorCmpLess<double>(double* S, double* V, int n, int* res);

