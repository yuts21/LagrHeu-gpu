#include <iostream>
#include "GAP.cuh"
#include "Lagrangian.cuh"
#include "gpu.cuh"
#include "helper.cuh"
using namespace std;


/******************
 *
 * 需要预先将所有相关数据放入gpu中，方便运算
 * 根据算法复杂度，优化关键步骤即可，无需全部优化 
 * 
*******************/

/*
__global__ void dodo(int*V)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index == 0)
        printf("0x%x %d", (long)V, *V);
        *V = 1020;
}*/

int main(int argc, char *argv[]) {
    size_t heapsize = sizeof(double) * 1600*800*8000;
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize));

    // GAP类与对象
    GeneralizedAssignemnt* GAP = new GeneralizedAssignemnt();
    int res;

    bool isVerbose = true; // 是否输出调试信息
    string filePath = argv[1]; // 数据文件相对路径名
    if (filePath.empty()) {
        cout << "Input filePath is empty!" << endl;
        return 1;
    }
    double alpha = 2.546;
    double alphastep = 0.936; // alpha更新的频率
    double minalpha = 0.01;
    int innerIter = 98;     // 每循环x次，更新alpha
    int maxIter = 619;       // 最多循环次数
    unsigned long seed = 505;
    if (isVerbose) {
       cout << "alpha="      << alpha << endl;
       cout << "alphastep="  << alphastep << endl;
       cout << "minalpha="   << minalpha << endl;
       cout << "innerIter="  << innerIter << endl;
       cout << "maxiter="    << maxIter << endl;
       cout << "seed="       << seed << endl;
       cout << "inputFile: " << filePath << endl;
    }
    GAP->isVerbose = isVerbose;

    // 读取json格式存储的信息
    GAP->readData(filePath, seed);

    maxIter = maxIter*GAP->n*GAP->m;

    clock_t start_t = clock();
    // LAGR类与对象

    Lagrangian* LAGR = new Lagrangian(GAP, GAP->zub);

    if (isVerbose)
        cout << "Relaxing assignments ---------------" << endl;
    res = LAGR->lagrCap(GAP->c, alpha, alphastep, minalpha, innerIter, maxIter);
    delete LAGR;

    clock_t end_t = clock();
    if (isVerbose)
        cout << "Time: " << (double)(end_t - start_t)/CLOCKS_PER_SEC << endl;
    cout << GAP->zub << endl;
    int *solbest = new int[GAP->n];
    checkCudaErrors(cudaMemcpy(solbest, GAP->solbest, GAP->n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < GAP->n - 1; i++)
        cout << solbest[i] << " ";
    cout << solbest[GAP->n - 1] << endl;
    delete solbest;

    if (isVerbose) {
        cout << "\n<ENTER> to exit ..."; 
        cin.get();
    }
    delete GAP;
    return 0;
}
