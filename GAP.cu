#include <iostream>
#include <fstream>
#include <climits>
#include "GAP.cuh"
#include "gpu.cuh"
#include "node.cuh"
#include "helper.cuh"
using namespace std;


GeneralizedAssignemnt::GeneralizedAssignemnt() {
    //ctor
    cap_cpu = nullptr;
    cap = nullptr;
    c_cpu = nullptr;
    c = nullptr;
    req_cpu = nullptr;
    req = nullptr;
    sol = nullptr;
    solbest = nullptr;
    randStates = nullptr;
    temp_req = nullptr;
    f = nullptr;
    final_f = nullptr;
    anss = nullptr;
    val = nullptr;
    Ksol = nullptr;
    temp_ksol = nullptr;
    temp_cost = nullptr;
    fsol = nullptr;
    temp_res = nullptr;
    temp_subgrad = nullptr;
    capleft = nullptr;
    temp_c = nullptr;
    checkCudaErrors(cudaMalloc((void **)&temp_var, sizeof(node<double>)));
    EPS = 0.0001;
}


GeneralizedAssignemnt::~GeneralizedAssignemnt() {
    //dtor
    if (cap!=nullptr) checkCudaErrors(cudaFree(cap));
    if (c!=nullptr)    checkCudaErrors(cudaFree(c));
    if (req!=nullptr) checkCudaErrors(cudaFree(req));
    if (sol!=nullptr) checkCudaErrors(cudaFree(sol));
    if (solbest!=nullptr) checkCudaErrors(cudaFree(solbest));
    if (randStates!=nullptr) checkCudaErrors(cudaFree(randStates));
    if (temp_req!=nullptr) checkCudaErrors(cudaFree(temp_req));
    if (temp_reduction!=nullptr) checkCudaErrors(cudaFree(temp_reduction));
    if (f!=nullptr) checkCudaErrors(cudaFree(f));
    if (final_f!=nullptr) checkCudaErrors(cudaFree(final_f));
    if (anss!=nullptr) checkCudaErrors(cudaFree(anss));
    if (val!=nullptr) checkCudaErrors(cudaFree(val));
    if (Ksol!=nullptr) checkCudaErrors(cudaFree(Ksol));
    if (temp_ksol!=nullptr) checkCudaErrors(cudaFree(temp_ksol));
    if (temp_cost!=nullptr) checkCudaErrors(cudaFree(temp_cost));
    if (fsol!=nullptr) checkCudaErrors(cudaFree(fsol));
    if (temp_res!=nullptr) checkCudaErrors(cudaFree(temp_res));
    if (temp_subgrad!=nullptr) checkCudaErrors(cudaFree(temp_subgrad));
    if (capleft!=nullptr) checkCudaErrors(cudaFree(capleft));
    if (temp_c!=nullptr) checkCudaErrors(cudaFree(temp_c));
    checkCudaErrors(cudaFree(temp_var));

    if (c_cpu != nullptr) delete[] c_cpu;
    if (req_cpu != nullptr) delete[] req_cpu;
    if (cap_cpu != nullptr) delete[] cap_cpu;
}

/***********************************************
 * 初始化工厂已用容量        【O(m)->O(1)】
 * 判断得到总花费             【O(n)->O(log_n)】
 * 判断资源超限:              【O(n)->O(log_m)】
 *     申请矩阵空间 
 *     将解向量扩展为解矩阵  【O(1)】
 *     归约求和                 【O(log_m)】
 *     (需先将解化为矩阵，右乘全1矩阵（可使用cudnn）/直接归约，以得到各个工厂的资源情况)
 *     (解矩阵越稠密，即工厂数相对于工作数越小，优化越大)
 *     (也可以直接由线程0进行操作)
 *     结果对比资源总量        【O(1)】(使用syn_or/and)
 * 
 * 【O(max(m,n))->O(max(log_m,log_n))】
***********************************************/
// 得到当前解的总花费
// controllo ammissibilità soluzione
int GeneralizedAssignemnt::checkSol(int* sol)
{  

    int cost=0;                             // 总花费
    int res_cmp = 0;                      // 用于判断是否工厂全部满足容量

    // 各工作当前花费，并求和
    checkInit<<<NumBlocks(n), NUM_THREADS>>>(temp_cost, sol, c, n, m);
    // debug_vector<<<1,1>>>(44444, temp_cost, 1, n);
    int* temp = get_temp(cost);
    //printf("1\n");
    doReduction(temp_cost, 1, n, temp, 2);
    cost = delete_temp(temp);
    if(cost == INT_MAX)
        return cost;

    // 各工厂已占用资源
    vectorInit<int><<<NumBlocks(m*n), NUM_THREADS>>>(temp_req, n*m, 0);
    vectorToMatrixReq<<<NumBlocks(n), NUM_THREADS>>>(sol, temp_req, req, n, m);
    //printf("2\n");
    doReduction(temp_req, m, n, temp_req, 2);
    // 与资源可用数进行比较，若超过，则res!=0
    temp = get_temp(res_cmp);
    vectorCmpLess<int><<<NumBlocks(m), NUM_THREADS>>>(temp_req, cap, m, temp);
    res_cmp = delete_temp(temp);
    // debug_vector<<<1,1>>>(55555, sol, 1, n);
    // debug_vector<<<1,1>>>(55555, temp_req, 1, m);
    // debug_vector<<<1,1>>>(-55555, cap, 1, m);
    // printf("RSD::%d\n",res_cmp);
    if(res_cmp != 0) {
        return INT_MAX;
    }
    return cost;
}

__global__ void get_sort(int* A, int* B, int n, int* P){
    // tid：线程号  bid：块号    tnum:该block中计算的thread数目
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    
    if(index < n){
        A[index] = index;
        B[index] = P[index];
    }
}


/***********************************************
 * 初始化            【O(max(m,n))->O(1)】
 * 删超限解，更新容量 【O(n)】
 *    (暂无法并行)
 * 计算新解          【n】*【O(m)】
 *    (由于每个结果影响后面的，对于工作，无法并行)
 * 更新全局解        【O(n)->O(1)】
 * 
 * 【O(n*m)】
***********************************************/
// 贪心修正可行解，按顺序删除超限的解，目标函数值为当前下界、
// 参数1：旧解->新解 参数2：总花费
// recovers feasibility in case of partial or overassigned solution
int GeneralizedAssignemnt::fixSol(int* infeasSol, int* zsol)
{  int i,j,imin=-1;
   int minreq;
   int* capres = new int[m];
   int* sol = new int[n];

   // 初始化限制与旧解
   memcpy(capres, cap_cpu, sizeof(int) * m);
   checkCudaErrors(cudaMemcpy(sol, infeasSol, sizeof(int) * n, cudaMemcpyDeviceToHost));

   // 重新计算剩余容量。如果容量不足，则将此处解设置为-1
   // ricalcolo capacità residue. Se sovrassegnato, metto a sol a -1
   for(j=0;j<n;j++)
      if(sol[j]>=0 && (capres[sol[j]] >= req_cpu[ID2D(sol[j], j, n)]))
         capres[sol[j]] -= req_cpu[ID2D(sol[j], j, n)];
      else
         sol[j] = -1;

   // 当前总花费
   *zsol = 0;
   for(j=0;j<n;j++)
   {  
      // 不改变未超限的解
      if(sol[j]>=0)              // correct, do nothing
      {  *zsol += c_cpu[ID2D(sol[j], j, n)];
         continue;
      }

      // 对于i=-1，即超过工厂容量的工作解，获得新解
      // reassign i -1
      minreq = INT_MAX;
      imin = -1;
      // 遍历工厂，找到最后一个可以容纳的工厂
      for(i=0;i<m;i++)
         if(capres[i]>=req_cpu[ID2D(i, j, n)] && req_cpu[ID2D(i, j, n)] < minreq)
         {  minreq = req_cpu[ID2D(i, j, n)];
            imin    = i;
         }

      // 无解：设总花费为INT_MAX，并返回
      if(imin<0)
      {  *zsol = INT_MAX;
         delete capres;
         delete sol;
         return *zsol;           // could not recover feasibility
      }

      // 有解：更新解
      sol[j]=imin;
      capres[imin] -= req_cpu[ID2D(imin, j, n)];
      *zsol += c_cpu[ID2D(imin, j, n)];
   }

   checkCudaErrors(cudaMemcpy(infeasSol, sol, sizeof(int) * n, cudaMemcpyHostToDevice));
   // 如果更新后的解小于上限（最优解），更新并输出信息
   if(*zsol<zub)
   {  
      vectorCopy<<<NumBlocks(n), NUM_THREADS>>>(infeasSol, solbest, n);
      zub = *zsol;
      if(isVerbose) cout << "[fixSol] -------- zub improved! " << zub << endl;
   }
   
   delete capres;
   delete sol;
   return *zsol;
}

// reads instance data from files
void GeneralizedAssignemnt::readData(string filePath, unsigned long seed) {
    ifstream ifs(filePath);
    if (! ifs.is_open()) {
        cout << "Error opening file" << endl;
        return;
    }

    ifs >> m >> n;

    checkCudaErrors(cudaMalloc((void **)&cap, sizeof(int) * m));
    checkCudaErrors(cudaMalloc((void **)&c, sizeof(int) * m * n));
    checkCudaErrors(cudaMalloc((void **)&req, sizeof(int) * m * n));

    c_cpu = new int[m * n];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            ifs >> c_cpu[ID2D(i, j, n)];
    checkCudaErrors(cudaMemcpy(c, c_cpu, m * n * sizeof(int), cudaMemcpyHostToDevice));
    req_cpu = new int[m * n];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            ifs >> req_cpu[ID2D(i, j, n)];
    checkCudaErrors(cudaMemcpy(req, req_cpu, m * n * sizeof(int), cudaMemcpyHostToDevice));
    cap_cpu = new int[m];
    for (int i = 0; i < m; i++)
        ifs >> cap_cpu[i];
    checkCudaErrors(cudaMemcpy(cap, cap_cpu, m * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&sol, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&solbest, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&randStates, sizeof(curandState) * n));
    randStatesInit<<<NumBlocks(n), NUM_THREADS>>>(randStates, seed, n);
    checkCudaErrors(cudaMalloc((void **)&temp_req, sizeof(int) * n * m));
    checkCudaErrors(cudaMalloc((void **)&capleft, sizeof(int) * m));
    temp_reduction = capleft;
    // Kcap归约获得
    int* dev_Kcap = get_temp(Kcap);
    doReduction(cap, 1, m, dev_Kcap, 1);
    Kcap = delete_temp(dev_Kcap);
    Kcap++;
    temp_reduction = nullptr;
    checkCudaErrors(cudaMalloc((void **)&temp_reduction, sizeof(node<double>)* max(n, Kcap) * m));
    checkCudaErrors(cudaMalloc((void **)&f, sizeof(double) * n * Kcap * m));
    checkCudaErrors(cudaMalloc((void **)&final_f, sizeof(node<double>) * Kcap * m));
    checkCudaErrors(cudaMalloc((void **)&anss, sizeof(node<int>) * n * m));
    checkCudaErrors(cudaMalloc((void **)&val, sizeof(double) * m * n));
    checkCudaErrors(cudaMalloc((void **)&Ksol, sizeof(int) * m * n));
    checkCudaErrors(cudaMalloc((void **)&temp_ksol, sizeof(node<int>) * m * n));
    checkCudaErrors(cudaMalloc((void **)&temp_cost, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&fsol, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&temp_res, sizeof(node<int>) * n));
    checkCudaErrors(cudaMalloc((void **)&temp_subgrad, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&temp_c, sizeof(int) * n));
   


    // printf("GAP Malloc ! YES\n");

    

    zub = INT_MAX;
    if (isVerbose)
        cout << "data read, n="<< n << " m="<< m << endl;
    // debug_vector<<<1,1>>>(-1,cap,1,m);
    // debug_vector<<<1,1>>>(-2,c,m,n);
    // debug_vector<<<1,1>>>(-3,req,m,n);
}



// 使用knapsack背包，n:物品数目，Kcap:背包容量，Q:各物品大小，val:花费（-拉格朗日乘子）取反，Ksol:knapsack结果 
// dynamic programming recursion for the knapsack
/****************************************
 * (为简化问题，此处假设背包容量在 threads pre block 范围内)
 * (由于一次背包为对一个工厂求解，设每个block为1个工厂，每种背包容量为一个thread)
 * 依次遍历每个物品dp 【n】*【O(Kcap)->O(2)】
 *     (由于更新时的冲突，每次dp整体更新都需要同步，共同步2*n次，一次读，一次写)
 * 得到结果             【O(Kcap)->O(log_Kcap)】
 *     (归约找到最大的值，即最小的总容量)
 * 递归decode操作     【<O(n)】
 *     （无并行性，仅由线程0进行）
 * max(2*n,log_Kcap,n) / max(dp，最优花费，倒推最优解)  
 * 
 * 【O(m*n*Kcap)->O(max(n,log_Kcap))】
******************************************/

// 原函数为一个工厂，此处改成多个工厂并行
// 空间换时间，总空间需要m*n*Kcap
double GeneralizedAssignemnt::KDynRecur(int m, int n, int* cap, int* req, double* val, int* Ksol)
{  
    //cout << n << endl;

    // 初始化数组f为0
    vectorInit<double><<<NumBlocks(n*Kcap*m), NUM_THREADS>>>(f, n*Kcap*m, 0.0);

    // printf("m:%d Kcap:%d n:%d\n", m, Kcap, n);
    // debug_vector<<<1,1>>>(10001,req,m,n);
    // debug_vector<<<1,1>>>(10002,val,m,n);
    // debug_vector<<<1,1>>>(10003,cap,1,m);
    // DP求解，顺序加入每个物品
    for (int i = 0; i < n; i++) {
        //cout << "kdynDP " << i << endl;
        kdynDP<<<NumBlocks(Kcap*m), NUM_THREADS>>>(f, m, Kcap, n, req, val, cap, i);
    }
    // 归约得到各个工厂遍历到第n个物品时的各容量的最优解，pos为[Kcap]
    // debug_vector<<<1,1>>>(10000, f+(Kcap*m)*(n-1), Kcap, m);
    //cout << "vectorToNode" << endl;
    //cout << (n-1)*Kcap*m << endl;
    vectorToNode<<<NumBlocks(Kcap*m), NUM_THREADS>>>(f + (n-1) * Kcap*m, final_f, Kcap*m, m);
    //cout << "vectorToNode Finished" << endl;
    //printf("6\n");
    doReduction(final_f, m, Kcap, final_f, 1);
    // 根据res反推路径
    //cout << "kdynDecode" << endl;
    kdynDecode<<<NumBlocks(m), NUM_THREADS>>>(m, Kcap, final_f, req, val, n, f, Ksol);

    // 归约得到各个工厂最优解的总和
    //printf("7\n");
    doReduction(final_f, 1, m, final_f, 2);
    
    node<double> res;
    checkCudaErrors(cudaMemcpy(&res, final_f, sizeof(node<double>), cudaMemcpyDeviceToHost));
    // 释放空间

    return res.data;
}
