#include <iostream>
#include <fstream>
#include <climits>
#include "GAP.cuh"
#include "gpu.cuh"
using namespace std;


GeneralizedAssignemnt::GeneralizedAssignemnt() {
    //ctor
    cap = nullptr;
    c = nullptr;
    req = nullptr;
    sol = nullptr;
    solbest = nullptr;
    randStates = nullptr;
    temp_req = nullptr;
    temp_req_node = nullptr;
    capres = nullptr;
    f = nullptr;
    final_f = nullptr;
    anss = nullptr;
    val = nullptr;
    Ksol = nullptr;
    temp_ksol = nullptr;
    temp_cost = nullptr;
    fsol = nullptr;
    whoIs = nullptr;
    q = nullptr;
    sort_data = nullptr;
    sort_keys = nullptr;
    temp_res = nullptr;
    temp_subgrad = nullptr;
    capleft = nullptr;
    temp_c = nullptr;
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
    if (temp_req_node!=nullptr) checkCudaErrors(cudaFree(temp_req_node));
    if (capres!=nullptr) checkCudaErrors(cudaFree(capres));
    if (temp_reduction!=nullptr) checkCudaErrors(cudaFree(temp_reduction));
    if (f!=nullptr) checkCudaErrors(cudaFree(f));
    if (final_f!=nullptr) checkCudaErrors(cudaFree(final_f));
    if (anss!=nullptr) checkCudaErrors(cudaFree(anss));
    if (val!=nullptr) checkCudaErrors(cudaFree(val));
    if (Ksol!=nullptr) checkCudaErrors(cudaFree(Ksol));
    if (temp_ksol!=nullptr) checkCudaErrors(cudaFree(temp_ksol));
    if (temp_cost!=nullptr) checkCudaErrors(cudaFree(temp_cost));
    if (fsol!=nullptr) checkCudaErrors(cudaFree(fsol));
    if (whoIs!=nullptr) checkCudaErrors(cudaFree(whoIs));
    if (q!=nullptr) checkCudaErrors(cudaFree(q));
    if (sort_data!=nullptr) checkCudaErrors(cudaFree(sort_data));
    if (sort_keys!=nullptr) checkCudaErrors(cudaFree(sort_keys));
    if (temp_res!=nullptr) checkCudaErrors(cudaFree(temp_res));
    if (temp_subgrad!=nullptr) checkCudaErrors(cudaFree(temp_subgrad));
    if (capleft!=nullptr) checkCudaErrors(cudaFree(capleft));
    if (temp_c!=nullptr) checkCudaErrors(cudaFree(temp_c));
    
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
    int* temp = get_temp(cost, 1);
    //printf("1\n");
    doReduction(temp_cost, 1, n, temp, 2);
    cost = delete_temp(temp);
    if(cost == INT_MAX) goto lendcheckSol;

    // 各工厂已占用资源
    vectorInit<int><<<NumBlocks(m*n), NUM_THREADS>>>(temp_req, n*m, 0);
    vectorToMatrixReq<<<NumBlocks(n), NUM_THREADS>>>(sol, temp_req, req, n, m);
    //printf("2\n");
    doReduction(temp_req, m, n, temp_req, 2);
    // 与资源可用数进行比较，若超过，则res!=0
    temp = get_temp(res_cmp, 1);
    vectorCmpLess<int><<<NumBlocks(m), NUM_THREADS>>>(temp_req, cap, m, temp);
    res_cmp = delete_temp(temp);
    // debug_vector<<<1,1>>>(55555, sol, 1, n);
    // debug_vector<<<1,1>>>(55555, temp_req, 1, m);
    // debug_vector<<<1,1>>>(-55555, cap, 1, m);
    // printf("RSD::%d\n",res_cmp);
    if(res_cmp != 0) {
        cost = INT_MAX;
        goto lendcheckSol;
    }
lendcheckSol:     
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
 * 初始化，只留最优解：
 *     初始化工厂剩余容量         【O(m)->O(1)】
 *     初始化解并更新工厂容量    【O(n)->O(log_n)】
 *     获得每个工作的最优解      【O(n*m)->O(log_m)】
 *     删非最优解，更新容量      【O(n)】
 * 将工厂按照剩余容量排序         【O(n*logn)->O(kn/nlogn/n)】
 * 排序后遍历工厂，背包求解【m】：
 * （对于上次计算结果，下一次不用再进行背包，故循环无法并行）
 *     初始化背包dp数组            【O(m)->O(1)】
 *     背包求解                     【O(n*Kcap)->O(max(n,log_Kcap))】
 *     更新解并计数                 【O(n)->O(logn)】
 * 更新全局解                        【O(n)->O(1)】
 * 
 * 【O(m*n*Kcap)->O(m*max(n,log_Kcap))】
***********************************************/
// 只保留最优解，根据剩余容量，通过背包，得到可行解
// recovers feasibility via knapsacks on residual capacities
int GeneralizedAssignemnt::fixSolViaKnap(int* infeasSol, int* zsol)
{  
    //debug_vector<<<1,1>>>(-192, infeasSol, 1, n);
    //cin.get();
    //printf("12345 %d\n", *zsol);
    int* sol = fsol;                  // 当前解
    int nelem = 0;                         // nelem：不可行解的数目
    // 比较两个工厂的剩余容量 
    // auto compCap = [&capres](int a, int b){ return capres[a] < capres[b]; };  // ompare, ASC order

    // 初始化总花费、工厂容量
    *zsol = INT_MAX;
    vectorCopy<<<NumBlocks(m), NUM_THREADS>>>(cap, capres, m);
    // 初始化解（有可能不可行），若旧解为空则设为随机，并将该解所占用的资源放入矩阵
    vectorInit<<<NumBlocks(n*m), NUM_THREADS>>>(temp_req, n*m, 0);
    //debug_vector<<<1, 1>>>(-11, sol, 1, n);
    fixsolInit<<<NumBlocks(n), NUM_THREADS>>>(sol, infeasSol, temp_req, req, n, m, randStates);
    //debug_vector<<<1, 1>>>(-12, sol, 1, n);
    // 得到各工厂剩余总容量
    //printf("3\n");
    doReduction(temp_req, m, n, temp_req, 2);
    vectorSub<<<NumBlocks(m), NUM_THREADS>>>(capres, temp_req, capres, m);
    // debug_vector<<<1,1>>>(-123, capres, 1, m);

    // 找到每个工作对应最小占用资源的工厂(以第一个为准)
    nelem = 0;
    vectorToNode<<<NumBlocks(m*n), NUM_THREADS>>>(req,temp_req_node, m*n, n);
    //printf("4\n");
    doReduction(temp_req_node, n, m, temp_req_node, 0);

    // 遍历若不可行，删去非最优解
    vectorInit<<<NumBlocks(n), NUM_THREADS>>>(whoIs, n, 0);
    // printf("nelem1:    %d\n", nelem);
    int* temp = get_temp(nelem,1);
    fixsolMain<<<1, 1>>>(temp_req_node, capres, sol, whoIs, req, cap, temp, n);
    // debug_vector<<<1,1>>>(-123, capres, 1, m);
    nelem = delete_temp(temp);
    //printf("nelem2:    %d\n", nelem);
    //debug_vector<<<1,1>>>(9285, )

    
    //double* val（此处使用大小为size(double)*n）      // 物品重量，由于重点为fix，尽量让工厂容纳多个工作，此处需求的工作设为1，其他设为-1
    //int*     Ksol（此处使用大小为size(int)*n）      // 对每个工厂进行背包，所得解
    
     // 使用cuda的Thrust库，将序号按照剩余容量进行排序
    
    
    int* indCap = new int[m];
    get_sort<<<NumBlocks(m), NUM_THREADS>>>(sort_data, sort_keys , m, capres);


    thrust::device_ptr<int> dev_data_ptr(sort_data);
    thrust::device_ptr<int> dev_keys_ptr(sort_keys);
/*
    thrust::host_vector<int> indCap, indCap2;
    for(int i = 0; i < n; ++i){
        indCap.push_back(i);
        indCap2.push_back(host_capres[i]);
    }
    thrust::device_vector<node<int>> d_vec = indCap;
    node<int>* dv_ptr = thrust::raw_pointer_cast(d_vec.data());
*/
    thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + m, dev_data_ptr);
    // debug_vector<<<1,1>>>(-12345, sort_data, 1, m);

    checkCudaErrors(cudaMemcpy(indCap, sort_data, m * sizeof(int), cudaMemcpyDeviceToHost));
    

    // 如果所有工作的解都进行了更新
    if(nelem == 0)  
        goto lfeasfix;


    // printf("Before Sort\n");
    /*
    get_sort<<<NumBlocks(m), NUM_THREADS>>>(dv_ptr, m, capres);
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), indCap.begin());
    // printf("After Sort\n");
    */

  // 可加入判断capres的

    // 根据剩余容量升序遍历工厂
    for(int ii=0;ii<m;ii++)
    {  
        int i = indCap[ii];                    // consider first warehouaes with small residual capacity
        //if (i < 0 || i >= m)
        //    cout << "error!" << endl;
        // 由于进行了超限只保留最优的操作，此处if不会发生
        /*
        if(capres[i]<0)                    // should not happen, to be debugged
            continue;
        */
        // 初始化背包
        // printf("????4111\n");
        fixKdynInit<<<NumBlocks(n), NUM_THREADS>>>(q, i, n, req, Ksol, whoIs, val);

        // printf("????111111\n");

        // 通过knapsack求解，参数分别为物品数，背包容量，物品大小，是否在之前被选中，对于该工厂的解
        // 对于val=-1，由于要求得val最大值，故对结果无影响
        //cout << "KdynRecur " << ii << endl;
        nelem -= KDynRecur(1, n, capres+i, q, val, Ksol);
        //if (nelem < 0)
        //    cin.get();
        // printf("NELEM::::%d\n", nelem);
        // 根据knapsack的结果更新解
        fixKdynUpdate<<<NumBlocks(n), NUM_THREADS>>>(Ksol, sol, whoIs, i, n);
        
        // 未知解均求出时，不用继续遍历
        if(nelem == 0)                          // solution complete
            break;
    }
    // 未知解无法求出，总容量直接返回INT_MAX
    if(nelem != 0) goto lendfix;                 // could not recover fesibility

lfeasfix:
    // 得到可行解；若该解更优，则更新
    vectorCopy<int><<<NumBlocks(n), NUM_THREADS>>>(sol, infeasSol, n);
    *zsol = checkSol(sol);
    if(*zsol<zub)
    {  
        vectorCopy<int><<<NumBlocks(n), NUM_THREADS>>>(sol, solbest, n);
        zub = *zsol;
        if(isVerbose) cout << "[fixSol] -------- zub improved! " << zub << endl;
    }

lendfix:
    //cout << "end fixSolViaKnap" << endl;
    
    delete[] indCap;

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

    int* temp = new int[m * n];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            ifs >> temp[ID2D(i, j, n)];
    checkCudaErrors(cudaMemcpy(c, temp, m * n * sizeof(int), cudaMemcpyHostToDevice));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            ifs >> temp[ID2D(i, j, n)];
    checkCudaErrors(cudaMemcpy(req, temp, m * n * sizeof(int), cudaMemcpyHostToDevice));
    delete[] temp;
    temp = new int[m];
    for (int i = 0; i < m; i++)
        ifs >> temp[i];
    checkCudaErrors(cudaMemcpy(cap, temp, m * sizeof(int), cudaMemcpyHostToDevice));
    delete[] temp;

    checkCudaErrors(cudaMalloc((void **)&sol, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&solbest, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&randStates, sizeof(curandState) * n));
    randStatesInit<<<NumBlocks(n), NUM_THREADS>>>(randStates, seed, n);
    checkCudaErrors(cudaMalloc((void **)&temp_req, sizeof(int) * n * m));
    checkCudaErrors(cudaMalloc((void **)&temp_req_node, sizeof(node<int>) * m * n));
    checkCudaErrors(cudaMalloc((void **)&capres, sizeof(int) * m));
    temp_reduction = capres;
    // Kcap归约获得
    int* dev_Kcap = get_temp(Kcap,1);
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
    checkCudaErrors(cudaMalloc((void **)&whoIs, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&q, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&sort_data, sizeof(int) * m));
    checkCudaErrors(cudaMalloc((void **)&sort_keys, sizeof(int) * m));
    checkCudaErrors(cudaMalloc((void **)&temp_res, sizeof(node<int>) * n));
    checkCudaErrors(cudaMalloc((void **)&temp_subgrad, sizeof(int) * n));
    checkCudaErrors(cudaMalloc((void **)&capleft, sizeof(int) * m));
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
