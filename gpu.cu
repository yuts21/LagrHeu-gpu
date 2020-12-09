#include "gpu.cuh"

// reduction操作所使用的数组空间，其中以最大的T类型(node<double>)为准，而空间大小由max(n*m, m*Kcap)决定
void* temp_reduction;

// 使用归约方法计算，op为0时求最小值。op为1时求最大值，op为2时求和
// ans使用指针传入
// 每隔m个block计算同一个数组的归约，最终将结果分别归约在前m个block中
template<typename T> __global__ void vectorReduction(T* V, int m, int n, int bnum, T* ans, int op)
{
    SharedMemory<T> DATA;
    T* shared = DATA.getPointer();
    
    // tid：线程号  bid：块号   tnum:该block中计算的thread数目
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    // 该块所计算为第k个工厂的第i块
    int k = bid%m, i = bid/m;
    // 该块所计算的任务数
    int  tnum = n/bnum + ((i < n%bnum) ? 1 : 0);
    
    // 拷贝输入数组
    if(tid < tnum)
        copy(&shared[tid], V[tid*bnum*m + i*m + k]);
    
    // 归约方法，每次处理对应位置的两个数;其中当d为奇数时，不处理中间的数，其余的两两比较归到左边
    for (int d = tnum; d > 1; d = (d+1)>>1)
    {
        __syncthreads();
        if(tid < d/2)
        {
            switch(op){
                case 0: copy(&shared[tid],  minn(shared[tid], shared[tid+(d>>1)+(d&1)]));  break;
                case 1: copy(&shared[tid],  maxx(shared[tid], shared[tid+(d>>1)+(d&1)]));  break;
                case 2:
                    if(get_data(shared[tid]) >= INT_MAX)
                        break;
                    else if(get_data(shared[tid+(d>>1)+(d&1)]) >= INT_MAX) 
                        copy(&shared[tid], shared[tid+(d>>1)+(d&1)]);      
                    else
                        copy(&shared[tid], shared[tid] + shared[tid+(d>>1)+(d&1)]);      
                    break;
                default: 
                    //cout << "[vectorReduction]:Error op" << endl;  
                    break;
            }
        }
    }

    // 从共享内存中写回结果，只有一个线程执行该操作
    if (tid == 0) 
    {
        copy(&ans[i*m + k], shared[0]);
    }
}

template __global__ void vectorReduction<int>(int* V, int m, int n, int bnum,int* ans, int op);
template __global__ void vectorReduction<double>(double* V, int m, int n, int bnum,double* ans, int op);

template __global__ void vectorReduction<node<int>>(node<int>* V, int m, int n, int bnum,node<int>* ans, int op);
template __global__ void vectorReduction<node<double>>(node<double>* V, int m, int n, int bnum,node<double>* ans, int op);


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

template<typename T> void doReduction(T* V, int m, int n, T* ans, int op)
{
    // printf("doReduction\n");
    //printf("%d\n", sizeof(T));
    T* temp = (T*)temp_reduction;
    vectorCopy<<<NumBlocks(n*m), NUM_THREADS>>>(V, temp, n*m);

    while(n > 1){
        // 每个工厂所需的block数
        /*
        if(m == 1)
            debug_vector<<<1,1>>>(1,temp,m,n);
        else
            debug_vector<<<1,1>>>(1,temp,n,m);
        */
        int bnum = NumBlocks(n);
        vectorReduction<<<m * bnum , NUM_THREADS, sizeof(T)*(n/bnum + 1)>>>(temp, m, n, bnum, temp, op);
        n = bnum;
    }
    /*
    if(m == 1)
        debug_vector<<<1,1>>>(2,temp,m,n);
    else
        debug_vector<<<1,1>>>(2,temp,n,m);
    if(m == 1)
        debug_vector<<<1,1>>>(3,ans,m,n);
    else
        debug_vector<<<1,1>>>(3,ans,n,m);
    printf("%d %d\n", m, n);
    */
    vectorCopy<<<NumBlocks(m), NUM_THREADS>>>(temp, ans, m);
    // printf("%d %d\n", m, n);
    /*
    if(m == 1)
        debug_vector<<<1,1>>>(4,temp,m,n);
    else
        debug_vector<<<1,1>>>(4,temp,n,m);
    if(m == 1)
        debug_vector<<<1,1>>>(5,ans,m,n);
    else
        debug_vector<<<1,1>>>(5,ans,n,m);
    */
    // printf("yes\n");
}

template void doReduction<int>(int* V, int m, int n, int* ans, int op);
template void doReduction<double>(double* V, int m, int n, double* ans, int op);
template void doReduction<node<int>>(node<int>* V, int m, int n, node<int>* ans, int op);
template void doReduction<node<double>>(node<double>* V, int m, int n, node<double>* ans, int op);


// 将线程id转换为2维数组id
__device__ void convertIDto2D(int bid, int tid, int n, int& x, int& y)
{
    int s = bid*NUM_THREADS + tid;
    x = s/n;
    y = s%n;
} 


// 将线程id转换为3维数组id

__device__ void convertIDto3D(int bid, int tid, int m, int n, int& x, int& y, int& z)
{
    int s = bid*NUM_THREADS + tid;
    x = s/(m*n);
    y = s%m/n;
    z = s%n;
}

// 此处为方便reduction，req[i][j]，i为工作，j为工厂
__global__ void vectorToMatrixReq(int* sol, int* temp_req, int* req, int n,  int m)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n){
        //if (sol[index] < 0 || sol[index] >= m)
        //    printf("error!\n");
        int id = ID2D(index,sol[index], m);
        int id2 = ID2D(sol[index], index, n);
        temp_req[id] = req[id2];
    }
}

__global__ void randStatesInit(curandState* state, unsigned long seed, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
        curand_init(seed, index, 0, &state[index]);
}

// 初始工作的解均为-1，次梯度均为1（后续-决策变量，最终结果为该工作可以被分配为多少个工厂,取反）
__global__ void subInit(int* subgrad, int* lbsol, int n)
{
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n) {
        subgrad[index] = 1;
        lbsol[index] = -1;
    }
}


// 初始工作的解均为-1，次梯度均为1（后续-决策变量，最终结果为该工作可以被分配为多少个工厂,取反）
__global__ void checkInit(int* temp_cost, int* sol, int* c, int n, int m)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n){
        if(sol[index]<0 || sol[index]>=m)
            temp_cost[index] = INT_MAX;
        else
            temp_cost[index] = c[ID2D(sol[index], index, n)];
    }
}

__global__ void fixsolInit(int* sol, int* infeasSol, int* temp_req, int* req, int n, int m, curandState* state)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
    {  
        sol[index]=infeasSol[index];
        if(sol[index]<0 || sol[index] >= m) {
            sol[index] = curand(&state[index]) % m;
        }
        
        // residual capacities
        int id = ID2D(index,sol[index], m);
        int id2 = ID2D(sol[index], index, n);
        temp_req[id] = req[id2];
    }
}


__global__ void fixsolMain(node<int>* min_req, int* capres, int* sol, int* whoIs, int* req, int* cap, int* nelem, int n)
{
    const int tid = threadIdx.x;
    if(tid == 0){
        for(int j=0;j<n;j++)
        {  
            int i = sol[j];
            //if (i < 0 || i >= 20)
            //    printf("error11111111!\n");
            if(i == -1) continue;
            int index2D = ID2D(i, j, n);
            // printf("%d %d %d %d %d %d\n", i, j, cap[i], req[index2D], capres[i], min_req[j].pos);
            // 若工厂超限，则只保留旧解中最优的（对应最小花费的工厂），直到容量降低
            // 将被删除的解放入whoIs数组中，总删除数目为nelem，在knapsack中进行更新
            // if j assigned to an overloaded facility, and not the least req one, it must be reassigned
            if(cap[i] < req[index2D] || (capres[i] < 0 && i != min_req[j].pos))
            {  
                capres[i] += req[index2D];
                sol[j] = -1;
                whoIs[j] = 1;
                (*nelem)++;
            }
        }
        for(int j=0;j<n;j++)
        {  
            int i = sol[j];
            if(i == -1) continue;
            int index2D = ID2D(i, j, n);
            // 在第一遍后capres仍小于0，说明最优解也会超限，则从前往后删除
            if(capres[i] < 0)
            {  
                capres[i] += req[index2D];
                sol[j] = -1;
                whoIs[j] = 1;
                (*nelem)++;
            }
        }
        //for (int i = 0; i < 20; i++)
        //    if (capres[i]<0)
        //        printf("error1111!\n");
    }
}


__global__ void fixKdynInit(int* q, int i, int n, int* req, int* Ksol, int* whoIs, double* val)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n)
    {  
        q[index]    = req[i * n + index];         // requests to the i-th wh by the j-th elem to reassign
        Ksol[index] = 0;
        val[index] = (whoIs[index] == 0 ? -1 : 1); 
    }
}

__global__ void fixKdynUpdate(int* Ksol, int* sol, int* whoIs, int i, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int index = blockDim.x * bid + tid;
    if(index < n && Ksol[index] > 0)
    {  
        sol[index] = i;
        whoIs[index] = 0;           
    }
}


// 数组初始化为0
__global__ void kdynInit(double* val, int* ksol, int* c, double* lambda, int m, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i,j;
    convertIDto2D(bid, tid, n, i, j);
    int index = i * n + j;

    if(index < m*n){
        // 对每个工作的“花费-拉格朗日乘子”取反，以使用背包求得其最小值（即取反后的最大值）
        val[index]  = -(double)c[index]+lambda[j];  // inverted sign to make it minimzation
        // 初始解均为0
        ksol[index] = 0;
    }
}




__global__ void kdynDP(double* f, int m, int Kcap, int n, int* req, double* val, int* cap, int i)
{
    // tid：线程号  bid：块号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    // 遍历到第i个工作,对于线程所执行的第q容量，对于第k个工厂
    int q, k;
    convertIDto2D(bid, tid, m, q, k);
    // （i,q,k）在一维数组中的位置
    int index3D = i*Kcap*m + q*m + k;
    // (k,i)在一维数组中的位置
    int index2D = k*n + i;

    //printf("%d %d %d 3D:%d 2D:%d        cap:\n",i,q,k,index3D, index2D);
    if(k < m && q <= cap[k]){
        //if (index3D < 0 || index3D >= n*Kcap*m) {
        //    printf("error!\n");
        //}
        //if (index2D < 0 || index2D >= m * n) {
        //    printf("error!\n");
        //}
        if (i == 0) {
            if(q >= req[index2D] && val[index2D] > 0.0){
                // printf("yesyes\n");
                f[index3D] = val[index2D];
            }
        }
        else {
            // (i,q,k) = q<req[k][i] ? : max( (i-1,q,k), (i-1,q-req[k][i],k) )
            //if (index3D - Kcap * m < 0 || index3D - Kcap * m >= n*Kcap*m)
            //    printf("error!\n");
            if(q<req[index2D])
                f[index3D] = f[index3D - Kcap*m];
            else {
                //if (index3D - Kcap * m - req[index2D] * m < 0 || index3D - Kcap * m - req[index2D] * m >= n*Kcap*m)
                //    printf("error!\n");
                if(f[index3D - Kcap*m] > f[index3D - Kcap*m - req[index2D]*m] + val[index2D])
                f[index3D] = f[index3D - Kcap*m];
            else
                f[index3D] = f[index3D - Kcap*m - req[index2D]*m] + val[index2D];
            }
        }
    }
}


// decode背包问题，得到完整路径
__global__ void kdynDecode(int m, int Kcap, node<double>* imax, int* req, double* val, int n, double* f, int* Ksol)
{  
    // tid：线程号  bid：块号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int k = blockDim.x * bid + tid;
    if(k < m){
        double eps = 0.0001;
        int q = imax[k].pos;    // 最优解所占的容量
        int i = n - 1;          // 当前的工作i
        int index3D;            // （i,q,k）在一维数组中的位置
        int index2D;            // (k,i)在一维数组中的位置
        // 初始从0->n-1遍历物品，故decode时从n-1->0
        while(q>0 && i>0)
        {  
            index3D = i*Kcap*m + q*m + k;
            index2D = k*n + i;
            // 如果不选该物品仍可以，则不选
            // (i-1, q, k) - (i, q, k)
            if(abs(f[index3D - Kcap*m] - f[index3D]) < eps)
            {  
                i--;
                continue;
            }

            // 需要选择该物品
            // (i-1, q-req[k][i], k) - ((i, q, k) - val[k][i])
            if(abs(f[index3D - Kcap*m - req[index2D]*m] - (f[index3D]-val[index2D])) < eps)
            {  
                q -= req[index2D];
                Ksol[index2D] = 1;
                i--;
                continue;
            }

            // 错误
            // cout << "[KP decodeSol] generic error" << endl;
            return;
        }

        // 起始物品
        if(i==0 && q>0){
            index3D = q*m + k;
            index2D = k*n;
            if(f[index3D] == val[index2D])
                Ksol[index2D] = 1;
            else
                Ksol[index2D] = 0;
        }
    }

    //checkSol();
    return;
}

__global__ void lagrInit(int* subgrad, int* temp, int n)
{
    // tid：线程号  bid：块号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int k = blockDim.x * bid + tid;
    if(k < n){
        temp[k] = subgrad[k]*subgrad[k];
    }
}

__global__ void lagrUpdate(double* lambda, int* subgrad, double step, int n){
    // tid：线程号  bid：块号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int k = blockDim.x * bid + tid;
    if(k < n){
        lambda[k] = max(0.0,lambda[k]+step*subgrad[k]);
    }
}

__global__ void opt10Init(int* temp_c, int* c, int* sol, int n){
    // tid：线程号  bid：块号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int k = blockDim.x * bid + tid;
    if(k < n){
        //if (sol[k] < 0 || sol[k] >= 20)
        //    printf("error!\n");
        temp_c[k] = c[ID2D(sol[k], k, n)];
    }
}

__global__ void opt10Main(int* c, int* capleft, int* req, int* sol, int z, int m, int n,node<int>* ans){
    // tid：线程号  bid：块号o
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i, j;
    convertIDto2D(bid, tid, n, i, j);
    int isol = sol[j];
    //if (isol < 0 || isol >= m)
    //    printf("error!\n");
    int id = i*n+j;
    int id_old = isol*n + j;

    if(id < m * n){
        ans[id].pos = id;

        if (i == isol) {
            ans[id].data = INT_MAX;
        }
        else if (c[id] < c[id_old] && capleft[i] >= req[id])
        {  // remove from isol and assign to i
            z -= (c[id_old] - c[id]);
            ans[id].data = z;
        }
        else{
            ans[id].data = INT_MAX;
        }
    }
}

__global__ void opt10Update(int* sol, int* capleft, int* req, int i, int j, int n){
    // tid：线程号  bid：块号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int k = blockDim.x * bid + tid;
    if(k == 0){
        int isol = sol[j];    
        //if (isol < 0  || isol >= 20)
        //    printf("error!\n");
        sol[j] = i;
        capleft[i]    -= req[i*n + j];
        capleft[isol] += req[isol*n + j];
    }
}
