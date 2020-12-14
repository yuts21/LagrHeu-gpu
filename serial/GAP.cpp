#include "GAP.h"
#include <fstream>

GeneralizedAssignemnt::GeneralizedAssignemnt()
{
   //ctor
   EPS = 0.0001;
}

GeneralizedAssignemnt::~GeneralizedAssignemnt()
{  int i;
   //dtor
   if (req!=NULL)
   {  for (i = 0; i<m; i++) free(req[i]);
      free(req);
   }
   if (c!=NULL)
   {  for (i = 0; i<m; i++) free(c[i]);
      free(c);
   }
   if (cap!=NULL)
      free(cap);
}

/***********************************************
 * 初始化工厂已用容量      【O(m)->O(1)】
 * 判断得到总花费          【O(n)->O(log_n)】
 * 判断资源超限:           【O(n)->O(log_m)】
 *    申请矩阵空间 
 *    将解向量扩展为解矩阵  【O(1)】
 *    归约求和             【O(log_m)】
 *    (需先将解化为矩阵，右乘全1矩阵（可使用cudnn）/直接归约，以得到各个工厂的资源情况)
 *    (解矩阵越稠密，即工厂数相对于工作数越小，优化越大)
 *    (也可以直接由线程0进行操作)
 *    结果对比资源总量      【O(1)】(使用syn_or/and)
 * 
 * 【O(max(m,n))->O(max(log_m,log_n))】
***********************************************/
// 得到当前解的总花费
// controllo ammissibilità soluzione
int GeneralizedAssignemnt::checkSol(int* sol)
{  int cost=0;
   int i,j;
   /**************** 申请gpu空间 *******************************************************************/
   int* capused = new int[m];
   /***************** 1 个block，m个thread，无需共享内存 ********************************************/
   for(i=0;i<m;i++) capused[i] = 0;

   /***************** 1个block，n个thread，需要共享内存n*int；归约求和 ******************************/
   // controllo assegnamenti
   for(j=0;j<n;j++)
      if(sol[j]<0 || sol[j]>=m)
      {  cost = INT_MAX;
         goto lend;
      }
      else
         cost += c[sol[j]][j];

   /***************** 1个block，n个thread，无需共享内存；对每个解对应的矩阵位置进行赋值 ****************/
   /***************** m个block，n个thread，需要共享内存n*int；归约求资源和 ****************************/
   /***************** 1个block，m个thread，无需共享内存；对比资源总量，以判断是否超限 ******************/
   /*** 操作3可直接由线程0进行 ****/
   // controllo capacità
   for(j=0;j<n;j++)
   {  capused[sol[j]] += req[sol[j]][j];
      if(capused[sol[j]] > cap[sol[j]])
      {  cost = INT_MAX;
         goto lend;
      }
   }
   delete capused;
lend:    
   return cost;
}

/***********************************************
 * 初始化            【O(max(m,n))->O(1)】
 * 删超限解，更新容量 【O(n)】
 *    (暂无法并行)
 * 计算新解          【n】*【O(m)->O(log_m)】
 *    (由于每个结果影响后面的，对于工作，无法并行)
 * 更新全局解        【O(n)->O(1)】
 * 
 * 【O(n*m)->O(n*log_m)】
***********************************************/
// 贪心修正可行解，按顺序删除超限的解，目标函数值为当前下界、
// 参数1：旧解->新解 参数2：总花费
// recovers feasibility in case of partial or overassigned solution
int GeneralizedAssignemnt::fixSol(int* infeasSol, int* zsol)
{  int i,j,imin=-1;
   int minreq;
   /**************** 申请gpu空间 *******************************************************************/
   vector<int> capres, sol;

   // 初始化限制与旧解
   /***************** 1个block，m个thread，无需共享内存；初始化赋值 **********************************/
   for(i=0;i<m;i++) capres.push_back(cap[i]);
   /***************** 1个block，n个thread，无需共享内存；初始化赋值 **********************************/
   for(i=0;i<n;i++) sol.push_back(infeasSol[i]);

   /***************** 由于有序，暂无法进行并行计算 ****************************************************/
   // 重新计算剩余容量。如果容量不足，则将此处解设置为-1
   // ricalcolo capacità residue. Se sovrassegnato, metto a sol a -1
   for(j=0;j<n;j++)
      if(sol[j]>=0 && (capres[sol[j]] >= req[sol[j]][j]))
         capres[sol[j]] -= req[sol[j]][j];
      else
         sol[j] = -1;

   // 当前总花费
   *zsol = 0;
   for(j=0;j<n;j++)
   {  
      // 不改变未超限的解
      if(sol[j]>=0)              // correct, do nothing
      {  *zsol += c[sol[j]][j];
         continue;
      }

      // 对于i=-1，即超过工厂容量的工作解，获得新解
      // reassign i -1
      minreq = INT_MAX;
      imin = -1;
      /***************** 1个block，m个thread，需要共享内存m*int；找到可以容纳的工厂 **************************/
      // 遍历工厂，找到最后一个可以容纳的工厂
      for(i=0;i<m;i++)
         if(capres[i]>=req[i][j] && req[i][j] < minreq)
         {  minreq = req[i][j];
            imin    = i;
         }

      // 无解：设总花费为INT_MAX，并返回
      if(imin<0)
      {  *zsol = INT_MAX;
         goto lend;           // could not recover feasibility
      }

      // 有解：更新解
      sol[j]=imin;
      capres[imin] -= req[imin][j];
      *zsol += c[imin][j];
   }

   // 如果更新后的解小于上限（最优解），更新并输出信息
   if(*zsol<zub)
   {  
      /***************** 1个block，n个thread，无需共享内存 **************************************************/
      for(i=0;i<n;i++) solbest[i]=sol[i];
      zub = zub = *zsol;
      if(isVerbose) cout << "[fixSol] -------- zub improved! " << zub << endl;
   }
   /***************** 1个block，n个thread，无需共享内存 **************************************************/
   for(i=0;i<n;i++) infeasSol[i]=sol[i];

lend:
   return *zsol;
}

/***********************************************
 * 初始化，只留最优解：
 *    初始化工厂剩余容量       【O(m)->O(1)】
 *    初始化解并更新工厂容量   【O(n)->O(log_n)】
 *    获得每个工作的最优解     【O(n*m)->O(log_m)】
 *    删非最优解，更新容量     【O(n)->O(log_n)】
 * 将工厂按照剩余容量排序       【O(n*logn)->O(kn/nlogn/n)】
 * 排序后遍历工厂，背包求解【m】：
 * （对于上次计算结果，下一次不用再进行背包，故循环无法并行）
 *    初始化背包dp数组         【O(m)->O(1)】
 *    背包求解                【O(n*Kcap)->O(max(n,log_Kcap))】
 *    更新解并计数             【O(n)->O(logn)】
 * 更新全局解                  【O(n)->O(1)】
 * 
 * 【O(m*n*Kcap)->O(m*max(n,log_Kcap))】
***********************************************/
// 只保留最优解，根据剩余容量，通过背包，得到可行解
// recovers feasibility via knapsacks on residual capacities
int GeneralizedAssignemnt::fixSolViaKnap(int* infeasSol, int* zsol)
{  int i,j,imin=-1,minreq;
   int *sol, *minReqFacility;
   // 未分配和已分配的·工作数目
   int nelem,nreset;                    // number of clients to reallocate and reallocated
   /**************** 申请gpu空间 *******************************************************************/
   vector<int> capres,indCap;  
   // 比较两个工厂的剩余容量 
   auto compCap = [&capres](int a, int b){ return capres[a] < capres[b]; };  // ompare, ASC order

   // 初始化总花费、工厂容量
   *zsol = INT_MAX;
   /***************** 1个block，m个thread，无需共享内存；初始化赋值 **********************************/
   for(i=0;i<m;i++) capres.push_back( cap[i] );

   /**************** 申请gpu空间 *******************************************************************/
   // 初始化解（不可行），若旧解为空则设为随机
   sol = (int*) malloc(n * sizeof(int));
   /***************** 1个block，n个thread，无需共享内存；初始化赋值,并放入解矩阵 ****************/
   /***************** m个block，n个thread，需要共享内存n*int；归约求资源和 ****************************/
   for(j=0;j<n;j++)     // randomly fill unassigned clients
   {  sol[j]=infeasSol[j];
      if(sol[j] == NULL || sol[j]<0 || sol[j] >= m)
         sol[j] = rand()%m;
      // residual capacities
      capres[sol[j]] -= req[sol[j]][j];
   }

   // 找到每个工作对应最小花费的工厂
   // finds the least requiring assignment for each client
   nelem = 0;
   /**************** 申请gpu空间 *******************************************************************/
   minReqFacility = (int*) malloc(n * sizeof(int));   // best server for each client
   vector<int> whoIs;

   for(j=0;j<n;j++)
   {  minreq = INT_MAX;
      /***************** n个block，m个thread，需要共享内存m*int；归约求最小值，得每个工作对应的最优工厂 ****/
      for(i=0;i<m;i++) 
         if(req[i][j] < minreq)
         {  minReqFacility[j]=i;
            minreq = req[i][j];
         }

      /**************** 申请gpu空间 ********************************************************************/
      /*********** 1个block，n个thread，无需共享内存；将删去的解放入矩阵中 ********************************/
      /*********** m个block，n个thread，需要共享内存n*int；归约累加删去的数目和容量，容量加到工厂剩余容量 ***/
      /*********** 不使用whoIs数组，而转用sol判断 *******************************************************/
      // 若工厂超限，则只保留旧解中最优的（对应最小花费的工厂），直到容量降低
      // 将被删除的解放入whoIs数组中，总删除数目为nelem，在knapsack中进行更新
      // if j assigned to an overloaded facility, and not the least req one, it must be reassigned
      if(capres[sol[j]] < 0 && sol[j] != minReqFacility[j])
      {  capres[sol[j]] += req[sol[j]][j];
         sol[j] = -1;
         nelem++;
         whoIs.push_back(j);
      }
   }

   /**************** 申请gpu空间，一维 *****************************************************************/
   int*    q   = new int[nelem];     // knapsack requests
   double* val = new double[nelem];  // knapsack profits
   int*    Ksol= new int[nelem];     // knapsack solution

   // 如果所有工作的解都进行了更新
   if(nelem == 0)    // if we got a feasible solution by pure chance, done
      goto lfeas;

   // 将序号按照剩余容量进行排序
   // order facilities by increasing residual capacity
   /*********** 1个block，m个thread，无需共享内存；初始化赋值 *******************************************/
   for(i=0;i<m;i++) indCap.push_back(i);
   /********************* 可使用cuda的Thrust库，进行排序 ***********************************************/
   std::sort(indCap.begin(), indCap.end(), compCap);

   /*********** 1个block，n个thread，无需共享内存；初始化赋值 *******************************************/
   for(j=0;j<nelem;j++) val[j] = 1; // any element could be chosen (val > 0)
   nreset = 0;

   // 根据剩余容量升序遍历工厂
   for(int ii=0;ii<m;ii++)
   {  i = indCap[ii];               // consider first warehouaes with small residual capacity
      // 由于进行了超限只保留最优的操作，此处if不会发生
      if(capres[i]<0)               // should not happen, to be debugged
         continue;

      /*********** 1个block，n个thread，无需共享内存；初始化赋值 *******************************************/
      // 遍历待求解的工作，q中存工作对于该工厂的花费
      for (j = 0; j < nelem; j++)
      {  q[j]    = req[i][whoIs[j]];         // requests to the i-th wh by the j-th elem to reassign
         Ksol[j] = 0;
      }

      /******************************** 改为核函数 *******************************************************/
      // 通过knapsack求解，参数分别为物品数，背包容量，物品大小，是否在之前被选中，对于该工厂的解
      // 对于val=-1，由于要求得val最大值，故对结果无影响
      KDynRecur(nelem,capres[i],q,val,Ksol); // solve the knapsack
      /*********** 1个block，n个thread，需要共享内存n*int；更新解并计算更新数目 *****************************/
      // 遍历待求解的工作，根据knapsack的结果更新解
      for(j=0;j<nelem;j++)
         if(Ksol[j] > 0)
         {  sol[whoIs[j]] = i;
            val[j] = -1;                     // won't be chosen again
            nreset++;
         }
      // 未知解均求出时，不用继续遍历
      if(nreset == nelem)                    // solution complete
         break;
   }
   // 未知解无法求出，总容量直接返回INT_MAX
   if(nreset < nelem) goto lend;             // could not recover fesibility

lfeas:
   // 得到可行解；若该解更优，则更新
   for(i=0;i<n;i++) infeasSol[i]=sol[i];
   /****************** 化为核函数 ***************************/
   *zsol = checkSol(sol);
   if(*zsol<zub)
   {  for(i=0;i<n;i++) solbest[i]=sol[i];
      zub = *zsol;
      if(isVerbose) cout << "[fixSol] -------- zub improved! " << zub;
   }

lend:
   if(q!=NULL) delete(q);
   if(val!=NULL) delete(val);
   if(Ksol!=NULL) delete(Ksol);
   free(sol);
   free(minReqFacility);
   return *zsol;
}


// reads instance data from json formatted files
void GeneralizedAssignemnt::readData(string fileName)
{
   ifstream ifs(fileName);
   if (! ifs.is_open()) {
      cout << "Error opening file" << endl;
      return;
   }

   ifs >> m >> n;
   // cout << m << " " << n << endl;

   c = (int**)malloc(m * sizeof(int *));
   for (int i = 0; i < m; i++) {
      c[i] = (int*)malloc(n * sizeof(int));
      for (int j = 0; j < n; j++)
         ifs >> c[i][j];
   }
   
   req = (int**)malloc(m * sizeof(int *));
   for (int i = 0; i < m; i++) {
      req[i] = (int*)malloc(n*sizeof(int));
      for (int j = 0; j < n; j++)
         ifs >> req[i][j];
   }
   
   cap = (int*)malloc(m*sizeof(int));
   for (int i = 0; i < m; i++)
      ifs >> cap[i];

   sol = new int[n];
   solbest = new int[n];
   zub = INT_MAX;

   if(isVerbose) cout << "data read, n="<< n << " m="<< m << endl;;
}

// **************************************************************************** //
// *************************** Free functions ********************************* //

// computes assignment regrets for each client
void computeRegrets(int** c, int n, int m, vector<int> & regrets)
{  int i,j,first,second;

   for(j=0;j<n;j++)
   {  first = second = INT_MAX;
      for(i=0;i<m;i++)
      {  if(c[i][j] < first)
         {  second = first;
            first  = c[i][j];
         }
         else if (c[i][j] < second)
            second  = c[i][j];
      }
      regrets[j] = second - first;
   }
}

// 使用knapsack背包，n:物品数目，Kcap:背包容量，Q:各物品大小，val:花费（-拉格朗日乘子）取反，Ksol:knapsack结果 
// dynamic programming recursion for the knapsack
/****************************************
 * (为简化问题，此处假设背包容量在 threads pre block 范围内)
 * (由于一次背包为对一个工厂求解，设每个block为1个工厂，每种背包容量为一个thread)
 * 依次遍历每个物品dp 【n】*【O(Kcap)->O(2)】
 *    (由于更新时的冲突，每次dp整体更新都需要同步，共同步2*n次，一次读，一次写)
 * 得到结果          【O(Kcap)->O(log_Kcap)】
 *    (归约找到最大的值，即最小的总容量)
 * 递归decode操作    【<O(n)】
 *    （无并行性，仅由线程0进行）
 * max(2*n,log_Kcap,n) / max(dp，最优花费，倒推最优解)  
 * 
 * 【O(m*n*Kcap)->O(max(n,log_Kcap))】
******************************************/
double KDynRecur(int n, int Kcap, int* Q, double* val, int* Ksol)
{  int q,i,imax;
   double res=0;

   /************************** 申请gpu空间 *******************************************************/
   // 数组f：已用容量为q、遍历到第i个物品时的最小花费
   double** f = (double**) calloc((Kcap+1), sizeof(double*)); // init to 0
   for(i=0;i<Kcap+1;i++)
      f[i] = (double*) calloc(n, sizeof(double));             // init to 0
   /************************** m个block，Kcap个thread，无需共享内存 *******************************/
   for (i = 0; i < n; i++)
   {
      // 此处q为已用容量
      //if (val[i] < 0) continue;
      for (q = 0; q <= Kcap; q++)
         switch (i)
         {
            case 0:
               if (q >= Q[i])
                  f[q][i] = max(0.0, val[i]);
               else
                  f[q][i] = 0;
               break;
            default:
               if (q >= Q[i])
                  // dp更新方程
                  f[q][i] = max(f[q][i - 1], f[q - Q[i]][i - 1] + val[i]);
               else
                  f[q][i] = f[q][i - 1];
               break;
         }
   }

   /************ m个block，Kcap个thread，需要共享内存Kcap*double；归约求最大值（最小的总容量）*************/
   /****** 可与上一步合并 *******************************/
   imax=0;
   for(i=0;i<n;i++)
      // ？？？为啥是Kcap，而不是计算过程中得res，物品和不一定刚好等于Kcap吧...
      // 应该是写反了，应该是对于遍历到第n个物品时的每个容量进行遍历，得到最优解
      if(f[Kcap][i] > res)
      {  res = f[Kcap][i];
         imax = i;
      }

   // 根据res反推路径，仅由0号线程进行
   KdecodeSol(imax,Kcap,Q,val,n,f,Ksol);

   // deallocations
   if (f!=NULL)
   {  for (i = 0; i<Kcap+1; i++) 
         free(f[i]);
      free(f);
   }

   return res;
}

// decode背包问题，得到完整路径
// Decodes a knapsack DP recursion, given the recursion matrix f
void KdecodeSol(int i, int Kcap, int* Q, double* val, int n, double** f, int* Ksol)
{  int q=Kcap;
   double eps = 0.0001;

   // 初始从0->n-1遍历物品，故decode时从n-1->0
   while(q>0 && i>0)
   {  
      // 如果不选该物品仍可以，则不选
      if(abs(f[q][i-1] - f[q][i]) < eps)
      {  i--;
         continue;
      }

      // 需要选择该物品
      if(abs(f[q-Q[i]][i-1] - (f[q][i]-val[i])) < eps)
      {  q -= Q[i];
         Ksol[i] = 1;
         i--;
         continue;
      }

      // 错误
      cout << "[KP decodeSol] generic error" << endl;
      goto lend;
   }

   // 起始物品
   if(i==0 && q>0)
      if(f[q][i] == val[i])
         Ksol[i] = 1;
      else
         Ksol[i] = 0;

lend:
   //checkSol();
   return;
}

// print a 1D array of doubles contents
void printDblArray(double* a, int n)
{  int i;
   cout.precision(3);
   for(i=0;i<n;i++)
      cout << a[i] << " ";
   cout << endl;
}