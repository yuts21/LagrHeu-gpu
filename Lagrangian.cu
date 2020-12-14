#include "Lagrangian.cuh"
#include "gpu.cuh"
#include "LocalSearch.cuh"
#include "helper.cuh"


Lagrangian::Lagrangian(GeneralizedAssignemnt* GAPinstance, int & zz) : zub(zz)
{
   //ctor
   GAP = GAPinstance;
   m = GAP->m;
   n = GAP->n;
   sol = GAP->sol;
   solbest = GAP->solbest;
   req = GAP->req;
   // printf("ZUB:%d\n", zub);
}

Lagrangian::~Lagrangian()
{
   //dtor
}


// just logging on file flog
/*void Lagrangian::writeIterData(ofstream& flog, int iter, double zlb, int zub, int zcurr, double alpha,
                               int* lbsol, int* subgrad, double* lambda, double step)
{  
   int i;

   flog << "iter "<< iter <<" zlb = "<< zlb <<" zub = "<< zub <<" zcurr = "<< zcurr <<" alpha = "<< alpha <<" \nlbsol ";
   for(i=0;i<n;i++)
      flog << " "<<lbsol[i];

   flog << "\nsubgr ";
   for(i=0;i<m;i++)
      flog << " "<<subgrad[i];

   flog << "\nlambda ";
   for(i=0;i<m;i++)
      flog << " "<<lambda[i];

   flog << "\nstep "<<step<< endl;
};*/

/**************************************************
 * （无cpu拷贝，故所有运算均在gpu上进行，不能并行的计算只由线程0执行）
 * 初始化【O(n)->O(1)】
 * 迭代  【maxiter】:
 *    算初始下界及解【O(m*n*Kcap)->O(max(n,log_Kcap,log_m))】
 *    判断可行解    【O(max(m,n))->O(max(log_m,log_n))】
 *    (下述分类讨论)
 *    1) 达到最优解，更新并退出 【O(n)->O(1)】
 *    2) 初始解不可行，背包修正 【m】*【O(n*Kcap)->O(max(n,log_Ncap))】
 *    3) 接近最优解，局部搜索   【#】*【O(n/2*m/2)->O(max(log_n,log_m))】
 *    更新上界          【O(n)->O(1)】
 *    拉格朗日乘子更新   【O(n)->O(log_n)】
 * 
 * 在迭代过程中，第二种情况居多，故以其为准
 * 【maxiter】*【m】*【O(n*Kcap)->O(n)】
 **************************************************/
// 限制总花费，松弛任务分配
// Lagrangian, feasible for the capacities, relaxes assignments
int Lagrangian::lagrCap(int* c, double alpha, double alphastep, double minAlpha, int innerIter, int maxiter)
{  
   int iter = 0;
   int sumSubGrad2,zcurr;
   double step=0,fakeZub;
   double zlb,zlbBest = 0.0;

   double* lambda  = nullptr;   // 拉格朗日乘子，将对每个工作的花费加上权值，初始为0.0，根据次梯度更新
   int*    subgrad = nullptr;   // 次梯度，用于更新拉格朗日乘子，根据每次迭代的初解（背包求得）中每个工作可以加入的工厂数来更新
   int*    lbsol   = nullptr;   // 当前解
   int*    temp    = nullptr;   
   checkCudaErrors(cudaMalloc((void **)&lambda, sizeof(double) * n));
   checkCudaErrors(cudaMalloc((void **)&subgrad, sizeof(int) * n));
   checkCudaErrors(cudaMalloc((void **)&lbsol, sizeof(int) * n));
   checkCudaErrors(cudaMalloc((void **)&temp, sizeof(int) * n));

   // 初始化上下界
   zcurr     = INT_MAX;       // initial upper bound
   zlb       = DBL_MIN;
   //cout << zlb << endl;
   // cin.get();
   // 初始化拉格朗日乘子为0.0
   vectorInit<double><<<NumBlocks(n), NUM_THREADS>>>(lambda, n, 0);


   // 拉格朗日松弛主体
   iter  = 0;
   while(alpha > minAlpha && iter < maxiter)
   {  
      // if(iter == 50) return 0;
      // 算得初始下界及解
      //cout << "subproblem_cap" << endl;
      subproblem_cap(c, &zlb, &zlbBest, zub, lambda, subgrad, lbsol);
      zcurr = GAP->checkSol(lbsol);
      // cout << zcurr << endl;
      // cin.get();
      // 更新上界
      if(zcurr<zub)
      {
         vectorCopy<int><<<NumBlocks(n), NUM_THREADS>>>(lbsol, solbest, n);
         zub = zcurr;
         if(GAP->isVerbose) cout << "[lagrCap] -------- zub improved! " << zub << endl;
      }

      // 当前解约等于下界，即最优解
      if((zub-zlbBest) < 1.0)                       // -------------------------- Optimum found 
      {  
         if(GAP->isVerbose) cout << "[lagrCap] Found the optimum!!! zopt="<< zub << " zlb=" << zlbBest<<endl;
         checkCudaErrors(cudaFree(lambda));
         checkCudaErrors(cudaFree(subgrad));
         checkCudaErrors(cudaFree(lbsol));
         checkCudaErrors(cudaFree(temp));
         return zcurr;
      }
      // 继续更新解，启发式搜索
      else                                       // -------------------------- Heuristic block
      {  
         // 初始状态
         if(zcurr == INT_MAX)
         {
            // printf("222\n");
            // 通过背包更新解
            //cout << "fixSolViaKnap" << endl;
            //zcurr = GAP->fixSolViaKnap(lbsol, &zcurr);   // hope to fix infeasibilities
            zcurr = GAP->fixSol(lbsol, &zcurr);
         }
         // 如果获得的解与下界较接近，则通过局部搜索尝试得到更优解
         if(zcurr < 10*zlb)
         {  
            // printf("333\n");
            // cout << "opt10" << endl;
            // cin.get();
            vectorCopy<int><<<NumBlocks(n), NUM_THREADS>>>(lbsol, sol, n);
            // printf("??????\n");
            //cout << "localsearch" << endl;
            LocalSearch* LS = new LocalSearch(GAP, GAP->zub);
            // printf("!!!!!???\n");
            LS->opt10(c);
            delete LS;        
         }
      }

      // 计算步长   
      //cout << "cal step" << endl;
      lagrInit<<<NumBlocks(n), NUM_THREADS>>>(subgrad, temp, n);
      // debug_vector<<<1,1>>>(111122, subgrad, 1, n);
      int* temp_int = get_temp(sumSubGrad2);
      //printf("8\n");
      doReduction(temp, 1,n,temp_int,2);
      sumSubGrad2 = delete_temp(temp_int);
      // printf("sumSubGrad2 : %d\n", sumSubGrad2);
      fakeZub = min((double) zcurr,1.2*zlb);
      fakeZub = max(fakeZub,zlb+1);
      step = alpha*(fakeZub-zlb)/sumSubGrad2;
      // printf("Alpha : %lf   fakeZub: %lf  zlb : %lf\n", alpha, fakeZub, zlb);
      // printf("Step: %lf\n", step);

      // debug_vector<<<1,1>>>(-111133, lambda, 1, n);
      lagrUpdate<<<NumBlocks(n), NUM_THREADS>>>(lambda, subgrad, step, n);
      // debug_vector<<<1,1>>>(111133, lambda, 1, n);
      iter++;
      if(iter % innerIter == 0)
         alpha = alphastep*alpha;

      if(iter%100 == 0)                           // -------------------------- logging
      {  if(GAP->isVerbose) cout << "[lagrCap] iter="<<iter<<" zub="<<zub<<" zlb="<<zlbBest<<" zcurr="<<zcurr<<endl;
         // if(GAP->isVerbose) writeIterData(flog, iter, zlb, zub, zcurr, alpha, lbsol, subgrad, lambda, step);
      }
   }
   //cout << alpha << " " << iter << endl;

   //if (flog.is_open()) flog.close();
   checkCudaErrors(cudaFree(lambda));
   checkCudaErrors(cudaFree(subgrad));
   checkCudaErrors(cudaFree(lbsol));
   checkCudaErrors(cudaFree(temp));
   return zcurr;
   
}

/**************************************
 * （此处求解的过程，Q、val、Ksol等数组均为循环利用，即计算下一个工厂时并未用到上一个工厂的结果，故可以并行处理）
 * 初始化 【O(n)->O(1+log_n)】
 * 背包dp计算：
 *    （由于不再循环利用数组，Q、Kso不需要重新初始化，故最外层循环可分割为3段）
 *    （由于背包的threads数目与背包容量有关，故无法与上下核函数合并）
 *    初始化dp数组 【O(m*n)->O(1)】
 *    背包         【O(m*n*Kcap)->O(max(n,log_Kcap))】
 *    判断并更新   【O(m*n)->O(log_m)】
 * 
 * 【O(m*n*Kcap)->O(max(n,log_Kcap,log_m))】
*************************************/
// 将工作分配到工厂，将初始拉格朗日乘子作为惩罚项加在工作的花费上，算得初始下界
// fills servers up to their capacities
void Lagrangian::subproblem_cap(int* c, double *zlb, double *zlbBest, int zub, double* lambda, int* subgrad, int* lbsol)
{  
   // 由于对所有工厂并行计算，以空间换取时间，故此处数组为原数组的m倍
   // Q为工厂i对工作j的资源,直接用req取代
   // val为工厂i对工作j减去拉格朗日乘子权值后的花费，再取反以在背包求得最小值
   // Ksol为工厂i背包后求得的解，1为包含工作j、0为不包含
   double* val = GAP->val;
   int* Ksol = GAP->Ksol;
   // 更新下界、次梯度
   node<int>* temp_ksol = GAP->temp_ksol;
   node<int>* temp_res = GAP->temp_res;
   int* temp_subgrad = GAP->temp_subgrad;

   // 初始化次梯度为1，初始解为-1
   subInit<<<NumBlocks(n), NUM_THREADS>>>(subgrad, lbsol, n);
   //printf("Sub %lf\n", *zlb);
   // 对拉格朗日乘子求和，作为花费下界
   double* temp = get_temp(*zlb);
   //printf("9\n");
   doReduction(lambda, 1, n, temp, 2);
   *zlb = delete_temp(temp);
   //printf("Sub2 %lf\n", *zlb);
   // cout << "ans" << " " << zlb << endl;
   // 对Q、val、ksol进行初始化
   kdynInit<<<NumBlocks(m*n), NUM_THREADS>>>(val, Ksol, c, lambda, m, n);

   // 对于每个工厂均求一遍吸收惩罚项后的背包
   // 对于m个工厂并行，对于每个物品的dp过程不能并行
   // zlb减去每个工厂的总花费(有重复解)
   *zlb -= GAP->KDynRecur(m, n, GAP->cap, req, val, Ksol);
   //printf("Sub3 %lf\n", *zlb);
   // 启发式：对每个工作更新其状态（最终解为最后一个工厂），使用归约最大值得到
   vectorToNode<<<NumBlocks(m*n), NUM_THREADS>>>(Ksol, temp_ksol, m*n, n);
   
   //printf("10\n");
   doReduction(temp_ksol, n, m, temp_res, 1);
   nodeToPos<<<NumBlocks(n), NUM_THREADS>>>(temp_res, lbsol, n);
   // 如果可在该工厂执行，则次梯度-1，使用归约得到结果
   //printf("11\n");
   doReduction(temp_ksol, n, m, temp_res, 2);
   nodeToData<<<NumBlocks(n), NUM_THREADS>>>(temp_res, temp_subgrad, n);
   vectorSub<<<NumBlocks(n), NUM_THREADS>>>(subgrad, temp_subgrad, subgrad, n);

   // 下界大于上界，返回错误
   //printf("ZLB: %lf\n", *zlb);
   //cin.get();
   assert(*zlb<=zub);  // aborts if lb > ub
   if(*zlb>*zlbBest) 
      *zlbBest = *zlb;

   return;
}
