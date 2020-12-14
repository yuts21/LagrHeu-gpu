#include "Lagrangian.h"
#include "LocalSearch.h"
#include <ctime>

Lagrangian::Lagrangian(GeneralizedAssignemnt* GAPinstance, int & zz) : zub(zz)
{
   //ctor
   GAP = GAPinstance;
   m = GAP->m;
   n = GAP->n;
   sol = GAP->sol;
   solbest = GAP->solbest;
   req = GAP->req;
}

Lagrangian::~Lagrangian()
{
   //dtor
}

/**************************************************
 * （无cpu拷贝，故所有运算均在gpu上进行，不能并行的计算只由线程0执行）
 * 初始化【O(m)->O(1)】
 * 迭代  【maxiter】:
 *    算初始下界及解【O(m*n)->O(max(log_n,log_m))】
 *    判断可行解    【O(max(m,n))->O(max(log_m,log_n))】
 *    (下述分类讨论)
 *    1) 达到最优解，更新并退出 【O(n)->O(1)】
 *    2) 初始解不可行，贪心修正 【n】*【O(m)->O(log_m)】
 *    3) 接近最优解，局部搜索   【#】*【O(n/2*m/2)->O(max(log_n,log_m))】
 *    更新上界          【O(n)->O(1)】
 *    拉格朗日乘子更新   【O(m)->O(log_m)】
 * 
 * 在迭代过程中，第二种情况居多，故以其为准
 * 【maxiter】*【n】*【O(m)->O(log_m)】
 **************************************************/
// 限制任务分配，松弛总花费
// Lagrangian, feasible for the assignments, relaxes capacities
int Lagrangian::lagrAss(int** c, double alpha, double alphastep, double minAlpha, int innerIter, int maxiter)
{  int i,sumSubGrad2,iter=0,zcurr;
   double  zlb,step=0,zlbBest,fakeZub;

   /**************** 直接申请gpu空间，不进行cpu内存拷贝 ****************************************************/
   double* lambda  = new double[m];
   int*    subgrad = new int[m];
   int*    lbsol   = new int[n];

   ofstream flog;
   flog.open ("lagr.log");
   flog << fixed << setprecision(3);

   zcurr     = INT_MAX;       // initial upper bound
   zlb       = DBL_MIN;
   zlbBest   = DBL_MIN;
   /***************** 1 个block，n个thread，无需共享内存 *****************************************************/
   for(i=0;i<m;i++)
      lambda[i] = 0.0;

   iter  = 0;
   if(maxiter < 0) maxiter = INT_MAX;
   while(alpha > minAlpha && iter < maxiter)
   {  
      /*************** 此处改为核函数 **********************************************************************/
      // 算得初始下界及解
      lbsol = subproblem_ass(c, &zlb, &zlbBest, zub, lambda, subgrad);
      /*************** 此处改为核函数 **********************************************************************/
      zcurr = GAP->checkSol(lbsol);

      // 当前解约等于下界，即最优解
      if(zcurr == zlbBest  || (zub-zlbBest) < 1.0) // -------------------------- Optimum found 
      {  
         /***************** 1 个block，n个thread，无需共享内存 ***********************************************/
         for(i=0;i<n;i++) solbest[i]=sol[i];
         zub = zcurr;
         if(GAP->isVerbose) cout << "[lagrAss] Found the optimum !!! zopt="<<zub<<" iter "<< iter << endl;
         goto lend;
      }
      // 继续更新解，启发式搜索
      else                                       // -------------------------- Heuristic block
      {  
         // 初始状态
         if(zcurr == INT_MAX)
            /********************* 改为核函数 ***************************************************************/
            // 更新解
            //GAP->fixSolViaKnap(lbsol, &zcurr);   // hope to fix infeasibilities
            GAP->fixSol(lbsol, &zcurr);
         // 如果获得的解与下界较接近，则通过局部搜索尝试得到更优解       
         if(zcurr < 10*zlb)
         {  for(int j=0;j<n;j++) sol[j] = lbsol[j];
            /********************* 改为核函数 ***************************************************************/
            LocalSearch* LS = new LocalSearch(GAP, GAP->zub);
            LS->opt10(c);    
            delete LS;        
         }
         // 更新上界
         if(zcurr<zub)
         {  for(i=0;i<n;i++) solbest[i]=sol[i];
            zub = zcurr;
            if(GAP->isVerbose) cout << "[lagrAss] -------- zub improved! " << zub << endl;
         }
      }

      // 计算步长
      sumSubGrad2 = 0;                            // ------------------------ step redefinition
      /***************** 1 个block，m个thread，需要共享内存n*int ***********************************************/
      for(i=0;i<m;i++)
         sumSubGrad2 += subgrad[i]*subgrad[i];
      fakeZub = min((double) zcurr,1.2*zlb);
      fakeZub = max(fakeZub,zlb+1);
      step = alpha*(fakeZub-zlb)/sumSubGrad2;

      /***************** 1 个block，m个thread，无需共享内存 ***************************************************/
      for(i=0;i<m;i++)                            // -------------------------- penalty update
         lambda[i] = max(0.0,lambda[i]+step*subgrad[i]);
      iter++;
      if(iter % innerIter == 0)
         alpha = alphastep*alpha;

      if(iter%1000 == 0)                          // -------------------------- logging
      {  if(GAP->isVerbose) cout << "[lagrAss] iter="<<iter<<" zub="<<zub<<" zlb="<<zlbBest<<" zcurr="<<zcurr<<endl;
         if(GAP->isVerbose) writeIterData(flog, iter, zlb, zub, zcurr, alpha, lbsol, subgrad, lambda, step);
      }
   }

lend:    
   if (flog.is_open()) flog.close();
   if (lambda!=NULL)  free(lambda);
   if (subgrad!=NULL) free(subgrad);
   if (lbsol!=NULL)   free(lbsol);
   return zcurr;
}

/**************************************
 * 初始化 【O(max(n,m))->O(max(log_n,1+log_m))】
 * 求初解：
 *    各工作的最优解 【O(m*n)->O(log_m)】
 *    更新解、下界等 【O(m)->O(log_n)】

 * 【O(m*n)->O(max(log_n,log_m))】
*************************************/
// 将工作分配到工厂，将初始拉格朗日乘子作为惩罚项加在工厂的资源上，算得初始下界
// assigns each client to a server
int* Lagrangian::subproblem_ass(int** c, double *zlb, double *zlbBest, int zub, double* lambda, int* subgrad)
{  int i,j,mini;
   double mincost;
   /********************** 直接申请gpu空间 ********************************************************************************/
   int* sol = new int[n];

   /************** 1个block，n个thread，无需共享内存；赋值直接初始化 ********************************************************/
   // 初始工作的解均为-1
   for(j=0;j<n;j++) sol[j]     = -1;
   /************** 1个block，m个thread，需要共享内存m*2*int；归约求和 ********************************************************/
   // 次梯度均为工厂的容量取反（后续-决策变量，最终结果为剩余容量，取反）
   for(i=0;i<m;i++) subgrad[i] =  0;

   *zlb = 0;
   for(i=0;i<m;i++)
   {  subgrad[i] -= GAP->cap[i];
      *zlb -= GAP->cap[i]*lambda[i];       // penalties sum in the lagrangian function
   }

   // 对于每个工作，找到吸收惩罚项后的最小花费
   for(j=0;j<n;j++)
   {  mincost = DBL_MAX;
      mini    = INT_MAX;
      /*************** n个block，m个thread，需要共享内存n*2*int；归约求最小值，得到解 *******************************************/
      for(i=0;i<m;i++)                    // finds the minimum cost assignment
         if( (c[i][j]+(req[i][j]*lambda[i])) < mincost)
         {  mincost = c[i][j]+(req[i][j]*lambda[i]);
            mini    = i;
         }
      /*************** 1个block，n个thread，需要共享内存n*2*int；归约求和总花费，同时将原解转为矩阵 *******************************/
      /*************** m个block，n个thread，需要共享内存n*2*int；归约求和已用资源，更新每个工厂的次梯度 ***************************/
      sol[j] = mini;
      // 次梯度增加已用资源
      subgrad[mini] += req[mini][j];
      *zlb += mincost; 
   }
   // 更新下界
   assert(*zlb<=zub);  // aborts if lb > ub
   if(*zlb>*zlbBest) 
      *zlbBest = *zlb;
   return sol;
}

// just logging on file flog
void Lagrangian::writeIterData(ofstream& flog, int iter, double zlb, int zub, int zcurr, double alpha,
                               int* lbsol, int* subgrad, double* lambda, double step)
{  int i;

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
};

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
int Lagrangian::lagrCap(int** c, double alpha, double alphastep, double minAlpha, int innerIter, int maxiter)
{  int i,j,sumSubGrad2,iter=0,zcurr;
   double  zlb,step=0,zlbBest=0,fakeZub;

   /**************** 直接申请gpu空间，不进行cpu内存拷贝 ****************************************************/
   double* lambda  = new double[n]; // 拉格朗日乘子
   int*    subgrad = new int[n];    // 次梯度
   int*    lbsol   = new int[n];    // 目前的解

   ofstream flog;
   flog.open ("lagr.log");
   flog << fixed << setprecision(3);

   // 初始化上下界、拉格朗日乘子
   zcurr     = INT_MAX;       // initial upper bound
   zlb       = DBL_MIN;
   /***************** 1 个block，n个thread，无需共享内存 *****************************************************/
   for(i=0;i<n;i++)
      lambda[i] = 0.0;

   iter  = 0;
   while(alpha > minAlpha && iter < maxiter)
   {  
      /*************** 此处改为核函数 **********************************************************************/
      // 算得初始下界及解
      //clock_t start_t = clock();
      subproblem_cap(c, &zlb, &zlbBest, zub, lambda, subgrad, lbsol);
      //clock_t end_t = clock();
      //cout << "Time: " << end_t - start_t << endl;
      /*************** 此处改为核函数 **********************************************************************/
      zcurr = GAP->checkSol(lbsol);
      //cout << zcurr << endl;
      //cin.get();
      // 更新上界
      if(zcurr<zub)
      {  
         for(i=0;i<n;i++) solbest[i]=lbsol[i];
         zub = zcurr;
         if(GAP->isVerbose) cout << "[lagrCap] -------- zub improved! " << zub << endl;
      }
      // 当前解约等于下界，即最优解
      if((zub-zlbBest) < 1.0)                       // -------------------------- Optimum found 
      {  
         /***************** 1 个block，n个thread，无需共享内存 ***********************************************/
         if(GAP->isVerbose) cout << "[lagrCap] Found the optimum!!! zopt="<< zub << " zlb=" << zlbBest<<endl;
         goto lend;
      }
      // 继续更新解，启发式搜索
      else                                       // -------------------------- Heuristic block
      {  
         // 初始状态
         if(zcurr == INT_MAX)
            /********************* 改为核函数 ***************************************************************/
            // 通过背包更新解
            //GAP->fixSolViaKnap(lbsol, &zcurr);   // hope to fix infeasibilities
            GAP->fixSol(lbsol, &zcurr);       
         // 如果获得的解与下界较接近，则通过局部搜索尝试得到更优解
         if(zcurr < 10*zlb)
         {  for(int j=0;j<n;j++) sol[j] = lbsol[j];
            LocalSearch* LS = new LocalSearch(GAP, GAP->zub);
            /********************* 改为核函数 ***************************************************************/
            LS->opt10(c);    
            delete LS;        
         }
      }

      // 计算步长   
      // -------------------------- calcolo passo
      sumSubGrad2 = 0;
      /***************** 1 个block，n个thread，需要共享内存n*int ***********************************************/
      for(j=0;j<n;j++)
         sumSubGrad2 += subgrad[j]*subgrad[j];
      fakeZub = min((double) zcurr,1.2*zlb);
      fakeZub = max(fakeZub,zlb+1);
      step = alpha*(fakeZub-zlb)/sumSubGrad2;

      /***************** 1 个block，n个thread，无需共享内存 ***************************************************/
      for(j=0;j<n;j++)                            // -------------------------- penalty update
         lambda[j] += step*subgrad[j];
      iter++;
      if(iter % innerIter == 0)
         alpha = alphastep*alpha;

      if(iter%100 == 0)                           // -------------------------- logging
      {  if(GAP->isVerbose) cout << "[lagrCap] iter="<<iter<<" zub="<<zub<<" zlb="<<zlbBest<<" zcurr="<<zcurr<<endl;
         if(GAP->isVerbose) writeIterData(flog, iter, zlb, zub, zcurr, alpha, lbsol, subgrad, lambda, step);
      }
   }

lend:    
   if (flog.is_open()) flog.close();
   if (lambda!=NULL)  delete(lambda);
   if (subgrad!=NULL) delete(subgrad);
   if (lbsol!=NULL)   delete(lbsol);
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
void Lagrangian::subproblem_cap(int** c, double *zlb, double *zlbBest, int zub, double* lambda, int* subgrad, int* lbsol)
{  int i,j;
   /********************** 直接申请gpu空间，由于并行，申请的空间为二维数组 ***************************************************/
   // 工厂循环利用数组；工厂i执行工作j时消耗的资源
   int* Q   = new int[n];
   double* val = new double[n];
   int* Ksol = new int[n];

   /************** 1个block，n个thread，需要共享内存n*double；赋值直接初始化，zlb使用归约 *************************************/
   // 初始工作的解均为-1，次梯度均为1（后续-决策变量，最终结果为该工作可以被分配为多少个工厂,取反）
   *zlb = 0;
   for(j=0;j<n;j++)
   {  lbsol[j]   = -1;
      subgrad[j] = 1;
      *zlb += lambda[j];               // penalty sum in the lagrangian function
   }
   //printf("Sub2 %lf\n", *zlb);

   // 对于每个工厂均求一遍吸收惩罚项后的背包
   for(i=0;i<m;i++)
   {  
      /*************** m个block，n个thread，无需共享内存；直接赋值初始化 *******************************************************/
      for(j=0;j<n;j++)
      {  // 每个工作消耗的资源
         Q[j]    = req[i][j];
         // 对每个工作的“花费-拉格朗日乘子”取反，以使用背包求得其最小值（即取反后的最大值）
         val[j]  = -c[i][j]+lambda[j];  // inverted sign to make it minimzation
         // 初始解均为0
         Ksol[j] = 0;
      }
      /***************************** 改为核函数 *****************************************************************************/
      // zlb为消耗的总花费
      *zlb -= KDynRecur(n,GAP->cap[i],Q,val,Ksol);  // minus because it is minimization
      /*************** n个block，m个thread，需要共享内存m*int；归约求每个工作对应的总共的工厂数，并选择其解 **********************/
      /************* 注意此处n在前，m在后 **********************/
      // 启发式：对每个工作更新其状态（最终解为最后一个工厂）
      // 如果可在该工厂执行，则次梯度-1
      for(j=0;j<n;j++)
         if(Ksol[j] > 0)
         {  lbsol[j] = i;              // could be a reassignment, obviously. This is heuristic
            subgrad[j] -= 1;
         }
   }
   //printf("Sub3 %lf\n", *zlb);
   //cin.get();

   assert(*zlb<=zub);  // aborts if lb > ub
   if(*zlb>*zlbBest) 
      *zlbBest = *zlb;

   free(Q);
   free(val);
   free(Ksol);
   return;
}
