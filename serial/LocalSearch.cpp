#include "LocalSearch.h"

LocalSearch::LocalSearch(GeneralizedAssignemnt* GAPinstance, int & zz) : zub(zz)
{
   //ctor
   GAP = GAPinstance;
   m = GAP->m;
   n = GAP->n;
   sol = GAP->sol;
   solbest = GAP->solbest;
   req = GAP->req;
}

LocalSearch::~LocalSearch()
{
   //dtor
}

/**********************
 * 计算旧解中每个工厂剩余的资源量与总花费 【O(max(m,n))->O(max(log_n,log_m))】
 * 无数次循环，直到无法搜索【#】:
 *    依次寻找直到可以更优【O(n/2*m/2)->O(max(log_n,log_m))】
 * 
 * 【O(n/2*m/2)->O(max(log_n,log_m))】 (前者复杂度存疑)
**********************/
// tries each client with each other facility
int LocalSearch::opt10(int** c)
{  int z=0;
   int i, isol, j;
   /**************** 申请gpu空间 **********************************************************************/
   vector<int> capleft(m);

   /***************** 1个block，n个thread，无需共享内存；将解转化为矩阵 *********************************/
   /***************** m个block，n个thread，需要共享内存n*int；归约累加，得到已占用的资源量、总花费 ********/
   /***************** 1个block，m个thread，需要共享内存m*int；更新剩余资源量，并归约得到总花费数目 ********/
   for(i=0;i<m;i++) capleft[i] = GAP->cap[i];
   for (j = 0; j < n; j++)
   {  capleft[sol[j]] -= req[sol[j]][j];
      z += c[sol[j]][j];
   }

// 毒瘤goto
l0:
   /***************** n个block，m个thread，需要共享内存m*int；同时进行搜索，归约选择最优的 ****************/
   /***************** 1个block，n个thread，需要共享内存n*int；归约选择最优的 ****************************/
   // 对于每个工作，遍历每个不是当前解的工厂，直到找到花费更少且有剩余容量的。
   // 找到后更新当前数组，并从头开始搜索，直到无法找到更优解。
   for (j = 0; j < n; j++)
   {
      isol = sol[j];
      for (i = 0; i < m; i++)
      {
         if (i == isol) continue;
         if (c[i][j] < c[isol][j] && capleft[i] >= req[i][j])
         {  // remove from isol and assign to i
            sol[j] = i;
            capleft[i]    -= req[i][j];
            capleft[isol] += req[isol][j];
            z -= (c[isol][j] - c[i][j]);
            if(z<zub)
            {  zub = z;
               /***************** 1个block，n个thread，无需共享内存 *************************************/
               for(int k=0;k<n;k++) solbest[k] = sol[k];
               if(GAP->isVerbose) cout << "[1-0 opt] new zub " << zub << endl;
            }
            goto l0;
         }
      }
   }

   return z;
}

// scambio assegnamento fra due clienti
double LocalSearch::opt11(int** c)
{  int i,j,j1,j2,temp,cap1,cap2;
   int delta, z=0, zcheck;
   vector<int> capleft(m);

   for(i=0;i<m;i++) capleft[i] = GAP->cap[i];
   for (j = 0; j < n; j++)
   {  capleft[sol[j]] -= req[sol[j]][j];
      z += c[sol[j]][j];
   }
   zcheck = GAP->checkSol(sol);

l0:
   for(j1=0;j1<n;j1++)
   {  for(j2=j1+1;j2<n;j2++)
      {  delta = (c[sol[j1]][j1] + c[sol[j2]][j2]) - (c[sol[j1]][j2] + c[sol[j2]][j1]);
         if(delta > 0)
         {  cap1 = capleft[sol[j1]] + req[j1] - req[j2];
            cap2 = capleft[sol[j2]] + req[j2] - req[j1];
            if(cap1>=0 && cap2 >=0)
            {  capleft[sol[j1]] += req[j1] - req[j2];
               capleft[sol[j2]] += req[j2] - req[j1];
               temp    = sol[j1];
               sol[j1] = sol[j2];
               sol[j2] = temp;
               z -= delta;
               zcheck = GAP->checkSol(sol);
               if(abs(z-zcheck) > GAP->EPS)
                  if(GAP->isVerbose) cout << "[1-1] ohi" << endl;
               if(z<zub)
               {  zub = z;
                  if(GAP->isVerbose) cout << "[1-1 opt] new zub " << zub << endl;
               }
               goto l0;
            }
         }
      }
   }

   zcheck = 0;
   for(j=0;j<n;j++)
      zcheck += c[sol[j]][j];
   if(abs(zcheck - z) > GAP->EPS)
      if(GAP->isVerbose) cout << "[1.1opt] Ahi ahi" << endl;
   zcheck = GAP->checkSol(sol);
   return z;
}

// un vicino 21 a caso, scambio due (stesso deposito) vs 1 (altro deposito)
void LocalSearch::neigh21()
{  int i,i1,i2,j,j11,j12,j21,iter;
   vector<int> lst1, lst2;
   vector<int> capleft(m);
   int zcheck;


   for(i=0;i<m;i++) capleft[i] = GAP->cap[i];
   for (j = 0; j < n; j++)
      capleft[sol[j]] -= req[sol[j]][j];

   i1 = rand()%m;
   i2 = rand()%m;
   if(i1==i2)
      i2 = (i2+1) % m;

   for(j=0;j<n;j++)
   {  if(sol[j]==i1)
         lst1.push_back(j);
      if(sol[j]==i2)
         lst2.push_back(j);
   }

   // 2 randomly chosen in i1 and one in i2
   iter = 0;
loop:    
   j11 = rand()%lst1.size();   // first indices then elements, in same variables!!
   j12 = rand()%lst1.size();
   j21 = rand()%lst2.size();
   if(j12==j11)
      j12=(j12+1)%lst1.size();
   j11 = lst1[j11];
   j12 = lst1[j12];
   j21 = lst2[j21];

   if( ((capleft[i1]+req[i1][j11]+req[i1][j12]-req[i1][j21]) >= 0) &&
       ((capleft[i2]-req[i2][j11]-req[i2][j12]+req[i2][j21]) >= 0) )
   {  sol[j11]=i2;
      sol[j12]=i2;
      sol[j21]=i1;
      zcheck = GAP->checkSol(sol);
      if(zcheck == INT_MAX)
         if(GAP->isVerbose) cout << "[2-1] ohi" << endl;
   }
   else  // try another, maybe feasible, neighbor
   {  iter++;
      if(iter < 50)
         goto loop;
   }

   return;
}
