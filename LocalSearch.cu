#include "LocalSearch.cuh"
#include "gpu.cuh"

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
int LocalSearch::opt10(int* c)
{  
   int z=0;
   int isol;
   int* capleft = GAP->capleft;
   int* temp_req = GAP->temp_req;
   int* temp_c = GAP->temp_c;
   node<int>* anss = GAP->anss;
   node<int> ans;
   

   vectorCopy<<<NumBlocks(m), NUM_THREADS>>>(GAP->cap, capleft, m);
   vectorInit<<<NumBlocks(m*n), NUM_THREADS>>>(temp_req, m*n, 0);
   vectorToMatrixReq<<<NumBlocks(n), NUM_THREADS>>>(sol, temp_req, req, n, m);
   //printf("12\n");
   doReduction(temp_req,m,n,temp_req,2);
   vectorSub<int><<<NumBlocks(m), NUM_THREADS>>>(capleft,temp_req, capleft,m);

   opt10Init<<<NumBlocks(n), NUM_THREADS>>>(temp_c,c,sol,n);
   int* temp = get_temp(z);
   //printf("13\n");
   doReduction(temp_c,1,n,temp,2);
   z = delete_temp(temp);

   for (; ;) {
      // 对于每个工作，遍历每个不是当前解的工厂，直到找到花费更少且有剩余容量的。
      // 找到后更新当前数组，并从头开始搜索，直到无法找到更优解。
      opt10Main<<<NumBlocks(n*m), NUM_THREADS>>>(c, capleft, req, sol, z, m, n, anss);
      node<int>* temp2 = get_temp(ans);
      //printf("14\n");
      doReduction(anss, 1, m * n, temp2, 0);
      ans = delete_temp(temp2);
      z = ans.data;
      if (z == INT_MAX)
         break;
      int i = ans.pos/n, j = ans.pos%n;
      opt10Update<<<1,1>>>(sol, capleft, req, i, j, n);
      if(z<zub)
      {  
         vectorCopy<int><<<NumBlocks(n), NUM_THREADS>>>(sol, solbest, n);
         zub = z;
         if(GAP->isVerbose) cout << "[1-0 opt] new zub " << zub << endl;
      }
   }

   return z;
}
