#ifndef GAP_H
#define GAP_H
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <cstdlib>
#include <curand_kernel.h>
#include "node.cuh"
using namespace std;


class GeneralizedAssignemnt {
   public:
      GeneralizedAssignemnt();
      ~GeneralizedAssignemnt();
      int n;       // number of clients
      int m;       // number of servers
      int Kcap;    // Kcap：设为工厂最大容量+1，以方便申请数组空间
      int nDays;   // planning horizon (if scheduling)
      // c与req本质上为2维
      int* c_cpu;
      int* c;     // assignment costs
      int* req_cpu;
      int* req;   // client requests
      int* cap_cpu;
      int* cap;   // server capacities

      int *sol,*solbest;    // for ewach client, its server
      int zub,zlb;
      bool isVerbose;

      curandState* randStates;

      int* temp_req;
      double* f; // 遍历到第i个物品，已用容量为q，工厂k时的最小花费。归约需要，不能改变顺序。
      node<double>* final_f; // dp完成后，已用容量为q、工厂k的最小花费；归约后为各个工厂的最低花费
      node<int>* anss;
      double* val;
      int* Ksol ;
      node<int>* temp_ksol;
      int* temp_cost;          // 当前解对于各工作的花费

      int* fsol;                  // 当前解

      node<int>* temp_res;
      int* temp_subgrad;
     
      int* capleft;
      int* temp_c;
      

      double EPS;

      int checkSol(int* sol);                       // feasibility check
      int fixSol(int* infeasSol, int* zsol);        // recovers feasibility in case of partial or overassigned solution
      int fixSolViaKnap(int* infeasSol, int* zsol); // recovers feasibility via knapsacks on residual capacities
      double KDynRecur(int m, int n, int* cap, int* req, double* val, int* Ksol);

      void readData(string filePath, unsigned long seed);

};



#endif // GAP_H
