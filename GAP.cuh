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
using namespace std;


class GeneralizedAssignemnt {
   public:
      GeneralizedAssignemnt();
      ~GeneralizedAssignemnt();
      int n;       // number of clients
      int m;       // number of servers
      int nDays;   // planning horizon (if scheduling)
      // c与req本质上为2维
      int* c;     // assignment costs
      int* req;   // client requests
      int* cap;   // server capacities

      int *sol,*solbest;    // for ewach client, its server
      int zub,zlb;
      bool isVerbose;

      curandState* randStates;

      double EPS;

      int checkSol(int* sol);                       // feasibility check
      int fixSol(int* infeasSol, int* zsol);        // recovers feasibility in case of partial or overassigned solution
      int fixSolViaKnap(int* infeasSol, int* zsol); // recovers feasibility via knapsacks on residual capacities

      void readData(string filePath, unsigned long seed);
};

double KDynRecur(int m, int n, int* cap, int* req, double* val, int* Ksol);
void KdecodeSol(int i, int Kcap, int* Q, double* val, int n, double** f, int* sol);
void printDblArray(double* a, int n);

#endif // GAP_H
