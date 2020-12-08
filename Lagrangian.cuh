#ifndef LAGRANGIAN_H
#define LAGRANGIAN_H
#include "GAP.cuh"

class Lagrangian
{
   public:
      GeneralizedAssignemnt* GAP;

      Lagrangian(GeneralizedAssignemnt*, int&);
      ~Lagrangian();
      int lagrCap(int*,double, double, double, int, int);

   private:
      void subproblem_cap(int*, double*, double*, int, double*, int*, int*);
      //void writeIterData(ofstream&, int, double, int, int, double, int*, int*, double*, double);

      // local mirrors
      int m,n;
      int *sol,*solbest;
      int* req;
      int & zub;
};

#endif // LAGRANGIAN_H
