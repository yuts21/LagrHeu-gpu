#ifndef LOCALSEARCH_H
#define LOCALSEARCH_H
#include "GAP.cuh"

class LocalSearch
{
   public:
      GeneralizedAssignemnt* GAP;

      LocalSearch(GeneralizedAssignemnt*, int&);
      ~LocalSearch();
      int opt10(int*);

   private:
      // local mirrors
      int m,n;
      int *sol,*solbest;
      int* req;
      int & zub,zlb;
};

#endif // LOCALSEARCH_H
