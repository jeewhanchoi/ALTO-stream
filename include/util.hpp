#ifndef UTIL_HPP_
#define UTIL_HPP_

#include "common.hpp"
#include "assert.h"
#include <math.h>
#include "bitops.hpp"
#include "kruskal_model.hpp"
#include "matrix.hpp"

struct permutation_struct {
    IType * perms[MAX_NUM_MODES];
    IType * iperms[MAX_NUM_MODES];
}; 
typedef struct permutation_struct Permutation;

Permutation * perm_alloc(
    IType const * const dims, int const nmodes);

FType rand_val();
void fill_rand(FType * vals, IType num_el, unsigned int seed);

IType argmin_elem(
  IType const * const arr,
  IType const N);


inline int get_num_ptrn(IType nnz)
{
  int max_threads = omp_get_max_threads();
  int num_partitions = max_threads;
  int nnz_ptrn = (nnz + num_partitions - 1) / num_partitions;

  if (nnz < num_partitions) {
    num_partitions = 1;
  }
  else {
    while (nnz_ptrn < max_threads) {
    // Insufficient nnz per partition
    //fprintf(stderr, "Insufficient nnz per partition: %d ... Reducing # of partitions... \n", nnz_ptrn);               
      num_partitions /= 2;
      nnz_ptrn = (nnz + num_partitions - 1) / num_partitions;
    }
  }
  return num_partitions;  
}

inline int get_num_bits(IType* dims, int nmodes)
{
  int num_bits = 0;

  for (int n = 0; n < nmodes; ++n) {
      int mbits = (sizeof(IType) * 8) - clz(dims[n] - 1);
      num_bits += mbits;  
  }
  return num_bits;  
}
#endif // UTIL_HPP_