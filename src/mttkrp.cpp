#include "mttkrp.hpp"
#include <assert.h>

void mttkrp_par(SparseTensor* X, KruskalModel* M, IType mode, omp_lock_t* writelocks)
{
  IType nmodes = X->nmodes;
  IType nnz = X->nnz;
  IType** cidx = X->cidx;
  IType rank = M->rank;

  int max_threads = omp_get_max_threads();
  FType* rows = (FType*) AlignedMalloc(sizeof(FType) * rank * max_threads);
  assert(rows);

  #pragma omp parallel
  {
    // get thread ID
    int tid = omp_get_thread_num();
    FType* row = &(rows[tid * rank]);

    #pragma omp for schedule(static)
    for(IType i = 0; i < nnz; i++) {
      // initialize temporary accumulator
      for(IType r = 0; r < rank; r++) {
        row[r] = X->vals[i];
      }

      // calculate mttkrp for the current non-zero
      for(IType m = 0; m < nmodes; m++) {
        if(m != mode) {
          IType row_id = cidx[m][i];
          for(IType r = 0; r < rank; r++) {
            row[r] *= M->U[m][row_id * rank + r];
          }
        }
      }

      // update destination row
      IType row_id = cidx[mode][i];
      omp_set_lock(&(writelocks[row_id]));
      for(IType r = 0; r < rank; r++) {
        M->U[mode][row_id * rank + r] += row[r];
      }
      omp_unset_lock(&(writelocks[row_id]));
    } // for each nonzero
  } // #pragma omp parallel
  // free memory
  free(rows);
}

void mttkrp_par_scratchpad(
  SparseTensor* X, KruskalModel* M,
   IType mode, omp_lock_t* writelocks)
{
  IType I = X->dims[mode]; // Should be 1 since only for streaming mode
  IType rank = M->rank;
  FType * outmat = M->U[mode];
  IType nmodes = X->nmodes;

  IType num_threads;
  // Initialize mttkrp output row
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
    #pragma omp for schedule(static)
    for (IType x = 0; x < I * rank; ++x) {
      outmat[x] = 0.0;
    }
  }

  FType * outrow_tmp = (FType*)malloc(rank * num_threads * sizeof(FType));

  #pragma omp for schedule(static)
  for (IType i=0; i < rank * num_threads; i++) {
    outrow_tmp[i] = 0.0;
  }

  #pragma omp parallel
  {
    FType * accum = (FType*)malloc(rank * sizeof(FType));
    // Stream through nnz
    #pragma omp for schedule(static)
    for (IType n = 0; n < X->nnz; ++n) {
      IType tid = omp_get_thread_num();

      // Init with nnz value
      for (IType r = 0; r < rank; ++r) {
        accum[r] = X->vals[n];
      }

      for (IType m = 0; m < nmodes; ++m) {
        if (m == mode) {
          continue;
        }

        FType * inrow = M->U[m] + (X->cidx[m][n] * rank);
        for (IType r = 0; r < rank; ++r) {
          accum[r] *= inrow[r];
        }
      }

      FType * tmp_outrow = outrow_tmp + (tid * rank);
      // Write to buf
      for (IType r = 0; r < rank; ++r) {
        tmp_outrow[r] += accum[r];
      }
    }

    // Implicit barrier
    #pragma omp for schedule(static)
    for(IType i = 0; i < rank; ++i) {
      FType * outrow = outmat;
      for(IType t = 0; t < num_threads; ++t) {
        outrow[i] += outrow_tmp[t * rank + i];
      }
    }   
    free(accum);
  }
  free(outrow_tmp);
}

void mttkrp(SparseTensor* X, KruskalModel* M, IType mode)
{
  IType nmodes = X->nmodes;
  //IType* dims = X->dims;
  IType nnz = X->nnz;
  IType** cidx = X->cidx;
  IType rank = M->rank;

  FType row[rank];

  for(IType i = 0; i < nnz; i++) {
    // initialize temporary accumulator
    for(IType r = 0; r < rank; r++) {
      row[r] = X->vals[i];
    }

    // calculate mttkrp for the current non-zero
    for(IType m = 0; m < nmodes; m++) {
      if(m != mode) {
        IType row_id = cidx[m][i];
        for(IType r = 0; r < rank; r++) {
          row[r] *= M->U[m][row_id * rank + r];
        }
      }
    }

    // update destination row
    IType row_id = cidx[mode][i];
    for(IType r = 0; r < rank; r++) {
      M->U[mode][row_id * rank + r] += row[r];
    }
  } // for each nonzero

  // PrintFPMatrix("MTTKRP", dims[mode], rank, M->U[mode], rank);
}

