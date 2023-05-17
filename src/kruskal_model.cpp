#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <util.hpp>

#include "kruskal_model.hpp"

// #define DEBUG 1
#define MAX(x, y) (((x) > (y)) ? (x) : (y))


void ExportKruskalModel(KruskalModel *M, char *file_path)
{
    // factor matrices
    for (int n = 0; n < M->mode; n++) {
        char str[1000];
        sprintf(str, "%s.%d.out", file_path, n);
        FILE *fp = fopen(str, "w");
        assert(fp);
        fprintf(fp, "matrix\n");
        fprintf(fp, "2\n");
        fprintf(fp, "%llu %llu\n", M->dims[n], M->rank);
        for (IType i = 0; i < M->rank; i++) {
            for (IType j = 0; j < M->dims[n]; j++) {
                fprintf(fp, "%.20lf\n", M->U[n][j * M->rank + i]);
            }
        }
        fclose(fp);
    }

    // lambda
    char str[1000];
    sprintf(str, "%s.lambda.out", file_path);
    FILE* fp = fopen(str, "w");
    assert(fp);
    fprintf(fp, "vector\n");
    fprintf(fp, "1\n");
    fprintf(fp, "%llu\n", M->rank);
    for(IType i = 0; i < M->rank; i++) {
        fprintf(fp, "%.20lf\n", M->lambda[i]);
    }
    fclose(fp);
}


void PrintKruskalModel(KruskalModel *M)
{
    for (int n = 0; n < M->mode; n++) {
        fprintf(stderr, "mode %d:\n", n);
        for (IType j = 0; j < M->dims[n]; j++) {
            for (IType i = 0; i < M->rank; i++) {
                
                // fprintf(stderr, "%.15f ", M->U[n][j * M->rank + i]);
                fprintf(stderr, "%.4f ", M->U[n][j * M->rank + i]);
                // printf("i: %.15f \n", i);

            }
            fprintf(stderr, "\n");
        }
    }
    fprintf(stderr, "lambda:");
    for (IType r = 0; r < M->rank; r++) {
        fprintf(stderr, "%g ", M->lambda[r]);
    }
    fprintf(stderr, "\n");
}


void CreateKruskalModel(int mode, IType *dims, IType rank, KruskalModel **M_)
{
    assert(mode >= 1);
    assert(rank >= 1);
    for (int n = 0; n < mode; n++) {
        assert(dims[n] >= 1);
        //assert(rank <= dims[n]);
    }
    
    KruskalModel *M = (KruskalModel *)AlignedMalloc(sizeof(KruskalModel));
    assert(M != NULL);
    M->mode = mode;
    M->rank = rank;
    M->dims = (IType *)AlignedMalloc(mode * sizeof(IType));
    assert(M->dims != NULL);
    memcpy(M->dims, dims, sizeof(IType) * mode);  
    M->U = (FType **)AlignedMalloc(mode * sizeof(FType *));
    assert(M->U != NULL);
    for (int n = 0; n < mode; n++) {
        M->U[n] = (FType *)AlignedMalloc(dims[n] * rank * sizeof(FType));
        assert(M->U[n] != NULL);
    }
    M->lambda = (FType *)AlignedMalloc(rank * sizeof(FType));
    assert(M->lambda != NULL);

    *M_ = M;
}

void GrowKruskalModel(IType * dims, KruskalModel * M, FillValueType FillValueType_, unsigned int seed)
{
    IType mode = M->mode;
    IType rank = M->rank;
    IType * old_dims = M->dims; // previous dimension sizes
    FType ** U = M->U;

    assert(rank >= 1);
    assert(mode >= 1);

    for (int n = 0; n < mode; ++n) {
        U[n] = (FType *)realloc(U[n], dims[n] * rank * sizeof(FType));
        assert(U[n] != NULL);

        // Fill exceeding values with zeros
        int added_nrows = dims[n] - old_dims[n];
        if (added_nrows > 0) { // If we need to add more rows

            if (FillValueType_ == FILL_RANDOM) {
                fill_rand(&(U[n][old_dims[n] * rank]), added_nrows * rank, seed);
            } 

            else if (FillValueType_ == FILL_ZEROS) {
                #pragma omp for simd schedule(static)
                for (int r = 0; r < added_nrows * rank; ++r) {
                    U[n][old_dims[n] * rank + r] = 0.0;
                }

            }
        }
    }
    memcpy(M->dims, dims, sizeof(IType) * mode); // copy new dims to new kruskal
}

void CopyKruskalModel(KruskalModel *prev_M_, KruskalModel *M_)
{
    IType mode = M_->mode;
    IType rank = M_->rank;

    memcpy(prev_M_->dims, M_->dims, sizeof(IType) * mode);
    for (int n = 0; n < mode; ++n) {
        ParMemcpy(prev_M_->U[n], M_->U[n], M_->dims[n] * rank * sizeof(FType));
    }
    memcpy(prev_M_->lambda, M_->lambda, sizeof(FType) * rank);
}

void GrowTimeFactorMatrix(KruskalModel **M_, int streaming_mode) {
    int current_dim = (*M_)->dims[streaming_mode];
    int rank = (*M_)->rank;
    FType * tmp = (FType *)AlignedMalloc((current_dim + 1) * sizeof(FType));
    memcpy(tmp, (*M_)->U[streaming_mode], sizeof(FType) * current_dim * rank);
    AlignedFree((*M_)->U[streaming_mode]);

    // Set Time Factor to 0 for newly added row
    for (int r = 0; r < rank; ++r) {
        tmp[current_dim * rank + r] = 0.0;
    }

    (*M_)->U[streaming_mode] = tmp;
    (*M_)->dims[streaming_mode] = current_dim + 1;
}

void KruskalModelRandomInit(KruskalModel *M, unsigned int seed)
{
    for (IType i = 0; i < M->rank; i++) {
        M->lambda[i] = (FType) 1.0;
    }

    for (int n = 0; n < M->mode; n++) {
        #if DEBUG == 1
        for (IType i = 0; i < M->dims[n] * M->rank; i++) {
            M->U[n][i] = (FType) 1.0;
        }
        #else
        fill_rand(M->U[n], M->dims[n] * M->rank, seed);
        #endif
    }
}

void KruskalModelZeroInit(KruskalModel *M)
{
    for (IType i = 0; i < M->rank; i++) {
        M->lambda[i] = (FType) 0.0;
    }

    for (int n = 0; n < M->mode; n++) {
        #pragma omp parallel
        {
            #pragma omp for simd schedule(static)
            for (IType i = 0; i < M->dims[n] * M->rank; i++) {
                M->U[n][i] = (FType) 0.0;
            }
        }
    }
}


void KruskalModelNormalize(KruskalModel *M)
{
    for (int n = 0; n < M->mode; n++) {
        // For each factor
        IType dim = M->dims[n];
        for (IType j = 0; j < M->rank; j++) {
            // Calculate the norm for this column
            FType tmp = 0.0;
            for (IType k = 0; k < dim; k++) {
                #if ROW
                tmp = tmp + fabs(M->U[n][k * M->rank + j]);
                #else
                tmp = tmp + fabs(M->U[n][j * dim + k]);
                #endif
            }
            // Normalize the elements 
            for (IType k = 0; k < dim; k++) {
                #if ROW
                M->U[n][k * M->rank + j] = M->U[n][k * M->rank + j] / tmp;
                #else
                M->U[n][j * dim + k] = M->U[n][j * dim + k] / tmp;
                #endif
            }
            // Absorb the norm into lambda
            M->lambda[j] = M->lambda[j] * tmp;
        }
    }
}

static void inline Mat2Norm(IType dim, IType rank, FType * vals, FType * lambda, FType ** scratchpad)
{
    IType nthreads = omp_get_max_threads();
    // Find the max value in each column and store it in lambda
    #pragma omp parallel proc_bind(close)
     {
         IType tid = omp_get_thread_num();
         FType * _lambda = scratchpad[tid];

        #pragma omp for schedule(static) 
        for(IType i = 0; i < dim; i++) {
            #pragma omp simd
            for(IType j = 0; j < rank; j++) {
                _lambda[j] += vals[i * rank + j] * vals[i * rank + j];
            }
        }

        #pragma omp for reduction(+: lambda[:rank]) schedule(static)
        for (IType t = 0; t < nthreads; ++t) {
            #pragma omp simd
            for (IType j = 0; j < rank; ++j) {
                lambda[j] += scratchpad[t][j];
            }
        }
    }

    #pragma omp for schedule(static)
    for(IType j=0; j < rank; ++j) {
      // TODO - More performant way perhaps
      // There is a bug where all columns are 0.0 
      // Divide by zero error occurs
    //   lambda[j] = std::max(lambda[j], 1e-12);
        lambda[j] = MAX(lambda[j], 1e-12);
        // if (lambda[j] == 0.0f) {
        //     fprintf(stderr, "lambda is zero!!!!\n");
        //     exit(1);
        // }
        lambda[j] = sqrt(lambda[j]);
    }

    #pragma omp parallel for schedule(static)
    for(IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            vals[i * rank + j] /= lambda[j];
        }
    }

}

static void inline MatMaxNorm(IType dim, IType rank, FType * vals, FType * lambda, FType ** scratchpad)
{
    IType nthreads = omp_get_max_threads();

    // Find the max value in each column and store it in lambda
    #pragma omp parallel proc_bind(close)
     {
         IType tid = omp_get_thread_num();
         FType * _lambda = scratchpad[tid];

        #pragma omp for schedule(static) 
        for(IType i = 0; i < dim; i++) {
            #pragma omp simd
            for(IType j = 0; j < rank; j++) {
                _lambda[j] = std::max(_lambda[j], vals[i * rank + j]);
            }
        }

        // If any entry is less than 1, set it to 1
        #pragma omp simd
        for(IType i = 0; i < rank; i++) {
            _lambda[i] = std::max(_lambda[i], 1.);
        }

        #pragma omp for reduction(max: lambda[:rank]) schedule(static)
        for (IType t = 0; t < nthreads; ++t) {
          #pragma omp simd
          for (IType j = 0; j < rank; ++j) {
              lambda[j] = std::max(lambda[j], scratchpad[t][j]);
          }
        }
    }

    #pragma omp parallel for schedule(static)
    for(IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            vals[i * rank + j] /= lambda[j];
        }
    }
}

void KruskalModelNorm(KruskalModel* M, IType mode, mat_norm_type which, FType ** scratchpad)
{
    IType dim = M->dims[mode];
    IType rank = M->rank;
    FType * vals = M->U[mode];
    FType * lambda = M->lambda;

    IType nthreads = omp_get_max_threads();

    // Initialize lambda scratchpad
    #pragma omp parallel for schedule(static)
    for (IType t = 0; t < nthreads; ++t) {
        #pragma omp simd
        for (IType r = 0; r < rank; ++r) {
            scratchpad[t][r] = 0.0;
        }
    }
    #pragma omp simd
    for (IType r = 0; r < rank; ++r) {
        lambda[r] = 0.0;
    }

    // Call normalization accordingly...
    switch (which) {
    case MAT_NORM_2:
        Mat2Norm(dim, rank, vals, lambda, scratchpad);
        break;
    case MAT_NORM_MAX:
        MatMaxNorm(dim, rank, vals, lambda, scratchpad);
        break;

    default:
        abort();
    }
}


void DestroyKruskalModel(KruskalModel *M)
{
    AlignedFree(M->dims);
    for (int n = 0; n < M->mode; n++) {
        AlignedFree(M->U[n]);
    }
    AlignedFree(M->U);
    AlignedFree(M->lambda);
    AlignedFree(M);
}

void RedistributeLambda (KruskalModel *M, int n)
{
    FType *U = M->U[n];
    IType rank = M->rank;
    FType *lambda = M->lambda;
    IType dim = M->dims[n];

    for(IType r = 0; r < rank; r++) {
        for(IType i = 0; i < dim; i++) {
            #if ROW
            U[i * rank + r] = U[i * rank + r] * lambda[r];
            #else
            U[r * dim + i] = U[r * dim + i] * lambda[r];
            #endif
        }
        lambda[r] = 1.0;
    }
}

void PrintKruskalModelInfo(KruskalModel *M) {
    for (int i = 0; i < M->mode; ++i) {
        fprintf(stdout, "%llu", M->dims[i]);
        if (i != M->mode - 1) fprintf(stdout, " x ");
        else fprintf(stdout, "\n");
    }
};


double KruskalTensorFit()
{
  return 0.0;
}

double kruskal_norm(KruskalModel * M) {
  IType const rank = M->rank;
  FType * const scratch = (FType *) malloc(rank * rank * sizeof(*scratch));

  Matrix * ata = zero_mat(rank, rank);

  /* initialize scratch space */
  for(IType i=0; i < rank; ++i) {
    for(IType j=i; j < rank; ++j) {
      scratch[j + (i*rank)] = 1.;
    }
  }

  /* scratch = hada(aTa) */
  for(IType m=0; m < M->mode; ++m) {
    Matrix * matptr = mat_fillptr(M->U[m], M->dims[m], rank);
    // PrintMatrix("matptr", matptr);
    mat_aTa(matptr, ata);

    FType * atavals = ata->vals;
    for(IType i=0; i < rank; ++i) {
      for(IType j=i; j < rank; ++j) {
        scratch[j + (i*rank)] *= atavals[j + (i*rank)];
      }
    }
    free_mat(matptr);
  }

  /* now compute weights^T * aTa[MAX_NMODES] * weights */
  FType norm = 0;
  FType const * const column_weights = M->lambda;
  for(IType i=0; i < rank; ++i) {
    norm += scratch[i+(i*rank)] * column_weights[i] * column_weights[i];
    for(IType j=i+1; j < rank; ++j) {
      norm += scratch[j+(i*rank)] * column_weights[i] * column_weights[j] * 2;
    }
  }

  free(scratch);
  free_mat(ata);
    
  return fabs(norm);
}


KruskalModel * get_prev_kruskal(
    KruskalModel * M, IType mode_of_interest, IType previous)
{
    IType nmodes = M->mode;
    IType rank = M->rank;
    /* store output */
    KruskalModel * cpd = (KruskalModel *) AlignedMalloc(sizeof(*cpd));

    cpd->mode = nmodes;
    cpd->lambda = (FType *) AlignedMalloc(rank * sizeof(*cpd->lambda));
    cpd->dims = (IType *)AlignedMalloc(nmodes * sizeof(IType));
    cpd->U = (FType **) AlignedMalloc(nmodes * sizeof(FType*));
    cpd->rank = rank;

    for(IType r=0; r < rank; ++r) {
      cpd->lambda[r] = 1.;
    }

    Matrix fm;
    mat_hydrate(&fm, M->U[mode_of_interest], M->dims[mode_of_interest], rank);

    for(IType m=0; m < nmodes; ++m) {
      if(m == mode_of_interest) {
        IType const nrows = std::min(previous, fm.I);
        IType const startrow = fm.I - nrows;

        cpd->dims[m] = nrows;
        cpd->U[m] = (FType *)AlignedMalloc(nrows * rank * sizeof(FType));
        ParMemcpy(cpd->U[m], &(fm.vals[startrow * rank]), nrows * rank * sizeof(FType));
      } else {

        IType const nrows = M->dims[m];

        cpd->dims[m] = nrows;
        cpd->U[m] = (FType *) AlignedMalloc(nrows * rank * sizeof(FType));
        /* permute rows */
        #pragma omp parallel for schedule(static)
        for(IType i=0; i < nrows; ++i) {
          IType const new_id = i;
          memcpy(&(cpd->U[m][i * rank]),
                &(M->U[m][new_id * rank]),
                rank * sizeof(FType));
        }
      }
    }

    return cpd;
}