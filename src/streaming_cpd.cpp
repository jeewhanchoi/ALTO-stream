#include "streaming_cpd.hpp"

#include <vector>
#include <map>
#include <algorithm>


using namespace std;
// #define DEBUG 1
// #define SKIP_TEST 1
#define TIMER 1
#define FG_TIMER 1

// #define CHECK_FIT 1
// #define CHECK_FIT_INTERVAL 1

//#define COMPOUND_FORGETTING_FACTOR 1
#define FORGETTING_FACTOR 0.99

#if DEBUG == 1
    char _str[512];
#endif


// Implementations
void cpstream(
    SparseTensor* X,
    int rank,
    int max_iters,
    int streaming_mode,
    FType epsilon,
    IType seed,
    cpd_constraint * con,
    bool use_spcpstream,
    IType nnz_threshold,
    IType timeslice_limit)
{
    // Define timers (cpstream)
    uint64_t ts = 0;
    uint64_t te = 0;
    double t_preprocess = 0.0;
    double t_iter = 0.0;
    double t_postprocess = 0.0;
    double t_tensor = 0.0;
    double wtime, tot_time = 0.0, tot_iter = 0.0;

    IType nmodes = X->nmodes;

#ifdef MKL
    mkl_set_dynamic(1);
#endif

    // Step 1. Preprocess SparseTensor * X
    printf("Processing Streaming Sparse Tensor (nnz threshold: %llu, timeslice_limit: %llu)\n", nnz_threshold, timeslice_limit);
    StreamingSparseTensor sst(X, streaming_mode);
    sst.print_tensor_info();

    // If we're using ALTO format
    AltoTensor<LIType> * AT;
#if ADAPT_LIT == 1
    AltoTensor<RLIType> * ATR;
#endif
    FType ** ofibs = NULL;
    if (timeslice_limit > 1)
      init_salto(X, &AT, omp_get_max_threads(), -1); // skip vectorized streaming mode optimizations
    else
      init_salto(X, &AT, omp_get_max_threads(), streaming_mode);

#if ADAPT_LIT == 1
    if (timeslice_limit > 1)
      init_salto(X, &ATR, omp_get_max_threads(), -1); // skip vectorized streaming mode optimizations
    else 
      init_salto(X, &ATR, omp_get_max_threads(), streaming_mode);
#endif

    // Step 2. Prepare variables and load first time batch
    // Instantiate kruskal models
    KruskalModel * M; // Keeps track of current factor matrices
    KruskalModel * prev_M; // Keeps track of previous factor matrices

    Matrix ** grams;
    // concatencated s_t's
    StreamMatrix * global_time = new StreamMatrix(rank);
    SparseCPGrams * scpgrams;

    // Set up workspace
    admm_ws * ws = admm_ws_init(nmodes);

    if (use_spcpstream) {
        // Hypersparse ALS specific
        scpgrams = InitSparseCPGrams(nmodes, rank);
    }

    fprintf(stderr, "==== Executing %s (%s) ====\n", use_spcpstream ? "spCPSTREAM" : "CPSTREAM",  "ALTO"
    );

    int it = 0;
    while (!sst.last_batch()) {
#if TIMER == 1
        //BEGIN_TIMER(&ts);
        wtime = omp_get_wtime();
#endif
        SparseTensor * t_batch = sst.next_dynamic_batch(nnz_threshold, timeslice_limit);

#if DEBUG == 0
        PrintTensorInfo(rank, max_iters, t_batch);
#endif
        if (it == 0) {
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &M);
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &prev_M);
            KruskalModelRandomInit(M, (unsigned int)seed);
            KruskalModelZeroInit(prev_M);

            // Set factor matrix for streaming mode as 0
            memset(M->U[streaming_mode], 0,
              sizeof(FType) * t_batch->dims[streaming_mode] * rank);

            init_grams(&grams, M);

            for (int i = 0; i < rank * rank; ++i) {
                grams[streaming_mode]->vals[i] = 0.0;
            }
        } else {
            GrowKruskalModel(t_batch->dims, M, FILL_RANDOM, seed); // Expands the kruskal model to accomodate new dimensions
            GrowKruskalModel(t_batch->dims, prev_M, FILL_ZEROS, seed); // Expands the kruskal model to accomodate new dimensions

            for (int j = 0; j < M->mode; ++j) {
                if (j != streaming_mode) {
                    update_gram(grams[j], M, j);
                }
            }
        }

        /* Start: Setup workspace */
        int max_dim = 0;
        for (int m = 0; m < nmodes; ++m) {
          ws->duals[m] = zero_mat(t_batch->dims[m], rank);
          if (t_batch->dims[m] > max_dim) {
            max_dim = t_batch->dims[m];
          }
        }
        ws->mttkrp_buf = zero_mat(max_dim, rank);
        ws->auxil = zero_mat(max_dim, rank);
        ws->mat_init = zero_mat(max_dim, rank);
        /* End: Setup workspace */

#if TIMER == 1
        t_preprocess = omp_get_wtime() - wtime;
#endif

#if TIMER == 1
        //BEGIN_TIMER(&ts);
        wtime = omp_get_wtime();
#endif
        if (use_spcpstream) {
          spcpstream_alto_iter(
            t_batch, AT, M, prev_M, grams, scpgrams,
            max_iters, epsilon, streaming_mode, it, con, ws, &t_tensor);
        } else {
#if FG_TIMER == 1
          double temp = 0.0;
#endif
          int num_partitions = get_num_ptrn(t_batch->nnz);
#if ADAPT_LIT == 1
          int num_bits = get_num_bits(t_batch->dims, nmodes);
#if DEBUG == 1
          fprintf(stderr, "alto_bits=%d, num_partitions=%d\n", num_bits, num_partitions);
#endif
          if (num_bits > ((int)sizeof(RLIType) * 8)) { // full index
#endif
#if FG_TIMER == 1
            temp = omp_get_wtime();
#endif
            update_salto(t_batch, AT, num_partitions);
            // Create local fiber copies
            create_da_mem(-1, rank, AT, &ofibs);
#if FG_TIMER == 1
            t_tensor += omp_get_wtime() - temp;
#endif
            cpstream_alto_iter(
              AT, ofibs, M, prev_M, grams,
              max_iters, epsilon, streaming_mode, it, con, ws, &t_tensor);
#if FG_TIMER == 1
            temp = omp_get_wtime();
#endif
            destroy_da_mem(AT, ofibs, rank, -1);
#if FG_TIMER == 1
            t_tensor += omp_get_wtime() - temp;
#endif
#if ADAPT_LIT == 1
          } else { // reduced index
#if FG_TIMER == 1
            temp = omp_get_wtime();
#endif
            update_salto(t_batch, ATR, num_partitions);
            // Create local fiber copies
            create_da_mem(-1, rank, ATR, &ofibs);
#if FG_TIMER == 1
            t_tensor += omp_get_wtime() - temp;
#endif
            cpstream_alto_iter(
              ATR, ofibs, M, prev_M, grams,
              max_iters, epsilon, streaming_mode, it, con, ws, &t_tensor);
#if FG_TIMER == 1
            temp = omp_get_wtime();
#endif
            destroy_da_mem(ATR, ofibs, rank, -1);
#if FG_TIMER == 1
            t_tensor += omp_get_wtime() - temp;
#endif
          } // reduced index
#endif
        } // cpstream

#if TIMER == 1
        //END_TIMER(&te);
        //ELAPSED_TIME(ts, te, &t_iter);
        t_iter = omp_get_wtime() - wtime;
#endif
        // ++it; // increment of it has to precede global_time memcpy
        it += t_batch->dims[streaming_mode];

#if TIMER == 1
        //BEGIN_TIMER(&ts);
        wtime = omp_get_wtime();
#endif
        // Copy M -> prev_M
        CopyKruskalModel(prev_M, M);
        global_time->grow_zero(it);
        memcpy(
          &(global_time->mat()->vals[rank * (it-t_batch->dims[streaming_mode])]),
          M->U[streaming_mode], M->dims[streaming_mode] * rank * sizeof(FType));
#if TIMER == 1
        //END_TIMER(&te);
        //ELAPSED_TIME(ts, te, &t_postprocess);
        t_postprocess = omp_get_wtime() - wtime;
#endif

#if CHECK_FIT == 1
        {
            KruskalModel * factored;
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &factored);
            // Over write streaming mode -- hacky
            free(factored->U[streaming_mode]);
            factored->dims[streaming_mode] = it;
            factored->U[streaming_mode] = (FType *)AlignedMalloc(it * rank * sizeof(FType));

            // Copy factor matrix values
            for (int m = 0; m < X->nmodes; ++m) {
                if (m == streaming_mode) {
                    // Copy from global_time Stream matrix
                    ParMemcpy(factored->U[m], global_time->mat()->vals, it * rank * sizeof(FType));
                }
                else {
                    // Are there cases where X->dims[m] > M->dims[m]?
                    ParMemcpy(factored->U[m], M->U[m], M->dims[m] * rank * sizeof(FType));

                }
            }
            memcpy(factored->lambda, M->lambda, rank * sizeof(FType));

            double local_err   = compute_errorsq(&sst, factored, 1);
            double global_err  = -1.;
            double local10_err = -1.;
            double cpd_err     = -1.;

            if((it > 0) && ((it % CHECK_FIT_INTERVAL == 0) || sst.last_batch())) {

              global_err  = compute_errorsq(&sst, factored, it);
              // local10_err = compute_errorsq(&sst, factored, 10);
              // cpd_err     = compute_cpd_errorsq(&sst, rank, it);

              if(std::isnan(cpd_err)) {
                cpd_err = -1.;
              }
            }

            fprintf(stderr, "batch %5lu: %7lu nnz (%0.5fs) (%0.3e NNZ/s) "
                  "cpd: %+0.5f global: %+0.5f local-1: %+0.5f local-10: %+0.5f\n",
                it, t_batch->nnz, 1.0,
                (double) t_batch->nnz / 1.0,
                cpd_err, global_err, local_err, local10_err);

            DestroyKruskalModel(factored);
        }
#endif
        /* Free ADMM workspace */
        for (int m = 0; m < nmodes; ++m) {
          free_mat(ws->duals[m]);
        }
        free_mat(ws->mttkrp_buf);
        free_mat(ws->auxil);
        free_mat(ws->mat_init);
        /* End: Free ADMM workspace */

        DestroySparseTensor(t_batch);

#if TIMER == 1
    tot_time += t_preprocess + t_iter + t_postprocess;
    tot_iter += t_iter;
    fprintf(stderr, "timing CPSTREAM (#it, pre, iter, post)\n");
    fprintf(stderr, "%d\t%f\t%f\t%f\n", it, t_preprocess, t_iter, t_postprocess);
#endif
    } /* End streaming */

#if TIMER == 1
    wtime = omp_get_wtime();
#endif
    KruskalModel * factored;
    CreateKruskalModel(X->nmodes, X->dims, rank, &factored);

    // Copy factor matrix values
    for (int m = 0; m < X->nmodes; ++m) {
        if (m == streaming_mode) {
            // Copy from global_time Stream matrix
            ParMemcpy(factored->U[m], global_time->mat()->vals, rank * X->dims[m] * sizeof(FType));
        }
        else {
            // Are there cases where X->dims[m] > M->dims[m]?
            ParMemcpy(factored->U[m], M->U[m], M->dims[m] * rank * sizeof(FType));
        }
    }
    memcpy(factored->lambda, M->lambda, rank * sizeof(FType));

    // Print lambda
    // PrintFPMatrix("lambda", M->lambda, 1, rank);
    // PrintKruskalModel(factored);
    double const final_err = cpd_error(X, factored);

#if DEBUG == 1
    // PrintKruskalModel(factored);
    // ExportKruskalModel(factored, "./cpstream_alto");
#endif

#if TIMER == 1
    tot_time += omp_get_wtime() - wtime;
#endif
#if FG_TIMER == 1
    fprintf(stderr, "Total CPSTREAM time (#it, total, iter, tensor)\n");
    fprintf(stderr, "%d\t%f\t%f\t%f\n", it, tot_time, tot_iter, t_tensor);
#else
    fprintf(stderr, "Total CPSTREAM time (#it, total, iter)\n");
    fprintf(stderr, "%d\t%f\t%f\n", it, tot_time, tot_iter);
#endif
    fprintf(stdout, "final fit error: %f\n", final_err * final_err);

    // Clean up
    DestroySparseTensor(X);
    destroy_grams(grams, M);
    DestroyKruskalModel(M);
    DestroyKruskalModel(prev_M);
    DestroyKruskalModel(factored);
      //destroy_da_mem(AT, ofibs, rank, -1);
      destroy_alto(AT);
#if ADAPT_LIT == 1
      destroy_alto(ATR);
#endif

    if (use_spcpstream) {
        DeleteSparseCPGrams(scpgrams, nmodes);
    }
    free(ws);

    delete global_time;
    return;
}

double cpd_error(SparseTensor * tensor, KruskalModel * factored) {
    // set up OpenMP locks
    IType max_mode_len = 0;
    IType min_mode_len = tensor->dims[0];
    IType min_mode_idx = 0;

    for(int i = 0; i < factored->mode; i++) {
        if(max_mode_len < factored->dims[i]) {
            max_mode_len = factored->dims[i];
        }
        if(min_mode_len > factored->dims[i]) {
            min_mode_len = factored->dims[i]; // used to compute mttkrp
            min_mode_idx = i;
        }
    }
    IType nrows = factored->dims[min_mode_idx];
    IType rank = factored->rank;
    omp_lock_t* writelocks = (omp_lock_t*) AlignedMalloc(sizeof(omp_lock_t) *
                                                        max_mode_len);
    assert(writelocks);
    for(IType i = 0; i < max_mode_len; i++) {
        omp_init_lock(&(writelocks[i]));
    }
    // MTTKRP
    // Copy original matrix
    Matrix * smallmat = mat_fillptr(factored->U[min_mode_idx], nrows, rank);
    Matrix * mttkrp = zero_mat(nrows, rank);

    FType * tmp = factored->U[min_mode_idx];
    factored->U[min_mode_idx] = mttkrp->vals;

    mttkrp_par(tensor, factored, min_mode_idx, writelocks);

    mttkrp->vals = factored->U[min_mode_idx];
    factored->U[min_mode_idx] = tmp;

    // FType * mttkrp = (FType *) malloc(nrows * rank * sizeof(FType));
    // ParMemcpy(mttkrp, factored->U[min_mode_idx], nrows * rank * sizeof(FType));

    // Restore factored->U[min_mode_idx]
    // ParMemcpy(factored->U[min_mode_idx], smallmat->vals, nrows * rank * sizeof(FType));

    for(IType i = 0; i < max_mode_len; i++) {
        omp_destroy_lock(&(writelocks[i]));
    }
    // inner product between tensor and factored
    double inner = 0;
    #pragma omp parallel reduction(+:inner)
    {
      int const tid = omp_get_thread_num();
      FType * accumF = (FType *) malloc(rank * sizeof(*accumF));

      #pragma omp simd
      for(IType r=0; r < rank; ++r) {
        accumF[r] = 0.;
      }

      /* Hadamard product with newest factor and previous MTTKRP */
      #pragma omp for schedule(static)
      for(IType i=0; i < nrows; ++i) {
        FType const * const smallmat_row = &(smallmat->vals[i * rank]);
        FType const * const mttkrp_row = mttkrp->vals + (i*rank);
        #pragma omp simd
        for(IType r=0; r < rank; ++r) {
          accumF[r] += smallmat_row[r] * mttkrp_row[r];
        }
      }

      /* accumulate everything into 'inner' */
      for(IType r=0; r < rank; ++r) {
        // inner += accumF[r] * factored->lambda[r];
        // DELETE LATER -- VERIFICATION
        inner += accumF[r];
        factored->lambda[r] = 1.0;
      }

      free(accumF);
    } /* end omp parallel -- reduce myinner */

    // Compute ttnormsq to later compute fit
    FType Xnormsq = 0.0;
    FType* vals = tensor->vals;
    IType nnz = tensor->nnz;

    #pragma omp parallel for reduction(+:Xnormsq) schedule(static)
    for(IType i = 0; i < nnz; ++i) {
      Xnormsq += vals[i] * vals[i];
    }
    free_mat(mttkrp);
    free_mat(smallmat);
    double const Znormsq = kruskal_norm(factored);
    double const residual = sqrt(Xnormsq + Znormsq - (2 * inner));

    fprintf(stderr, "Xnormsq: %f, Znormsq: %f, inner: %f\n", Xnormsq, Znormsq, inner);
    double const err = residual / sqrt(Xnormsq);
    return err;
}

template <typename LIT>
void spcpstream_alto_iter(SparseTensor* X, AltoTensor<LIT>* at, KruskalModel* M, KruskalModel * prev_M,
    Matrix** grams, SparseCPGrams * scpgrams, int max_iters, double epsilon,
    int streaming_mode, int iter, cpd_constraint * con, admm_ws * ws, double * t_tensor)
{

#if FG_TIMER == 1
    fprintf(stderr,
        "Running Sparse-CP-Stream (%s, iter: %d) with %d max iters and %.2e epsilon\n",
        "ALTO", iter, max_iters, epsilon);

  /* Timing stuff */
  uint64_t ts = 0;
  uint64_t te = 0;

  double t_mttkrp_sm = 0.0;
  double t_mttkrp_om = 0.0;
  double t_bs_sm = 0.0;
  double t_bs_om = 0.0;

  double t_add_historical = 0.0;
  double t_memset = 0.0;

  double t_conv_check = 0.0;
  double t_gram_mat = 0.0;
  double t_norm = 0.0;

  // Needed to measure alto time
  double t_alto = 0.0;

  // spcpstream specific
  double t_row_op = 0.0; // row related operations
  double t_mat_conversion = 0.0; // fm <-> rsp mat conversion op
  double t_upd_fm = 0.0; // update full factor matrix
  /* End - Timing stuff */

  double wtime;
#endif

  /* Unpack stuff */
  // basic params
  int nmodes = X->nmodes;
  IType* dims = X->dims;
  IType rank = M->rank;
  IType nthreads = omp_get_max_threads();

  // Unpack scpgrams;
  Matrix ** c = scpgrams->c;
  Matrix ** h = scpgrams->h;

  Matrix ** c_nz = scpgrams->c_nz;
  Matrix ** h_nz = scpgrams->h_nz;

  Matrix ** c_z = scpgrams->c_z;
  Matrix ** h_z = scpgrams->h_z;

  Matrix ** c_prev = scpgrams->c_prev;
  Matrix ** c_z_prev = scpgrams->c_z_prev;
  Matrix ** c_nz_prev = scpgrams->c_nz_prev;
  /* End: Unpack stuff */

  /* Init variables */
  int num_inner_iter = 0;
  // keep track of the fit for convergence check
  double fit = 0.0, prev_fit = 0.0, delta = 0.0, prev_delta = 0.0;

  FType ** ofibs = NULL;

  Matrix ** Q = (Matrix**) AlignedMalloc(nmodes * sizeof(Matrix*));
  Matrix * Phi = init_mat(rank, rank);
  Matrix * old_gram = zero_mat(rank, rank);

  // Needed to formulate full-sized factor matrix within the convergence loop
  // The zero rows still change inbetween iters due to Q and Phi changing
  RowSparseMatrix ** A_nz = (RowSparseMatrix**) AlignedMalloc(nmodes * sizeof(RowSparseMatrix*));
  RowSparseMatrix ** A_nz_prev = (RowSparseMatrix**) AlignedMalloc(nmodes * sizeof(RowSparseMatrix*));

  FType** nz_factors = (FType**) AlignedMalloc(nmodes * sizeof(FType*));;
  vector<vector<size_t>> z_rows((size_t)nmodes, vector<size_t> (0, 0));
  vector<vector<size_t>> idx((size_t)nmodes, vector<size_t> (0, 0));

  // Q * Phi^-1 is needed to update A_z[m] after inner convergence
  Matrix ** Q_Phi_inv = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  for (int m = 0; m < nmodes; ++m) {
    Q_Phi_inv[m] = init_mat(rank, rank);
    Q[m] = init_mat(rank, rank);
  }
  /* End: Init variables */

  /* Housekeeping - Generic, repetitive code */
  // Lambda scratchpad
  FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
  assert(lambda_sp);
  #pragma omp parallel for schedule(static, 1)
  for (IType t = 0; t < nthreads; ++t) {
      lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
      assert(lambda_sp[t]);
  }
  /* End: Housekeeping */

#if DEBUG == 1
  PrintSparseTensor(X);
#endif

  // ==== Step 0. ==== Normalize factor matrices for first iter
  /*
  for (int m = 0; m < nmodes; ++m) {
      if (m == streaming_mode) continue;
      KruskalModelNorm(M, m, MAT_NORM_2, lambda_sp);
  }
  if (iter == 0) {
      #pragma omp simd
      for (int r = 0; r < rank; ++r) {
          // Just normalize the columns and reset the lambda
          M->lambda[r] = 1.0;
      }
  }
  */

  if(iter == 0) {
      for(IType m=0; m < nmodes; ++m) {
          if(m == streaming_mode) {
              continue;
          }
          KruskalModelNorm(M, m, MAT_NORM_2, lambda_sp);
          update_gram(grams[m], M, m);
      }
#pragma omp simd
      for (int r = 0; r < rank; ++r) {
          // Just normalize the columns and reset the lambda
          M->lambda[r] = 1.0;
      }
  }
  /**
   * Before ALS performs the following
   * 1. computes the nz_rows, z_rows given tensor X and mode m
   * 2. Instantiates A_nz_prev[m]
   * 3. computes c[m] = A[m].T * A[m]
   * 4. computes h[m] = A_prev[m].T * A[m]
   * 5. Instantiates A_nz[m] for all modes
   */
  { // To free locally scoped temp. data
  // Used to store non_zero row informatoin for all modes
  vector<vector<size_t>> nz_rows((size_t)nmodes, vector<size_t> (0, 0));
  vector<vector<size_t>> buckets((size_t)nmodes, vector<size_t> (0, 0));
  // For storing mappings of indices in I to indices in rowind
  vector<vector<int>> ridx((size_t)nmodes, vector<int> (0, 0));

  for (IType m = 0; m < nmodes; ++m) {
#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    // Identify nonzero slices for all modes
    nonzero_slices(X, m, nz_rows[m], z_rows[m], idx[m], ridx[m], buckets[m]);
#if FG_TIMER == 1
    t_row_op += omp_get_wtime() - wtime;
#endif
    size_t nnzr = nz_rows[m].size();
    size_t * rowind = &nz_rows[m][0];

    // temp mat to Use two times for A_nz_prev[m] and c[m]
    Matrix _fm; // temp mat to compute A_nz_prev and c[m]
    mat_hydrate(&_fm, prev_M->U[m], prev_M->dims[m], rank);

#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    A_nz_prev[m] = convert_to_rspmat(&_fm, nnzr, rowind);
#if FG_TIMER == 1
    t_mat_conversion += omp_get_wtime() - wtime;
#endif
    mat_hydrate(&_fm, M->U[m], M->dims[m], rank);
#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    mat_aTa(&_fm, c[m]);
#if FG_TIMER == 1
    t_gram_mat += omp_get_wtime() - wtime;
#endif
#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    matmul(
      prev_M->U[m], true,
      M->U[m], false, h[m]->vals,
      prev_M->dims[m], rank, M->dims[m], rank, 0.0);
#if FG_TIMER == 1
    t_add_historical += omp_get_wtime() - wtime;
#endif

#if FG_TIMER == 1
  wtime = omp_get_wtime();
#endif
    // if (m == streaming_mode) {
    //   A_nz[m] = rspmat_init(1, rank, 1);
    //   A_nz[m]->rowind[0] = 0; // this never changes
    //   #pragma omp simd
    //   for (int r = 0; r < rank; ++r) {
    //     A_nz[m]->mat->vals[r] = 0.0;
    //   }
    // } else {
    {
      Matrix _fm;
      mat_hydrate(&_fm, M->U[m], M->dims[m], rank);
      A_nz[m] = convert_to_rspmat(&_fm, nz_rows[m].size(), &nz_rows[m][0]);
    }
    // }
    nz_factors[m] = A_nz[m]->mat->vals;
#if FG_TIMER == 1
   t_mat_conversion += omp_get_wtime() - wtime;
#endif
  } /* End - Computing c[m], h[m], A_nz[m], A_nz_prev[m] for all modes */

#if FG_TIMER == 1
  wtime = omp_get_wtime();
#endif
  /* Compute c_z_prev using c_prev, c_nz_prev - it > 0 */
  if (iter > 0) {
    for (IType m = 0; m < nmodes; ++m) {
      if (m == streaming_mode) continue;
      Matrix _fm;
      mat_hydrate(&_fm, prev_M->U[m], prev_M->dims[m], rank);
      mataTa_idx_based(&_fm, nz_rows[m], c_nz_prev[m]);
      for (IType i = 0; i < rank * rank; ++i) {
        c_z_prev[m]->vals[i] = c_prev[m]->vals[i] - c_nz_prev[m]->vals[i];
      }
    }
  }
  /* End */
#if FG_TIMER == 1
  t_gram_mat += omp_get_wtime() - wtime;
#endif
  // ALTO format
  int num_partitions = get_num_ptrn(X->nnz);
#if FG_TIMER == 1
  wtime = omp_get_wtime();
#endif
  update_salto_rowsparse(X, at, A_nz, ridx, idx, num_partitions);
  // Create local fiber copies
  create_da_mem(-1, rank, at, &ofibs);
#if FG_TIMER == 1
  t_alto += omp_get_wtime() - wtime;
#endif
} // end initial row_sparse operations

  // ==== Step 4-1. ===== Compute s_t



  // ==== Step 4. ==== Inner iter for-loop
  int tmp_iter = 0; // To log number of iters until convergence
  for (int i = 0; i < max_iters; i++) {
    delta = 0.0; // Reset to 0.0 for every iter

    FType * s_t = A_nz[streaming_mode]->mat->vals;
    memset(s_t, 0, A_nz[streaming_mode]->nnzr * rank * sizeof(FType));

  #if FG_TIMER == 1
    //BEGIN_TIMER(&ts);
    wtime = omp_get_wtime();
  #endif
    // rowsparse_mttkrp_alto_par(streaming_mode, A_nz, ridx, rank, at, NULL, ofibs);
  mttkrp_alto_par(streaming_mode, nz_factors, rank, at, NULL, ofibs, idx);

  #if FG_TIMER == 1
    // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_mttkrp_sm);
    t_mttkrp_sm += omp_get_wtime() - wtime;
  #endif

  #if FG_TIMER == 1
    //BEGIN_TIMER(&ts);
    wtime = omp_get_wtime();
  #endif

    Matrix * Phi_st = hadamard_product(c, nmodes, streaming_mode);
    add_diag(Phi_st, 1e-12);
    pseudo_inverse(Phi_st, A_nz[streaming_mode]->mat);
    free_mat(Phi_st);
  #if FG_TIMER == 1
    // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_bs_sm);
    t_bs_sm += omp_get_wtime() - wtime;
  #endif
    // Update A_nz[streaming_mode] and M->U[streaming_mode]
    // memcpy(M->U[streaming_mode], A_nz[streaming_mode]->mat->vals, rank * sizeof(FType));
  #if DEBUG == 1
    PrintRowSparseMatrix("s_t after solve", A_nz[streaming_mode]);
    exit(0);
  #endif

    // ==== Step 4-2. ==== Compute G_t-1(old_gram), G_t-1 + ssT (grams[streaming_mode])
    // Update gram matrix
    copy_upper_tri(grams[streaming_mode]);
    // Copy newly computed gram matrix G_t to old_gram
    memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

  #if DEBUG == 1
    PrintMatrix("gram mat before updating s_t", grams[streaming_mode]);
  #endif

  #if FG_TIMER == 1
    //BEGIN_TIMER(&ts);
    wtime = omp_get_wtime();
  #endif
    // Accumulate new time slice into temporal Gram matrix
    // Update grams
    for (IType r = 0; r < M->dims[streaming_mode]; ++r) {
      for (int n = 0; n < rank; ++n) {
        for (int m = 0; m < rank; ++m) {
            grams[streaming_mode]->vals[m + n * rank] += s_t[rank * r + m] * s_t[rank * r + n];
        }
      }
    }
  #if FG_TIMER == 1
    // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);
    t_gram_mat += omp_get_wtime() - wtime;
  #endif
  #if DEBUG == 1
    PrintMatrix("gram mat after updating s_t", grams[streaming_mode]);
  #endif
    // Maintainence; updating c and h gram matrices
    // to incorporate latest s_t, this may not be needed
    // If we clean out the Matrix ** grams part
    // This just makes it more explicit
    memcpy(h[streaming_mode]->vals, old_gram->vals, rank * rank * sizeof(FType));
    memcpy(c[streaming_mode]->vals, grams[streaming_mode]->vals, rank * rank * sizeof(FType));

    // ==== Step 4-3. ==== Compute for all other modes
    for(int m = 0; m < X->nmodes; m++) {
      if (m == streaming_mode) continue;

#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // ==== Step 4-3-1. ==== Compute Phi[m], Q[m]
      // h[sm], c[sm] each contains old_gram, old_gram * s*s.T
      mat_form_gram(h, Q[m], nmodes, m);
      mat_form_gram(c, Phi, nmodes, m);
      // Add frob reg so that it can be used throughoutt
      add_diag(Phi, 1e-12);
#if FG_TIMER == 1
      // BEGIN_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);
      t_gram_mat += omp_get_wtime() - wtime;
#endif
#if DEBUG == 1
      memset(_str, 0, 512);
      sprintf(_str, "Before mttkrp: %d, A_nz[%d]", iter, m);
      PrintRowSparseMatrix(_str, A_nz[m]);
#endif

#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // ==== Step 4-3-2. ==== Compute rowsparse MTTKRP for mode m

      // The non-ALTO counterpart inits A_nz[m] within the mttkrp func.
      // The ALTO version does it here
      // #pragma omp parallel for schedule(static)
      // for (int jj = 0; jj < A_nz[m]->nnzr * rank; ++jj) {
      //   A_nz[m]->mat->vals[jj] = 0.0;
      // }
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < A_nz[m]->nnzr * rank; ++i) {
        nz_factors[m][i] = 0.0;
      }
#if FG_TIMER == 1
      t_memset += omp_get_wtime() - wtime;
#endif

#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // Updates mttkrp result to directly to A_nz
      // rowsparse_mttkrp_alto_par(m, A_nz, ridx, rank, at, NULL, ofibs);
      mttkrp_alto_par(m, nz_factors, rank, at, NULL, ofibs, idx);

#if FG_TIMER == 1
      // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_mttkrp_om);
      t_mttkrp_om += omp_get_wtime() - wtime;
#endif

#if DEBUG == 1
      memset(_str, 0, 512);
      sprintf(_str, "After mttkrp: %d, A_nz[%d]", iter, m);
      PrintRowSparseMatrix(_str, A_nz[m]);

      for (int mm = 0; mm < nmodes; ++mm) {
        fprintf(stderr, "mode: %d\n", mm);
        PrintMatrix("c", c[mm]);
      }
      PrintMatrix("Phi matrix", Phi);
#endif

      // fprintf(stderr, "add_hist\n");
#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // ==== Step 4-3-3 ==== Add historical (mttkrp_res + A_nz_prev[m] * Q[m])
      // TODO: ????? Do we update A_nz_prev between iters or is A_nz_prev static for
      // current time slice
      RowSparseMatrix * A_nz_prev_Q = rsp_mat_mul(A_nz_prev[m], Q[m]);
      rsp_mat_add(A_nz[m], A_nz_prev_Q);
#if FG_TIMER == 1
      // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_add_historical);
      t_add_historical += omp_get_wtime() - wtime;
#endif

#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // ==== Step 4-3-4 Solve for A_nz ====
      Matrix * _Phi = init_mat(rank, rank);
      memcpy(_Phi->vals, Phi->vals, rank * rank * sizeof(FType));
      pseudo_inverse(_Phi, A_nz[m]->mat);

#if FG_TIMER == 1
      // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_bs_om);
      t_bs_om += omp_get_wtime() - wtime;
#endif

#if DEBUG == 1
      PrintRowSparseMatrix("After solve", A_nz[m]);
#endif

#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // ==== Step 4-3-6 Update h_nz[m] c_nz[m] ====
      rsp_mataTb(A_nz[m], A_nz[m], c_nz[m]);
      rsp_mataTb(A_nz_prev[m], A_nz[m], h_nz[m]);
#if FG_TIMER == 1
      // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);
      t_gram_mat += omp_get_wtime() - wtime;
#endif
      // fprintf(stderr, "gram_mat\n");
      // ==== Step 4-3-7 Solve for zero slices (h_z[m], c_z[m]) ====
      memcpy(_Phi->vals, Phi->vals, rank * rank * sizeof(FType));
      memcpy(Q_Phi_inv[m]->vals, Q[m]->vals, rank * rank * sizeof(FType));

#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      pseudo_inverse(_Phi, Q_Phi_inv[m]); // _Q now is Q_Phi_inv
#if FG_TIMER == 1
      // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_bs_om);
      t_bs_om += omp_get_wtime() - wtime;
#endif
#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      matmul(c_z_prev[m], false, Q_Phi_inv[m], false, h_z[m], 0.0);
      matmul(Q_Phi_inv[m], true, h_z[m], false, c_z[m], 0.0);

      // ==== Step 4-3-8 Update h[m], c[m] ====
      for (int i = 0; i < rank * rank; ++i) {
        c[m]->vals[i] = c_nz[m]->vals[i] + c_z[m]->vals[i];
        h[m]->vals[i] = h_nz[m]->vals[i] + h_z[m]->vals[i];
      }
#if FG_TIMER == 1
      // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);
      t_gram_mat += omp_get_wtime() - wtime;
#endif
#if SKIP_TEST == 1
#else
#if FG_TIMER == 1
      // BEGIN_TIMER(&ts);
      wtime = omp_get_wtime();
#endif
      // ==== Step 4-3-9 Compute delta ====
      FType tr_c = mat_trace(c[m]);
      FType tr_h = mat_trace(h[m]);
      FType tr_c_prev = mat_trace(c_prev[m]);

      delta += sqrt(fabs(((tr_c + tr_c_prev - 2.0 * tr_h) / (tr_c + 1e-12))));
#if FG_TIMER == 1
      // END_TIMER(&te);
      // AGG_ELAPSED_TIME(ts, te, &t_conv_check);
      t_conv_check += omp_get_wtime() - wtime;
#endif
#endif
      free_mat(_Phi);
      rspmat_free(A_nz_prev_Q);
    } // for each non-streaming mode

    // May compute fit here - probably not the best idea due to being slow

    tmp_iter = i;
    // fprintf(stderr, "it: %d delta: %e prev_delta: %e (%e diff)\n", i, delta, prev_delta, fabs(delta - prev_delta));
#if SKIP_TEST == 1
        //prev_delta = delta;
        //prev_delta_diff = delta_diff;
#else
    if ((i > 0) && fabs(prev_delta - delta) < epsilon) {
      prev_delta = 0.0;
      break;
    } else {
      prev_delta = delta;
    }
#endif
  } // end for loop: max_iters

  num_inner_iter += tmp_iter; // track number of iters per time-slice

  // ==== Step 5. ==== Update factor matrices M->U[m]
  for (int m = 0; m < nmodes; ++m) {
    // if (m == streaming_mode) continue;
    size_t nzr = z_rows[m].size();

#if FG_TIMER == 1
    // BEGIN_TIMER(&ts);
    wtime = omp_get_wtime();
#endif
    Matrix _fm;
    mat_hydrate(&_fm, prev_M->U[m], prev_M->dims[m], rank);
    RowSparseMatrix * prev_A_z = convert_to_rspmat(&_fm, nzr, &z_rows[m][0]);
    RowSparseMatrix * prev_A_z_Q_Phi_inv = rsp_mat_mul(prev_A_z, Q_Phi_inv[m]);
#if FG_TIMER == 1
    t_mat_conversion += omp_get_wtime() - wtime;
#endif

#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nzr; ++i) {
      size_t ridx = z_rows[m].at(i);
      memcpy(
        &(M->U[m][ridx * rank]),
        &(prev_A_z_Q_Phi_inv->mat->vals[i * rank]),
        sizeof(FType) * rank);
    }
    rspmat_free(prev_A_z_Q_Phi_inv);
    rspmat_free(prev_A_z);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < A_nz[m]->nnzr; ++i) {
      size_t ridx = A_nz[m]->rowind[i];
      memcpy(
        &(M->U[m][ridx * rank]),
        &(A_nz[m]->mat->vals[i * rank]),
        sizeof(FType) * rank);
    }
#if FG_TIMER == 1
    // END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_upd_fm);
    t_upd_fm += omp_get_wtime() - wtime;
#endif
  }

  // ==== Step 6. ==== Housekeeping...?
  // ==== Step 6. ==== Apply forgetting factor
#if COMPOUND_FORGETTING_FACTOR == 1
  // fprintf(stderr, "==== APPLYING FORGETTING FACTOR: %f ====\n", std::pow(FORGETTING_FACTOR, M->dims[streaming_mode]));
  for (IType x = 0; x < rank * rank; ++x) {
    grams[streaming_mode]->vals[x] *= std::pow(FORGETTING_FACTOR, M->dims[streaming_mode]);
  }
#else
  // fprintf(stderr, "==== APPLYING FORGETTING FACTOR: %f ====\n", FORGETTING_FACTOR);
  for (IType x = 0; x < rank * rank; ++x) {
    grams[streaming_mode]->vals[x] *= FORGETTING_FACTOR;
  }
#endif

  for (IType m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) continue;
    // Copy all c's to prev_c's
    ParMemcpy(c_prev[m]->vals, c[m]->vals, rank * rank * sizeof(FType));
    ParMemcpy(c_nz_prev[m]->vals, c_nz[m]->vals, rank * rank * sizeof(FType));
  }
  // Apply forgetting factor ..?
  // ==== Step 7. Cleaning up ====
  for (int m = 0; m < nmodes; ++m) {
    free_mat(Q_Phi_inv[m]);
    rspmat_free(A_nz[m]);
    rspmat_free(A_nz_prev[m]);
    free_mat(Q[m]);
  }
  free(nz_factors);
  free(Q_Phi_inv);
  free(A_nz);
  free(A_nz_prev);
  free(Q);

  free_mat(Phi);
  free_mat(old_gram);

  #pragma omp parallel for schedule(static, 1)
  for (IType t = 0; t < nthreads; ++t) {
      free(lambda_sp[t]);
  }
  free(lambda_sp);
  /* End: Cleaning up */
#if FG_TIMER == 1
  wtime = omp_get_wtime();
#endif
  destroy_da_mem(at, ofibs, rank, -1);
#if FG_TIMER == 1
  t_alto += omp_get_wtime() - wtime;
#endif
#if FG_TIMER == 1
// *t_tensor += t_alto + t_mttkrp_sm + t_mttkrp_om;
  fprintf(stderr, "timing SPCPSTREAM-ITER\n");
  fprintf(stderr, "#ts\t#nnz\t#it\talto\tmttkrp_sm\tbs_sm\tmemset\tmttkrp_om\thist\tbs_om\tupd_gram\tconv_check\trow op\tmat conv\tupd fm\n");
  fprintf(stderr, "%llu\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
      iter+M->dims[streaming_mode], X->nnz, num_inner_iter+1,
      t_alto, t_mttkrp_sm, t_bs_sm,
      t_memset, t_mttkrp_om, t_add_historical,
      t_bs_om, t_gram_mat, t_conv_check,
      t_row_op, t_mat_conversion, t_upd_fm);
      *t_tensor += t_alto + t_mttkrp_sm + t_mttkrp_om;
#endif

  // PrintFPMatrix("lambda", M->lambda, 1, rank);
}


SparseCPGrams * InitSparseCPGrams(IType nmodes, IType rank) {
  SparseCPGrams * grams = (SparseCPGrams *) AlignedMalloc(sizeof(SparseCPGrams));

  assert(grams);

  grams->c_nz_prev = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->c_z_prev = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->c_nz = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->c_z = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->h_nz = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->h_z = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->c = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->h = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->c_prev = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  for (IType m = 0; m < nmodes; ++m) {
    grams->c_nz_prev[m] = zero_mat(rank, rank);
    grams->c_z_prev[m] = zero_mat(rank, rank);

    grams->c_nz[m] = zero_mat(rank, rank);
    grams->c_z[m] = zero_mat(rank, rank);

    grams->h_nz[m] = zero_mat(rank, rank);
    grams->h_z[m] = zero_mat(rank, rank);

    grams->c[m] = zero_mat(rank, rank);
    grams->h[m] = zero_mat(rank, rank);

    grams->c_prev[m] = zero_mat(rank, rank);
  }

  return grams;
}

// add reverse idx
void nonzero_slices(
    SparseTensor * const tt, const IType mode,
    vector<size_t> &nz_rows,
    vector<size_t> &z_rows,
    vector<size_t> &idx,
    vector<int> &ridx,
    vector<size_t> &buckets)
{
  //TODO: refactor
  idxsort_hist(tt, mode, idx, buckets);
  size_t num_bins = buckets.size() - 1;
  nz_rows.resize(num_bins);

  #pragma omp parallel for schedule(static)
  for (IType i = 0; i < num_bins; i++) {
    //nz_rows.push_back(tt->cidx[mode][idx[buckets[i]]]);
    nz_rows[i] = tt->cidx[mode][idx[buckets[i]]];
  }
  // Create array for reverse indices
  // We traverse through all rows i
  // if it is a non zero row then add i to ridx array
  // if not, push value -1, which means invalid
  // For example if I = 10: [0, 1, 2, 3, 4, 5, ... 9] and non zero rows are [2, 4, 5]
  // then ridx would have [-1, -1, 0, -1, 1, 2, -1, ...]
  IType _ptr = 0;
  for (IType i = 0; i < tt->dims[mode]; i++) {
    if (nz_rows[_ptr] == i) {
      ridx.push_back(_ptr);
      _ptr++;
    } else {
      ridx.push_back(-1);
      z_rows.push_back(i);
    }
  }
}

void DeleteSparseCPGrams(SparseCPGrams * grams, IType nmodes) {
  for (IType m = 0; m < nmodes; ++m) {
    free_mat(grams->c_nz_prev[m]);
    free_mat(grams->c_z_prev[m]);

    free_mat(grams->c_nz[m]);
    free_mat(grams->c_z[m]);

    free_mat(grams->h_nz[m]);
    free_mat(grams->h_z[m]);

    free_mat(grams->c[m]);
    free_mat(grams->h[m]);

    free_mat(grams->c_prev[m]);
  }

  free(grams->c_nz_prev);
  free(grams->c_z_prev);

  free(grams->c_nz);
  free(grams->c_z);

  free(grams->h_nz);
  free(grams->h_z);

  free(grams->c);
  free(grams->h);

  free(grams->c_prev);

  free(grams);
  return;
}

template <typename LIT>
void cpstream_alto_iter(AltoTensor<LIT>* at, FType **ofibs, KruskalModel* M, KruskalModel * prev_M,
    Matrix** grams, int max_iters, double epsilon,
    int streaming_mode, int iter, cpd_constraint * con, admm_ws * ws, double * t_tensor)
{
    std::vector<std::vector<size_t>> oidx;
#if FG_TIMER == 1
    fprintf(stderr,
        "Running CP-Stream (%s, iter: %d) with %d max iters and %.2e epsilon\n",
        "ALTO", iter, max_iters, epsilon);

    // Timing stuff
    uint64_t ts = 0;
    uint64_t te = 0;

    double t_mttkrp_sm = 0.0;
    double t_mttkrp_om = 0.0;
    double t_bs_sm = 0.0;
    double t_bs_om = 0.0;

    double t_add_historical = 0.0;
    double t_memset = 0.0;

    double t_conv_check = 0.0;
    double t_gram_mat = 0.0;
    double t_norm = 0.0;

    // Needed to measure alto time
    double t_alto = 0.0;

    double wtime;
#endif

    int num_inner_iter = 0;

    int nmodes = at->nmode;
    IType* dims = at->dims;
    IType rank = M->rank;

    IType nthreads = omp_get_max_threads();

    // Lambda scratchpad
    FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
    assert(lambda_sp);
    #pragma omp parallel for schedule(static, 1)
    for (IType t = 0; t < nthreads; ++t) {
        lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
        assert(lambda_sp[t]);
    }

    if(iter == 0) {
        for(IType m=0; m < nmodes; ++m) {
            if(m == streaming_mode) {
                continue;
            }
            KruskalModelNorm(M, m, MAT_NORM_2, lambda_sp);
            update_gram(grams[m], M, m);
        }
#pragma omp simd
        for (int r = 0; r < rank; ++r) {
            // Just normalize the columns and reset the lambda
            M->lambda[r] = 1.0;
        }
    }

    Matrix * old_gram = zero_mat(rank, rank);

    // keep track of delta for convergence check
    double delta = 0.0, prev_delta = 0.0;
    int tmp_iter = 0;

    for(int i = 0; i < max_iters; i++) {
        delta = 0.0;

    // Solve for time mode (s_t)
    memset(M->U[streaming_mode], 0, sizeof(FType) * rank * M->dims[streaming_mode]);

#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    //mttkrp_alto(streaming_mode, M->U, rank, at);
    mttkrp_alto_par(streaming_mode, M->U, rank, at, NULL, ofibs, oidx);
#if FG_TIMER == 1
    t_mttkrp_sm += omp_get_wtime() - wtime;
#endif

#if DEBUG == 1
    PrintFPMatrix("mttkrp before s_t", M->U[streaming_mode], M->dims[streaming_mode], rank);
    exit(1);
#endif

#if FG_TIMER == 1
    wtime = omp_get_wtime();
#endif
    Matrix fm;
    mat_hydrate(&fm, M->U[streaming_mode], M->dims[streaming_mode], rank);

    // Init gram matrix aTa for all other modes
    Matrix * Phi = hadamard_product(grams, nmodes, streaming_mode);
    add_diag(Phi, 1e-12);
    pseudo_inverse(Phi, &fm);
    free_mat(Phi);

#if FG_TIMER == 1
    t_bs_sm += omp_get_wtime() - wtime;
#endif

#if DEBUG == 1
    PrintFPMatrix("s_t: after solve", M->U[streaming_mode], M->dims[streaming_mode], rank);
    // exit(1);
#endif

    copy_upper_tri(grams[streaming_mode]);

    // Copy newly computed gram matrix G_t to old_gram
    memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

#if DEBUG == 1
    PrintMatrix("gram mat before updating s_t", grams[streaming_mode]);
#endif
#if FG_TIMER == 1
    //BEGIN_TIMER(&ts);
    wtime = omp_get_wtime();
#endif
    // Accumulate new time slice into temporal Gram matrix
    // Update grams

    for (IType r = 0; r < M->dims[streaming_mode]; ++r) {
      for (int n = 0; n < rank; ++n) {
        for (int m = 0; m < rank; ++m) {
            grams[streaming_mode]->vals[m + n * rank] += M->U[streaming_mode][r * rank + m] * M->U[streaming_mode][r * rank + n];
        }
      }
    }
#if FG_TIMER == 1
    t_gram_mat += omp_get_wtime() - wtime;
#endif
#if DEBUG == 1
    PrintMatrix("gram mat after updating s_t", grams[streaming_mode]);
#endif

        // For all other modes
        for(int j = 0; j < nmodes; j++) {
            if (j == streaming_mode) continue;

#if FG_TIMER == 1
            wtime = omp_get_wtime();
#endif
            ParMemset(ws->mttkrp_buf->vals, 0, sizeof(FType) * dims[j] * rank);
#if FG_TIMER == 1
            t_memset += omp_get_wtime() - wtime;
#endif

#if DEBUG == 1
            char str[512];
            sprintf(str, "M[%d] before mttkrp", j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

#if FG_TIMER == 1
            wtime = omp_get_wtime();
#endif
            FType * tmp_vals = M->U[j];
            M->U[j] = ws->mttkrp_buf->vals;

            mttkrp_alto_par(j, M->U, rank, at, NULL, ofibs, oidx);
            ws->mttkrp_buf->vals = M->U[j];
            M->U[j] = tmp_vals;

#if FG_TIMER == 1
            t_mttkrp_om += omp_get_wtime() - wtime;
#endif
            // add historical
#if FG_TIMER == 1
            wtime = omp_get_wtime();
#endif
            Matrix * historical = zero_mat(rank, rank);
            Matrix * ata_buf = zero_mat(rank, rank);

            // Starts with mu * G_t-1
            memcpy(ata_buf->vals, old_gram->vals, rank * rank * sizeof(*ata_buf->vals));

            // Copmute A_t-1 * A_t for all other modes
            for (int m = 0; m < nmodes; ++m) {
                if ((m == j) || (m == streaming_mode)) {
                    continue;
                }
                // Check previous factor matrix has same dimension size as current factor matrix
                // this should be handled when new tensor is being fed in..
                // assert(prev_M->dims[m] == M->dims[m]);
                matmul(prev_M->U[m], true, M->U[m], false,
                  historical->vals, prev_M->dims[m], rank, M->dims[m], rank, 0.0);

                //#pragma omp parallel for schedule(static)
                for (int x = 0; x < rank * rank; ++x) {
                    ata_buf->vals[x] *= historical->vals[x];
                }
            }

#if DEBUG == 1
            sprintf(str, "ata buf for M[%d]", j);
            PrintMatrix(str, ata_buf);
            // END: Updating ata_buf (i.e. aTa matrices for all factor matrices)

            // A(n) (ata_buf)
            memset(str, 0, 512);
            sprintf(str, "prev_M for mode %d", j);
            PrintFPMatrix(str, prev_M->U[j], prev_M->dims[j], rank);
            memset(str, 0, 512);

            sprintf(str, "mttkrp part for mode %d", j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

            matmul(prev_M->U[j], false, ata_buf->vals, false,
              ws->mttkrp_buf->vals, prev_M->dims[j], rank, rank, rank, 1.0);
#if FG_TIMER == 1
            t_add_historical += omp_get_wtime() - wtime;
#endif

            free_mat(ata_buf);
            free_mat(historical);

#if DEBUG == 1
            sprintf(str, "after add historical for for mode %d", j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

#if FG_TIMER == 1
            wtime = omp_get_wtime();
#endif
            Matrix fm;
            mat_hydrate(&fm, M->U[j], M->dims[j], rank);

            admm_cf(j, &fm, grams, NULL, con, ws);

#if FG_TIMER == 1
            t_bs_om += omp_get_wtime() - wtime;
#endif

#if DEBUG == 1
            sprintf(str, "ts: %d, it: %d: updated factor matrix for mode %d", iter, i, j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

#if FG_TIMER == 1
            wtime = omp_get_wtime();
#endif
            update_gram(grams[j], M, j);
#if FG_TIMER == 1
            t_gram_mat += omp_get_wtime() - wtime;
#endif

#if SKIP_TEST == 1
#else
            int factor_mat_size = rank * M->dims[j];
#if FG_TIMER == 1
            wtime = omp_get_wtime();
#endif
            delta += mat_norm_diff(prev_M->U[j], M->U[j], factor_mat_size) / (mat_norm(M->U[j], factor_mat_size) + 1e-12);
#if FG_TIMER == 1
            t_conv_check += omp_get_wtime() - wtime;
#endif
#endif
        } // for each mode
        // PrintKruskalModel(M);
#if DEBUG == 1
        PrintKruskalModel(M);
#endif
        tmp_iter = i;
        // fprintf(stderr, "it: %d delta: %e prev_delta: %e (%e diff)\n", i, delta, prev_delta, fabs(delta - prev_delta));
#if SKIP_TEST == 1
        // prev_delta = delta;
        // prev_delta_diff = delta_diff;
#else
        if ((i > 0) && fabs(prev_delta - delta) < epsilon) {
          prev_delta = 0.0;
          break;
        }
#endif
        prev_delta = delta;
    } // for max_iters
    // if (iter == 1) exit(1);
    num_inner_iter += tmp_iter;

    // incorporate forgetting factor
#if COMPOUND_FORGETTING_FACTOR == 1
    // fprintf(stderr, "==== APPLYING FORGETTING FACTOR: %f ====\n", std::pow(FORGETTING_FACTOR, M->dims[streaming_mode]));
    for (IType x = 0; x < rank * rank; ++x) {
      grams[streaming_mode]->vals[x] *= std::pow(FORGETTING_FACTOR, M->dims[streaming_mode]);
    }
#else
    // fprintf(stderr, "==== APPLYING FORGETTING FACTOR: %f ====\n", FORGETTING_FACTOR);
    for (IType x = 0; x < rank * rank; ++x) {
      grams[streaming_mode]->vals[x] *= FORGETTING_FACTOR;
    }
#endif

    free_mat(old_gram);

    // cleanup
    #pragma omp parallel for schedule(static, 1)
    for (IType t = 0; t < nthreads; ++t) {
        free(lambda_sp[t]);
    }
    free(lambda_sp);

#if FG_TIMER == 1
    fprintf(stderr, "timing CPSTREAM-ITER\n");
    fprintf(stderr, "#ts\t#nnz\t#it\talto\tmttkrp_sm\tbs_sm\tmem set\tmttkrp_om\thist\tbs_om\tupd_gram\tconv_check\n");
    fprintf(stderr, "%llu\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
        iter+M->dims[streaming_mode], at->nnz, num_inner_iter+1,
        t_alto, t_mttkrp_sm, t_bs_sm,
        t_memset, t_mttkrp_om, t_add_historical,
        t_bs_om, t_gram_mat, t_conv_check);
        *t_tensor += t_alto + t_mttkrp_sm + t_mttkrp_om;
#endif
}

double compute_errorsq(
    StreamingSparseTensor * sst,
    KruskalModel * M,
    IType previous)
{
    KruskalModel * cpd = get_prev_kruskal(M, sst->_stream_mode, previous);

    SparseTensor * prev_tensor = sst->stream_prev(previous);
    double err = cpd_error(prev_tensor, cpd);
    DestroySparseTensor(prev_tensor);
    DestroyKruskalModel(cpd);
    return err * err;
}