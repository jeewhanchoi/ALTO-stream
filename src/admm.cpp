#include "admm.hpp"

// #define DIFFERENT_CHUNK_SIZES 1

admm_ws * admm_ws_init(int nmodes) {
    admm_ws * ws = (admm_ws *) malloc(sizeof(*ws));

    ws->nmodes = nmodes;
    ws->mttkrp_buf = NULL;
    ws->auxil = NULL;
    for (int m = 0; m < nmodes; ++m) {
        ws->duals[m] = NULL;
    }
    ws->mat_init = NULL;

    return ws;
}

#if DIFFERENT_CHUNK_SIZES == 1

FType admm_cf(   
  IType mode,
  Matrix * mat,
  Matrix ** aTa,
  FType * column_weights,
  cpd_constraint * con,
  admm_ws * ws,
  IType chunk_size) {
    FType avg_num_iters = 0.0;

    avg_num_iters = admm_cf_var_cs(
      mode,
      mat,
      aTa,
      column_weights,
      con,
      ws,
      chunk_size
    );

    return avg_num_iters;
}

#else

FType admm_cf(   
  IType mode,
  Matrix * mat,
  Matrix ** aTa,
  FType * column_weights,
  cpd_constraint * con,
  admm_ws * ws) {

  FType avg_num_iters = 0.0;

  avg_num_iters = admm_cf_(
      mode,
      mat,
      aTa,
      column_weights,
      con,
      ws
  );

  return avg_num_iters;
}

#endif // End DIFFERENT_CHUNK_SIZES


// Implement admm that consumes non zero rows only 
// Do zero rows (low signal rows) need admm iteration???
// This function contains the original
// pseudo_inverse_stream implementation
// However, it also adds constraint functionality
FType admm(
    IType mode,
    Matrix * mat, // kruskal model that contains all the factor matrics  do we need all the other modes?
    Matrix ** aTa, // gram matrices 
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
) {
  // Unpack variables
  Matrix * mttkrp_buf = ws->mttkrp_buf;
  int nmodes = ws->nmodes;
  
  IType I = mat->I;
  IType rank = mat->J;

  // compute gram matrix
  Matrix * Phi = hadamard_product(aTa, nmodes, mode);

  if (con->solve_type == UNCONSTRAINED) {
    // Apply frobenius regularization (1e-12)
    // This helps stablity 
    for (int i = 0; i < Phi->I; ++i) {
      Phi->vals[i * Phi->I + i] += 1e-12;
    }
    pseudo_inverse(Phi, mttkrp_buf);
    // Copy back to factor matrix
    ParMemcpy(mat->vals, mttkrp_buf->vals, sizeof(FType) * rank * I);
    return 0.0;
  }

  // Constrained version
  FType rho = mat_trace(Phi) / (FType) rank;

  // printf("\n\n === mode: %d, rho: %f ===\n\n", mode, rho);
  // Debug
  for (int i = 0; i < rank; ++i) {
    Phi->vals[i * rank + i] += rho;
    // Phi->vals[i * rank + i] += 1e-12;
  }

  // Perform cholesky only once
  // printf("before cholesky\n");
  // PrintMatrix("Phi matrix", Phi);

  // printf("before cholesky, rho: %f\n", rho);
  mat_cholesky(Phi);
  // printf("after cholesky\n");
  // printf("after cholesky\n");

  // Step 1. Set up matrix values used for admm
  FType * const auxil = ws->auxil->vals;
  FType * const mttkrp = mttkrp_buf->vals;
  FType * const dual = ws->duals[mode]->vals;
  FType * const mat_init = ws->mat_init->vals;
  Matrix * const factor_matrix = mat;

  FType * const primal = factor_matrix->vals;
  // printf("==== I: %d, J: %d ====\n\n", factor_matrix->I, factor_matrix->J);

  size_t N = I * rank;
  size_t bytes = N * sizeof(FType);

  // Set up norm
  FType primal_norm = 0.0;
  FType dual_norm = 0.0;
  FType primal_residual = 0.0;
  FType dual_residual = 0.0;

  // ADMM subroutine
  IType it;

  for (it = 0; it < 25; ++it) {
    memcpy(mat_init, primal, bytes);
    // Setup auxiliary
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
      auxil[i] = mttkrp[i] + rho * (primal[i] + dual[i]);
    }

    // Cholesky solve
    mat_cholesky_solve(Phi, ws->auxil);

    // Setup proximity
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
      primal[i] = auxil[i] - dual[i];
    }

    // PrintMatrix("factor matrix after constraint", factor_matrix);
    // Apply Constraints and Regularization
    // PrintFPMatrix("primal before", primal, factor_matrix->I, factor_matrix->J);
    con->func(primal, factor_matrix->I, factor_matrix->J);
    // PrintFPMatrix("primal after", primal, factor_matrix->I, factor_matrix->J);
    // Update dual: U += (primal - auxil)
    FType dual_norm = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:dual_norm)
    for (int i = 0; i < N; ++i) {
      dual[i] += primal[i] - auxil[i];
      dual_norm += dual[i] * dual[i];
    }

    int nrows = factor_matrix->I;
    int ncols = factor_matrix->J;

    primal_norm = 0.0;
    primal_residual = 0.0;
    dual_residual = 0.0;

    // Check ADMM convergence, calc residual
    // We need primal_norm, primal_residual, dual_residual
    #pragma omp parallel for reduction(+:primal_norm, primal_residual, dual_residual)
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        int index = j + (i * ncols);
        FType p_diff = primal[index] - auxil[index];
        FType d_diff = primal[index] - mat_init[index];

        primal_norm += primal[index] * primal[index];
        primal_residual += p_diff * p_diff;
        dual_residual += d_diff * d_diff;
      }
    }

    // #pragma omp barrier

    // fprintf(stderr, "p_res, p_norm, d_res, d_nrom, %f, %f, %f, %f\n", primal_residual, primal_norm, dual_residual, dual_norm);
    // Converged ?
    float tolerence = 1e-6;
    if ((primal_residual <= tolerence * primal_norm) && (dual_residual <= tolerence * dual_norm)) {
      ++it;
      break;
    }
  }
  free_mat(Phi);
  //fprintf(stderr, "ADMM (nnzr: %llu, rho: %f): num of iteration: %llu\n", mat->I, rho, it);
  return it;
}

// The non max col norm version
FType admm_cf_(
    IType mode,
    Matrix * mat, // factor matrix
    Matrix ** aTa, 
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
) {
  // Unpack matrices used in ADMM
  Matrix * primal_mat = mat;
  Matrix * auxil_mat = ws->auxil; 
  Matrix * dual_mat = ws->duals[mode];
  Matrix * mttkrp_buf = ws->mttkrp_buf;
  Matrix * init_buf = ws->mat_init;
  
  IType I = mat->I;
  IType rank = mat->J;
  IType nmodes = ws->nmodes;

  IType niter;

  Matrix * Phi = hadamard_product(aTa, nmodes, mode);

  if (con->solve_type == UNCONSTRAINED) {
    // Apply frobenius regularization (1e-12)
    // This helps stablity 
    //#pragma omp parallel for
    for (int i = 0; i < Phi->I; ++i) {
      Phi->vals[i * Phi->I + i] += 1e-12;
    }
    pseudo_inverse(Phi, mttkrp_buf);
    // PrintMatrix("mttkrp buf s_t: after solve", mttkrp_buf);

    // Copy back to factor matrix
    ParMemcpy(mat->vals, mttkrp_buf->vals, sizeof(FType) * rank * I);
    return 0.0;
  }

  FType rho = mat_trace(Phi) / (FType) rank;

  // printf("\n\n === mode: %d, rho: %f ===\n\n", mode, rho);
  // Debug
  for (int i = 0; i < rank; ++i) {
    Phi->vals[i * rank + i] += rho;
    // Phi->vals[i * rank + i] += 1e-12;
  }
  // PrintMatrix("Phi", Phi);

  mat_cholesky(Phi);
  // PrintMatrix("Phi", Phi);
  /* for checking convergence */
  FType p_norm = 0.;
  FType d_norm = 0.;
  FType p_res  = 0.;
  FType d_res  = 0.;

  // IType chunk_size = 4;
  // IType chunk_size = 256;
  IType chunk_size = 64;
  IType num_chunks =  (primal_mat->I / chunk_size);

  if(primal_mat->I % chunk_size > 0) {
    ++num_chunks;
  }

  // num_chunks = 1; // set for debugging
  #pragma omp parallel shared(p_norm,d_norm,p_res,d_res,niter)
  {
    #pragma omp for
    for(IType c=0; c < num_chunks; ++c) {
      IType const start = c * chunk_size;
      IType const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
      IType const offset = start * rank;
      IType const nrows = stop - start;
      IType const ncols = rank;

      /* extract all the workspaces per chunk */
      FType * primal = primal_mat->vals + offset;
      FType * auxil = auxil_mat->vals + offset;
      FType * dual = dual_mat->vals + offset;
      FType * mttkrp = mttkrp_buf->vals + offset;
      FType * init = init_buf->vals + offset;

      Matrix auxil_chunk_mat;

      mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);
      // row-wise/vector-wise fused formation of rhs
      #pragma simd
      #pragma vector aligned
      for (IType idx = 0; idx < ncols*nrows; ++idx) {
        auxil[idx] = mttkrp[idx] + rho*(primal[idx] + dual[idx]);
      }

      // chunk solve chol
      // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
      mat_cholesky_solve(Phi, &auxil_chunk_mat);

      // form prox and compute new norm
      for (IType i = 0; i < nrows; ++i) {
        for (IType j = 0; j < ncols; ++j) {
          IType idx = j + i*ncols;
          init[idx] = primal[idx];
          FType x = auxil[idx] - dual[idx];
          if (x < 0.0) {
            x = 0.0;
          }
          primal[idx] = x;
        }
      }
    }

    #pragma omp barrier

    IType it;
    int do_break = 0;
    for(it=0; it < 25; ++it) {
      { p_res = 0; d_res = 0; p_norm = 0; d_norm = 0; }

      #pragma omp for reduction(+:p_norm,p_res,d_norm,d_res)
      for(IType c=0; c < num_chunks; ++c) {
        IType const start = c * chunk_size;
        IType const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
        IType const offset = start * rank;
        IType const nrows = stop - start;
        IType const ncols = rank;

        /* extract all the workspaces per chunk */
        FType * const primal = primal_mat->vals + offset;
        FType * const auxil = auxil_mat->vals + offset;
        FType * const dual = dual_mat->vals + offset;
        FType * const mttkrp = mttkrp_buf->vals + offset;
        FType * const init = init_buf->vals + offset;

        Matrix auxil_chunk_mat;
        mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);

        // vectorized loop?
        // form prox and compute new norm
          // TODO:
          // instead do inner loop of some vector blocksize (64 bytes)
          // - duplicate norms to at least (B + ncols)
          // - ncols % blocksize  remainder, how to handle?
          // - block b is element b*B, which is column: b*B % ncols
        const IType cs = nrows*ncols;
        #pragma simd
        #pragma vector aligned
        for (IType idx = 0; idx < cs; ++idx) {

          // compute new primal and dual residual
          FType x = primal[idx];
          FType ddiff = x - init[idx];
          d_res += ddiff*ddiff;

          // update primal norm
          p_norm += x*x;

          // update dual U <- U + (pri - aux)
          FType y =  x - auxil[idx];
          FType di = dual[idx] + y;
          dual[idx] = di;

          // update dual norm and primal residual
          d_norm += di*di;
          p_res += y*y;

          // form next RHS for cholesky
          auxil[idx] = mttkrp[idx] + rho*(x + di);
        }

        // chunk solve chol
        // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
        mat_cholesky_solve(Phi, &auxil_chunk_mat);

        // form prox and compute new norm
        for (IType i = 0; i < nrows; ++i) {
          for (IType j = 0; j < ncols; ++j) {
            IType idx = j + i * ncols;
            FType x = auxil[idx] - dual[idx]; // updating primal (init)
            // printf("x: %f\n", x);
            // Non-neg
            if (x > 0.0) {
              init[idx] = x;
              primal[idx] = x;
            } else {
              init[idx] = 0.0;
              primal[idx] = 0.0;
            }
          }
        }
      }

      #pragma omp barrier

      float tolerence = 1e-6;
      if ((p_res <= tolerence * p_norm) && (d_res <= tolerence * d_norm)) {
        ++it;
        break;
      }

      #pragma omp barrier

    } /* admm iteration */

    #pragma omp master
    {
      niter = it; // niter is the number of iterations for convergence
    }

  } /* omp parallel */
  // fprintf(stderr, "ADMM C&F (nnzr: %llu, rho: %f): num of iteration: %llu\n", mat->I, rho, niter);

  free_mat(Phi);
  return niter;
}

FType admm_cf_var_cs(
    IType mode,
    Matrix * mat, // factor matrix
    Matrix ** aTa, 
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws,
    IType chunk_size
) {
  // Unpack matrices used in ADMM
  Matrix * primal_mat = mat;
  Matrix * auxil_mat = ws->auxil; 
  Matrix * dual_mat = ws->duals[mode];
  Matrix * mttkrp_buf = ws->mttkrp_buf;
  Matrix * init_buf = ws->mat_init;
  
  IType I = mat->I;
  IType rank = mat->J;
  IType nmodes = ws->nmodes;

  IType niter;

  Matrix * Phi = hadamard_product(aTa, nmodes, mode);

  if (con->solve_type == UNCONSTRAINED) {
    // Apply frobenius regularization (1e-12)
    // This helps stablity 
    //#pragma omp parallel for
    for (int i = 0; i < Phi->I; ++i) {
      Phi->vals[i * Phi->I + i] += 1e-12;
    }
    pseudo_inverse(Phi, mttkrp_buf);
    // PrintMatrix("mttkrp buf s_t: after solve", mttkrp_buf);

    // Copy back to factor matrix
    ParMemcpy(mat->vals, mttkrp_buf->vals, sizeof(FType) * rank * I);
    return 0.0;
  }

  FType rho = mat_trace(Phi) / (FType) rank;

  // printf("\n\n === mode: %d, rho: %f ===\n\n", mode, rho);
  // Debug
  for (int i = 0; i < rank; ++i) {
    Phi->vals[i * rank + i] += rho;
    // Phi->vals[i * rank + i] += 1e-12;
  }

  mat_cholesky(Phi);
  // PrintMatrix("Phi", Phi);
  /* for checking convergence */
  FType p_norm = 0.;
  FType d_norm = 0.;
  FType p_res  = 0.;
  FType d_res  = 0.;

  // IType chunk_size = 10;
  IType num_chunks =  (primal_mat->I / chunk_size);

  if(primal_mat->I % chunk_size > 0) {
    ++num_chunks;
  }

  // num_chunks = 1; // set for debugging
  #pragma omp parallel shared(p_norm,d_norm,p_res,d_res,niter)
  {
    #pragma omp for
    for(IType c=0; c < num_chunks; ++c) {
      IType const start = c * chunk_size;
      IType const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
      IType const offset = start * rank;
      IType const nrows = stop - start;
      IType const ncols = rank;

      /* extract all the workspaces per chunk */
      FType * primal = primal_mat->vals + offset;
      FType * auxil = auxil_mat->vals + offset;
      FType * dual = dual_mat->vals + offset;
      FType * mttkrp = mttkrp_buf->vals + offset;
      FType * init = init_buf->vals + offset;

      Matrix auxil_chunk_mat;

      mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);
      // row-wise/vector-wise fused formation of rhs
      #pragma simd
      #pragma vector aligned
      for (IType idx = 0; idx < ncols*nrows; ++idx) {
        auxil[idx] = mttkrp[idx] + rho*(primal[idx] + dual[idx]);
      }

      // chunk solve chol
      // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
      mat_cholesky_solve(Phi, &auxil_chunk_mat);

      // form prox and compute new norm
      for (IType i = 0; i < nrows; ++i) {
        for (IType j = 0; j < ncols; ++j) {
          IType idx = j + i*ncols;
          init[idx] = primal[idx];
          FType x = auxil[idx] - dual[idx];
          if (x < 0.0) {
            x = 0.0;
          }
          primal[idx] = x;
        }
      }
    }

    #pragma omp barrier

    IType it;
    int do_break = 0;
    for(it=0; it < 25; ++it) {
      { p_res = 0; d_res = 0; p_norm = 0; d_norm = 0; }

      #pragma omp for reduction(+:p_norm,p_res,d_norm,d_res)
      for(IType c=0; c < num_chunks; ++c) {
        IType const start = c * chunk_size;
        IType const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
        IType const offset = start * rank;
        IType const nrows = stop - start;
        IType const ncols = rank;

        /* extract all the workspaces per chunk */
        FType * const primal = primal_mat->vals + offset;
        FType * const auxil = auxil_mat->vals + offset;
        FType * const dual = dual_mat->vals + offset;
        FType * const mttkrp = mttkrp_buf->vals + offset;
        FType * const init = init_buf->vals + offset;

        Matrix auxil_chunk_mat;
        mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);

        // vectorized loop?
        // form prox and compute new norm
          // TODO:
          // instead do inner loop of some vector blocksize (64 bytes)
          // - duplicate norms to at least (B + ncols)
          // - ncols % blocksize  remainder, how to handle?
          // - block b is element b*B, which is column: b*B % ncols
        const IType cs = nrows*ncols;
        #pragma simd
        #pragma vector aligned
        for (IType idx = 0; idx < cs; ++idx) {

          // compute new primal and dual residual
          FType x = primal[idx];
          FType ddiff = x - init[idx];
          d_res += ddiff*ddiff;

          // update primal norm
          p_norm += x*x;

          // update dual U <- U + (pri - aux)
          FType y =  x - auxil[idx];
          FType di = dual[idx] + y;
          dual[idx] = di;

          // update dual norm and primal residual
          d_norm += di*di;
          p_res += y*y;

          // form next RHS for cholesky
          auxil[idx] = mttkrp[idx] + rho*(x + di);
        }

        // chunk solve chol
        // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
        mat_cholesky_solve(Phi, &auxil_chunk_mat);

        // form prox and compute new norm
        for (IType i = 0; i < nrows; ++i) {
          for (IType j = 0; j < ncols; ++j) {
            IType idx = j + i * ncols;
            FType x = auxil[idx] - dual[idx]; // updating primal (init)
            // printf("x: %f\n", x);
            // Non-neg
            if (x > 0.0) {
              init[idx] = x;
              primal[idx] = x;
            } else {
              init[idx] = 0.0;
              primal[idx] = 0.0;
            }
          }
        }
      }

      #pragma omp barrier

      float tolerence = 1e-6;
      if ((p_res <= tolerence * p_norm) && (d_res <= tolerence * d_norm)) {
        ++it;
        break;
      }

      #pragma omp barrier

    } /* admm iteration */

    #pragma omp master
    {
      niter = it; // niter is the number of iterations for convergence
    }

  } /* omp parallel */
  free_mat(Phi);
  fprintf(stderr, "ADMM C&F (nnzr: %llu, rho: %f): num of iteration: %llu\n", mat->I, rho, niter);

  return (FType) niter / mat->I; // this doesn't work
}

FType admm_cf_base(
    IType mode,
    Matrix * mat, // factor matrix
    Matrix ** aTa, 
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
) {
  // Unpack matrices used in ADMM
  Matrix * primal_mat = mat;
  Matrix * auxil_mat = ws->auxil; 
  Matrix * dual_mat = ws->duals[mode];
  Matrix * mttkrp_buf = ws->mttkrp_buf;
  Matrix * init_buf = ws->mat_init;
  
  IType I = mat->I;
  IType rank = mat->J;
  IType nmodes = ws->nmodes;

  IType niter;

  Matrix * Phi = hadamard_product(aTa, nmodes, mode);

  if (con->solve_type == UNCONSTRAINED) {
    // Apply frobenius regularization (1e-12)
    // This helps stablity 
    #pragma omp parallel for
    for (int i = 0; i < Phi->I; ++i) {
      Phi->vals[i * Phi->I + i] += 1e-12;
    }
    pseudo_inverse(Phi, mttkrp_buf);
    // PrintMatrix("mttkrp buf s_t: after solve", mttkrp_buf);

    // Copy back to factor matrix
    ParMemcpy(mat->vals, mttkrp_buf->vals, sizeof(FType) * rank * I);
    return 0.0;
  }

  FType rho = mat_trace(Phi) / (FType) rank;

  printf("\n\n === mode: %llu, rho: %f ===\n\n", mode, rho);
  // Debug
  for (int i = 0; i < rank; ++i) {
    Phi->vals[i * rank + i] += rho;
    // Phi->vals[i * rank + i] += 1e-12;
  }
  PrintMatrix("Phi", Phi);
  /* for checking convergence */
  FType p_norm = 0.;
  FType d_norm = 0.;
  FType p_res  = 0.;
  FType d_res  = 0.;

  IType chunk_size = 50;
  IType num_chunks =  (primal_mat->I / chunk_size);

  if(primal_mat->I % chunk_size > 0) {
    ++num_chunks;
  }

  num_chunks = 1; // set for debugging
  #pragma omp parallel shared(p_norm,d_norm,p_res,d_res,niter)
  {
    int tid = omp_get_thread_num();

    FType * norms = (FType*) malloc(rank*sizeof(FType));
    FType * colnorms = (FType*) malloc(rank*sizeof(FType));
    memset(norms, 0, rank*sizeof(FType));
    memset(colnorms, 0, rank*sizeof(FType));

    /*
    __assume_aligned(norms, 64);
    __assume_aligned(colnorms, 64);
    */

    #pragma omp for
    for(IType c=0; c < num_chunks; ++c) {
      IType const start = c * chunk_size;
      IType const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
      IType const offset = start * rank;
      IType const nrows = stop - start;
      IType const ncols = rank;

      /* extract all the workspaces per chunk */
      FType * primal = primal_mat->vals + offset;
      FType * auxil = auxil_mat->vals + offset;
      FType * dual = dual_mat->vals + offset;
      FType * mttkrp = mttkrp_buf->vals + offset;
      FType * init = init_buf->vals + offset;
      /*
      __assume_aligned(primal, 64);
      __assume_aligned(auxil, 64);
      __assume_aligned(dual, 64);
      __assume_aligned(mttkrp, 64);
      __assume_aligned(init, 64);
      */

      Matrix auxil_chunk_mat;

      mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);
      // row-wise/vector-wise fused formation of rhs
      #pragma simd
      #pragma vector aligned
      for (IType idx = 0; idx < ncols*nrows; ++idx) {
        auxil[idx] = mttkrp[idx] + rho*(primal[idx] + dual[idx]);
      }

      // chunk solve chol
      // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
      mat_cholesky_solve(Phi, &auxil_chunk_mat);

      // form prox and compute new norm
      for (IType i = 0; i < nrows; ++i) {
        for (IType j = 0; j < ncols; ++j) {
          IType idx = j + i*ncols;
          FType x = auxil[idx] - dual[idx];
          init[idx] = x; // primal
          // TODO: compute colnorm and perform possible thresholding (non-neg)
          colnorms[j] += x * x;
        }
      }
    }

    /* reduce norms */
    // TODO::: Deal with this later!!!
    #pragma omp barrier
    thread_allreduce(colnorms, rank);

#if d0520 == 1
    #pragma omp single
    PrintFPMatrix("colnorm", colnorms, rank, 1);
#endif
    for (IType j=0; j < rank; ++j) {
      colnorms[j] = sqrt(colnorms[j]);
      colnorms[j] = (colnorms[j] > 1.) ? colnorms[j] : 1.;
    }

    memcpy(norms, colnorms, rank*sizeof(FType));
    memset(colnorms, 0, rank*sizeof(FType));

    IType it;
    int do_break = 0;
    for(it=0; it < 25; ++it) {
      { p_res = 0; d_res = 0; p_norm = 0; d_norm = 0; }

      #pragma omp for reduction(+:p_norm,p_res,d_norm,d_res)
      for(IType c=0; c < num_chunks; ++c) {
        IType const start = c * chunk_size;
        IType const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
        IType const offset = start * rank;
        IType const nrows = stop - start;
        IType const ncols = rank;

        /* extract all the workspaces per chunk */
        FType * const primal = primal_mat->vals + offset;
        FType * const auxil = auxil_mat->vals + offset;
        FType * const dual = dual_mat->vals + offset;
        FType * const mttkrp = mttkrp_buf->vals + offset;
        FType * const init = init_buf->vals + offset;
        /*
        __assume_aligned(primal, 64);
        __assume_aligned(auxil, 64);
        __assume_aligned(dual, 64);
        __assume_aligned(mttkrp, 64);
        __assume_aligned(init, 64);
        */

        Matrix auxil_chunk_mat;
        mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);

        // vectorized loop?
        // form prox and compute new norm
          // TODO:
          // instead do inner loop of some vector blocksize (64 bytes)
          // - duplicate norms to at least (B + ncols)
          // - ncols % blocksize  remainder, how to handle?
          // - block b is element b*B, which is column: b*B % ncols
        for (IType i = 0; i < nrows; ++i) {
          for (IType j = 0; j < ncols; ++j) {
            IType idx = j + i*ncols;
            init[idx] /= norms[j];
          }
        }

        const IType cs = nrows*ncols;
        #pragma simd
        #pragma vector aligned
        for (IType idx = 0; idx < cs; ++idx) {

          // compute new primal and dual residual
          FType x = init[idx];
          FType pdiff = x - primal[idx];
          d_res += pdiff*pdiff;
          primal[idx] = x;

          // update primal norm
          p_norm += x*x;

          // update dual U <- U + (pri - aux)
          FType y =  x - auxil[idx];
          FType di = dual[idx] + y;
          dual[idx] = di;

          // update dual norm and primal residual
          d_norm += di*di;
          p_res += y*y;

          // form next RHS for cholesky
          auxil[idx] = mttkrp[idx] + rho*(x + di);
        }

        // chunk solve chol
        // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
        mat_cholesky_solve(Phi, &auxil_chunk_mat);

        // form prox and compute new norm
        for (IType i = 0; i < nrows; ++i) {
          for (IType j = 0; j < ncols; ++j) {
            IType idx = j + i*ncols;
            FType x = auxil[idx] - dual[idx];
            // Non-neg
            if (x > 0.0) {
              init[idx] = x;            
            } else {
              init[idx] = 0.0;
            }
            // TODO: compute colnorm and perform possible thresholding (non-neg)
            colnorms[j] += x * x;
          }
        }
      }

      #pragma omp barrier
      // printf("p_res, p_norm, d_res, d_norm, %f, %f, %f, %f\n", p_res, p_norm, d_res, d_norm);
      // exit(1);
      /* check convergence */

      float tolerence = 1e-2;
      if ((p_res <= tolerence * p_norm) && (d_res <= tolerence * d_norm)) {
        ++it;
        break;
      }
      if (p_norm <= 1e-12 && p_res <= 1e-12) {
        ++it;
        break;
      }
      if (d_norm <= 1e-12 && d_res <= 1e-12) {
        ++it;
        break;
      }

      /* reduce norms */
      // TODO:: Deal with this layer
      thread_allreduce(colnorms, rank);

      for (IType j=0; j < rank; ++j) {
        colnorms[j] = sqrt(colnorms[j]);
        colnorms[j] = (colnorms[j] > 1.) ? colnorms[j] : 1.;
      }

      memcpy(norms, colnorms, rank*sizeof(FType));
      memset(colnorms, 0, rank*sizeof(FType));

    } /* admm iteration */

    #pragma omp master
    {
      niter = it;
    }

    free(norms);
    free(colnorms);
  } /* omp parallel */
  free_mat(Phi);
  fprintf(stderr, "ADMM C&F (nnzr: %llu, rho: %f): num of iteration: %llu\n", mat->I, rho, niter);

  return niter;
}


/**
* @brief Perform a parallel SUM reduction.
*
* @param thds The data we are reducing (one array for each thread).
* @param buffer thread-local buffer.
* @param nelems How many elements in the scratch array.
*/
static void p_reduce_sum(
    FType * * reduce_ptrs,
    FType * buffer,
    IType const nelems)
{
  int const tid = omp_get_thread_num();
  int const nthreads = omp_get_num_threads();

  int half = nthreads / 2;
  while(half > 0) {
    if(tid < half && tid + half < nthreads) {
      FType const * const target = reduce_ptrs[tid+half];
      for(IType i=0; i < nelems; ++i) {
        buffer[i] += target[i];
      }
    }

    #pragma omp barrier

    /* check for odd number */
    #pragma omp master
    if(half > 1 && half % 2 == 1) {
        FType const * const last = reduce_ptrs[half-1];
        for(IType i=0; i < nelems; ++i) {
          buffer[i] += last[i];
        }
    }

    /* next iteration */
    half /= 2;
  }

  /* account for odd thread at end */
  #pragma omp master
  {
    if(nthreads % 2 == 1) {
      FType const * const last = reduce_ptrs[nthreads-1];
      for(IType i=0; i < nelems; ++i) {
        buffer[i] += last[i];
      }
    }
  }
}

void thread_allreduce(
    FType * const buffer,
    IType const nelems)
{
  int const tid = omp_get_thread_num();
  int const nthreads = omp_get_num_threads();

  /* used to get coherent all-to-all access to reduction data. */
  static FType ** reduce_ptrs;

  if(nthreads == 1) {
    return;
  }

  /* get access to all thread pointers */
  #pragma omp master
  reduce_ptrs = (FType**)malloc(nthreads * sizeof(*reduce_ptrs));
  #pragma omp barrier

  reduce_ptrs[tid] = buffer;
  #pragma omp barrier

  /* do the reduction */
  p_reduce_sum(reduce_ptrs, buffer, nelems);

  #pragma omp barrier

  /* now each thread grabs master values */
  for(IType i=0; i < nelems; ++i) {
    buffer[i] = reduce_ptrs[0][i];
  }

  #pragma omp barrier

  #pragma omp master
  free(reduce_ptrs);
}


/*
 * Function: admm_func
 * -------------------
 * pure function version of admm 
 *
 * Matrix * A
 * Matrix * Phi
 * Matrix * Psi
 * Matrix * U
 * cpd_constraint * con
 * 
 * returns: (Implicitly) A'
 **/
IType admm_func(
	Matrix * A,
	Matrix * Phi,
	Matrix * Psi,
	Matrix * U,
	cpd_constraint * con) {
	// Get dimensions
	IType I = A->I;
	IType rank = A->J;

	// Set dual variables U to zero
	for (size_t n = 0; n < I * rank; ++n) {
		// U->vals[n] = fill_rand();
	}

  fill_rand(U->vals, I * rank, NULL);

	FType rho = mat_trace(Phi) / (FType) rank;

	// Add rho * I to Phi
	for (size_t i = 0; i < rank; ++i) {
		Phi->vals[i * rank + i] += rho;
	}

	mat_cholesky(Phi);

	size_t N = I * rank;
	size_t bytes = N * sizeof(FType);

	FType primal_norm = 0.0;
	FType dual_norm = 0.0;
	FType primal_residual = 0.0;
	FType dual_residual = 0.0;

	IType it;

	// Create dummy matrices
	Matrix * auxil = zero_mat(I, rank);
	Matrix * dual = U;
	Matrix * init = zero_mat(I, rank);

	for (it = 0; it < 25; ++it) {
		memcpy(init, A, bytes);
		for (size_t i = 0; i < N; ++i) {
			auxil->vals[i] = Psi->vals[i] + rho * (auxil->vals[i] + dual->vals[i]);
		}

		mat_cholesky_solve(Phi, auxil);

		for (size_t i = 0; i < N; ++i) {
			A->vals[i] = auxil->vals[i] - dual->vals[i];
		}

		con->func(A->vals, A->I, A->J);

		FType dual_norm = 0.0;

		for (size_t i = 0; i < N; ++i) {
			dual->vals[i] += (A->vals[i] - auxil->vals[i]);
			dual_norm += (dual->vals[i] * dual->vals[i]);
		}

		primal_norm = 0.0;
		primal_residual = 0.0;
		dual_residual = 0.0;

		for (size_t i = 0; i < A->I * A->J; ++i) {
			FType p_diff = A->vals[i] - auxil->vals[i];
			FType d_diff = A->vals[i] - init->vals[i];

			primal_norm += A->vals[i] * A->vals[i];
			primal_residual += p_diff * p_diff;
			dual_residual += d_diff * d_diff;
		}

		float tolerence = 1e-4;
		printf("p_res: %f, p_norm: %f, d_res: %f, d_norm: %f\n", primal_residual, primal_norm, dual_residual, dual_norm);
    		if ((primal_residual <= tolerence * primal_norm) && (dual_residual <= tolerence * dual_norm)) {
      			++it;
      			break;
    		}
	}
	fprintf(stderr, "ADMM (nnzr: %llu, rho: %f): num of iteration: %llu\n", A->I, rho, it);
	// Free allocated matrix within function
  free_mat(auxil);
  free_mat(init);
  return it;
};

/*
 * Function: admm_cf_func
 * -------------------
 * pure function version of admm c&f
 *
 * Matrix * A
 * Matrix * Phi
 * Matrix * Psi
 * Matrix * U
 * size_t chunk_size,
 * cpd_constraint * con
 * 
 * returns: (Implicitly) A'
 **/
IType admm_cf_func(
	Matrix * A,
	Matrix * Phi, // Don't touch Phi
	Matrix * Psi, // ws->mttkrp
	Matrix * U, // ws->dual
  Matrix * mat_init, // ws->mat_init
  Matrix * auxil_mat, // ws->auxil_mat
  size_t chunk_size,
	cpd_constraint * con) {
	// Get dimensions
	IType I = A->I;
	IType rank = A->J;

  // printf("%d, %d\n", I, rank);
	// Set dual variables U to zero
	for (size_t n = 0; n < I * rank; ++n) {
		U->vals[n] = 0.0;
	}

  Matrix * p_Phi = init_mat(Phi->I, Phi->J);
  memcpy(p_Phi->vals, Phi->vals, sizeof(FType) * Phi->I * Phi->J);

  FType rho = mat_trace(p_Phi) / (FType) rank;

	// Add rho * I to Phi
	for (size_t i = 0; i < rank; ++i) {
		p_Phi->vals[i * rank + i] += rho;
	}

  // TODO: We should move this out of parallel region and reusce factorized Phi
	mat_cholesky(p_Phi);

	size_t N = I * rank;
	size_t bytes = N * sizeof(FType);

	FType p_norm = 0.0;
	FType d_norm = 0.0;
	FType p_res = 0.0;
	FType d_res = 0.0;

	// Create dummy matrices
	// Matrix * auxil_mat = zero_mat(I, rank);
	Matrix * dual_mat = U;
	// Matrix * init_mat = zero_mat(I, rank);

  IType niter;

	IType num_chunks = (A->I / chunk_size);

	if (A->I % chunk_size > 0) {
		++num_chunks;
	}

  #pragma omp parallel shared(p_norm,d_norm,p_res,d_res,niter)
  {
    #pragma omp for
    for(IType c=0; c < num_chunks; ++c) {
      IType const start = c * chunk_size;
      IType const stop = (c == num_chunks-1) ? A->I : (c+1)*chunk_size;
      IType const offset = start * rank;
      IType const nrows = stop - start;
      IType const ncols = rank;

      /* extract all the workspaces per chunk */
      FType * primal = A->vals + offset;
      FType * auxil = auxil_mat->vals + offset;
      FType * dual = dual_mat->vals + offset;
      FType * mttkrp = Psi->vals + offset;
      FType * init = mat_init->vals + offset;

      Matrix auxil_chunk_mat;
      mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);
      // row-wise/vector-wise fused formation of rhs
      #pragma simd
      #pragma vector aligned
      for (IType idx = 0; idx < ncols*nrows; ++idx) {
        auxil[idx] = mttkrp[idx] + rho*(primal[idx] + dual[idx]);
      }

      mat_cholesky_solve(p_Phi, &auxil_chunk_mat);

      // form prox and compute new norm
      for (IType i = 0; i < nrows; ++i) {
        for (IType j = 0; j < ncols; ++j) {
          IType idx = j + i*ncols;
          init[idx] = primal[idx];
          FType x = auxil[idx] - dual[idx];
          if (x < 0.0) {
            x = 0.0;
          }
          primal[idx] = x;
        }
      }
    }

    #pragma omp barrier
    IType it;
    int do_break = 0;
    for(it=0; it < 100; ++it) {
      { p_res = 0; d_res = 0; p_norm = 0; d_norm = 0; }
      #pragma omp for reduction(+:p_norm,p_res,d_norm,d_res)
      for(IType c=0; c < num_chunks; ++c) {
        IType const start = c * chunk_size;
        IType const stop = (c == num_chunks-1) ? A->I : (c+1)*chunk_size;
        IType const offset = start * rank;
        IType const nrows = stop - start;
        IType const ncols = rank;

        /* extract all the workspaces per chunk */
        FType * const primal = A->vals + offset;
        FType * const auxil = auxil_mat->vals + offset;
        FType * const dual = dual_mat->vals + offset;
        FType * const mttkrp = Psi->vals + offset;
        FType * const init = mat_init->vals + offset;

        Matrix auxil_chunk_mat;
        mat_hydrate(&auxil_chunk_mat, auxil, nrows, rank);

        // vectorized loop?
        // form prox and compute new norm
        // TODO:
        // instead do inner loop of some vector blocksize (64 bytes)
        // - duplicate norms to at least (B + ncols)
        // - ncols % blocksize  remainder, how to handle?
        // - block b is element b*B, which is column: b*B % ncols
        const IType cs = nrows*ncols;
        #pragma simd
        #pragma vector aligned
        for (IType idx = 0; idx < cs; ++idx) {

          // compute new primal and dual residual
          FType x = primal[idx];
          FType ddiff = x - init[idx];
          d_res += ddiff*ddiff;

          // update primal norm
          p_norm += x*x;

          // update dual U <- U + (pri - aux)
          FType y =  x - auxil[idx];
          FType di = dual[idx] + y;
          dual[idx] = di;

          // update dual norm and primal residual
          d_norm += di*di;
          p_res += y*y;

          // form next RHS for cholesky
          auxil[idx] = mttkrp[idx] + rho*(x + di);
        }

        // chunk solve chol
        // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
        mat_cholesky_solve(p_Phi, &auxil_chunk_mat);

        // form prox and compute new norm
        for (IType i = 0; i < nrows; ++i) {
          for (IType j = 0; j < ncols; ++j) {
            IType idx = j + i * ncols;
            FType x = auxil[idx] - dual[idx]; // updating primal (init)
            // printf("x: %f\n", x);
            // Non-neg
            if (x > 0.0) {
              init[idx] = x;
              primal[idx] = x;
            } else {
              init[idx] = 0.0;
              primal[idx] = 0.0;
            }
          }
        }
      }

      #pragma omp barrier

      // #pragma omp single
      // fprintf(stderr, "p_res, p_norm, d_res, d_nrom, %f, %f, %f, %f\n", p_res, p_norm, d_res, d_norm);

      float tolerence = 1e-4;
      if ((p_res <= tolerence * p_norm) && (d_res <= tolerence * d_norm)) {
        ++it;
        break;
      }
      #pragma omp barrier
    } /* admm iteration */

    #pragma omp master
    {
        niter = it; // niter is the number of iterations for convergence
    }
} /* omp parallel */

  free_mat(p_Phi);
	// fprintf(stderr, "ADMM C&F (nnzr: %llu, rho: %f): num of iteration: %llu\n", A->I, rho, niter);

  return niter;
};

FType admm_opt(
    IType mode,
    Matrix * mat,
    Matrix * Phi,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws,
    size_t block_size
) {
  // Unpack matrices used in ADMM
  Matrix * primal_mat = mat;
  Matrix * auxil_mat = ws->auxil; 
  Matrix * dual_mat = ws->duals[mode];
  Matrix * mttkrp_buf = ws->mttkrp_buf;
  Matrix * mat_init = ws->mat_init;

  // Use copy of Phi so that original Phi is in-tact
  Matrix * _Phi = init_mat(Phi->I, Phi->J);
  memcpy(_Phi->vals, Phi->vals, Phi->I * Phi->J * sizeof(*Phi->vals));

  IType I = mat->I;
  IType rank = mat->J;
  IType nmodes = ws->nmodes;

  if (con->solve_type == UNCONSTRAINED) {
    pseudo_inverse(_Phi, mttkrp_buf);
    // Copy back to factor matrix
    ParMemcpy(mat->vals, mttkrp_buf->vals, sizeof(FType) * rank * I);
    return 0.0;
  }

  free_mat(_Phi);

  // ADMM realm -- set up blocks
  Matrix * Psi = mttkrp_buf;
  Matrix * U = dual_mat;
  // Matrix U;

	// size_t block_size = 64;
	size_t num_blocks = (mat->I / block_size);

	if (mat->I % block_size > 0) {
		++num_blocks;
	}

  IType total_iters = 0;

  #pragma omp parallel for schedule(dynamic) reduction(+:total_iters)
	for (size_t bidx = 0; bidx < num_blocks; ++bidx) {
		size_t start = bidx * block_size;
		size_t stop = (bidx == num_blocks-1) ? mat->I : (bidx + 1) * block_size;
		size_t offset = start * rank;
		size_t nrows = stop - start;
		size_t ncols = rank;

		Matrix b_mat;
		Matrix b_init_mat;
		Matrix b_auxil_mat;
		Matrix b_Psi;
		Matrix b_U;

		mat_hydrate(&b_mat, mat->vals + offset, nrows, ncols);
		mat_hydrate(&b_init_mat, mat_init->vals + offset, nrows, ncols);
		mat_hydrate(&b_auxil_mat, auxil_mat->vals + offset, nrows, ncols);
		mat_hydrate(&b_Psi, Psi->vals + offset, nrows, ncols);
		mat_hydrate(&b_U, U->vals + offset, nrows, ncols);
		
    IType it = admm_cf_func(&b_mat, Phi, &b_Psi, &b_U, &b_init_mat, &b_auxil_mat, 64, con);

		total_iters += it * nrows;
	};


  return (FType) total_iters/(FType)mat->I;
};
