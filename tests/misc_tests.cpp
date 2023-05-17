#include "misc_tests.hpp"

void nonzero_row_sparsity(void) {
    // SparseTensor* X = load_tensor();
    // SparseTensor* X = load_frostt_dataset("uber.tns");
    SparseTensor * X = NULL;

    // dummy timer variable to time tensor operations
    double t_tensor = 0.0;
    double epsilon = 1e-3;
    unsigned char seed = 44;
    bool use_alto = false;
    bool use_spcpstream = true;

    ImportSparseTensor(DATASET(flickr-4d.tns), TEXT_FORMAT, &X, 0);
    IType streaming_mode = 3;

    /*
    ImportSparseTensor(DATASET(chicago-crime-comm.tns), TEXT_FORMAT, &X, 0);
    IType streaming_mode = 0;
    */

    /*
    ImportSparseTensor(DATASET(patents.tns), TEXT_FORMAT, &X, 0);
    IType streaming_mode = 0;
    */

    /*
    ImportSparseTensor(DATASET(uber.tns), TEXT_FORMAT, &X, 0);
    IType streaming_mode = 0;
    */

    IType rank = 16;
    // IType max_iters = 100;
    IType max_iters = 1;

    StreamingSparseTensor sst(X, streaming_mode);
    sst.print_tensor_info();

    IType nmodes = X->nmodes;

    KruskalModel * M; // Keeps track of current factor matrices
    KruskalModel * prev_M; // Keeps track of previous factor matrices

    Matrix ** grams;
    // concatencated s_t's
    Matrix * local_time = zero_mat(1, rank);
    StreamMatrix * global_time = new StreamMatrix(rank);

    SparseCPGrams * scpgrams;

    admm_ws * ws = admm_ws_init(nmodes);

    if (use_spcpstream) {
        // Hypersparse ALS specific
        scpgrams = InitSparseCPGrams(nmodes, rank);
    }

    cpd_constraint * con = init_constraint();
    apply_nonneg_constraint(con);


    int it = 0;
    while (!sst.last_batch()) {
        SparseTensor * t_batch = sst.next_batch();
        PrintTensorInfo(rank, max_iters, t_batch);
        if (it == 0) {
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &M);
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &prev_M);
            KruskalModelRandomInit(M, (unsigned int)seed);
            KruskalModelZeroInit(prev_M);

            // Override values for M->U[stream_mode] with last row of local_time matrix
            M->U[streaming_mode] = local_time->vals;

            // TODO: specify what "grams" exactly
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

        /* End: Setup ADMM workspace */
        // TODO: Change to StreamMatrix later on
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
        /* End: Setup ADMM workspace */

        if (use_spcpstream) {
          if (con->solve_type!=UNCONSTRAINED) {
            constrained_spcpstream_iter(
            t_batch, M, prev_M, grams, scpgrams,
            max_iters, epsilon, streaming_mode, it, con, ws, &t_tensor);
          } else {
            spcpstream_iter(
              t_batch, M, prev_M, grams, scpgrams,
              max_iters, epsilon, streaming_mode, it, con, ws, &t_tensor);
          }
        }
        else {
            cpstream_iter(
                t_batch, M, prev_M, grams,
                max_iters, epsilon, streaming_mode, it, con, ws, &t_tensor);
        }
        ++it; // increment of it has to precede global_time memcpy

        // Copy M -> prev_M
        CopyKruskalModel(prev_M, M);
        DestroySparseTensor(t_batch);
        global_time->grow_zero(it);
        memcpy(&(global_time->mat()->vals[rank * (it-1)]), M->U[streaming_mode], rank * sizeof(FType));

        /* Free ADMM workspace */
        for (int m = 0; m < nmodes; ++m) {
          free_mat(ws->duals[m]);
        }
        free_mat(ws->mttkrp_buf);        
        free_mat(ws->auxil);
        free_mat(ws->mat_init);
        /* End: Free ADMM workspace */

    } /* End streaming */

    // Streaming

    /*
    KruskalModel* M;
    CreateKruskalModel(X->nmodes, X->dims, rank, &M);

    unsigned int seed = 44;
    double epsilon = 1e-4;

    KruskalModelRandomInit(M, (unsigned int)seed);

    cpd(X, M, max_iters, epsilon);

    // ExportKruskalModel(M, DATASET(uber.tns));

    DestroySparseTensor(X);
    DestroyKruskalModel(M);
    */
}


