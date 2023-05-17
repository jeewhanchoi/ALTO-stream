#ifndef STREAMING_CPD_HPP_
#define STREAMING_CPD_HPP_

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "stream_matrix.hpp"
#include "streaming_sptensor.hpp"
#include "util.hpp"
#include "alto.hpp"
#include "gram.hpp"
#include "mttkrp.hpp"
#include "rowsparse_mttkrp.hpp"
#include "constraints.hpp"
#include "admm.hpp"
#include "cpd.hpp"

using namespace std;

// Signatures
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
    IType timeslice_limit);

// Structs
struct sparse_cp_grams {
    Matrix ** c_nz_prev;
    Matrix ** c_z_prev;

    Matrix ** c_nz;
    Matrix ** c_z;

    Matrix ** h_nz;
    Matrix ** h_z;

    Matrix ** c;
    Matrix ** h;

    Matrix ** c_prev;
}; typedef struct sparse_cp_grams SparseCPGrams;


template <typename LIT>
void spcpstream_alto_iter(
    SparseTensor* X, AltoTensor<LIT>* at, KruskalModel* M, KruskalModel * prev_M, 
    Matrix** grams, SparseCPGrams * scpgrams, int max_iters, double epsilon, 
    int streaming_mode, int iter, cpd_constraint * con, admm_ws * ws, double * t_tensor);

template <typename LIT>
void cpstream_alto_iter(AltoTensor<LIT>* at, FType **ofibs, KruskalModel* M, KruskalModel * prev_M, 
    Matrix** grams, int max_iters, double epsilon, 
    int streaming_mode, int iter, cpd_constraint * con, admm_ws * ws, double * t_tensor);

// Specifically for cpstream
static double cpstream_fit(SparseTensor* X, KruskalModel* M, Matrix ** grams, FType* U_mttkrp);

SparseCPGrams * InitSparseCPGrams(IType nmodes, IType rank);
void DeleteSparseCPGrams(SparseCPGrams * grams, IType nmodes);

void nonzero_slices(
    SparseTensor * const tt, const IType mode,
    vector<size_t>& nz_rows,
    vector<size_t>& z_rows,
    vector<size_t>& idx,
    vector<int>& ridx,
    vector<size_t>& buckets);

/* Given a time-slice subtensor, 
 * implicitly returns hsls_rows (nz_rows), hsls_idx (idx), hsls_buckets (buckets) 
 * However, it uses the same variable names due to compatibility with preexisting APIs
 */
void nonzero_slices_hsls(
    SparseTensor * const tt, const IType mode,
    vector<size_t>& nz_rows,
    vector<size_t>& z_rows,
    vector<size_t>& idx,
    vector<int>& ridx,
    vector<size_t>& buckets);

double cpd_error(SparseTensor * X, KruskalModel * M);

double compute_errorsq(
    StreamingSparseTensor * sst,
    KruskalModel * M,
    IType previous);

double compute_cpd_errorsq(
    StreamingSparseTensor * sst,
    IType rank,
    IType previous);

#endif // STREAMING_CPD_HPP_