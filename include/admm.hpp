#ifndef ADMM_HPP_
#define ADMM_HPP_

#include "common.hpp"
#include "matrix.hpp"
#include "kruskal_model.hpp"
#include "constraints.hpp"
#include "rowsparse_matrix.hpp"

typedef struct 
{
    int nmodes;
    Matrix * mttkrp_buf; // the output of the mttkrp operation
    Matrix * auxil; // aux matrix for AO-ADMM factorization
    Matrix * duals[MAX_NUM_MODES]; // dual matrices for AO-ADMM factorization
    Matrix * mat_init; // Store the initial primal variable from each ADMM iteration
} admm_ws;

FType admm(
    IType mode,
    Matrix * mat,
    Matrix ** aTa,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
);

// the wrapped version
FType admm_cf(
    IType mode,
    Matrix * mat,
    Matrix ** aTa,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
);

// the modified version (non negativity constraint)
FType admm_cf_(
    IType mode,
    Matrix * mat,
    Matrix ** aTa,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
);


// the copied version (has max col norm)
FType admm_cf_base(
    IType mode,
    Matrix * mat,
    Matrix ** aTa,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws
);

// Varying chunk size version
FType admm_cf_var_cs(
    IType mode,
    Matrix * mat,
    Matrix ** aTa,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws,
    IType chunk_size
);

admm_ws * admm_ws_init(int nmodes);

void thread_allreduce(
    FType * const buffer,
    IType const nelems);

IType admm_func(
	Matrix * A,
	Matrix * Phi,
	Matrix * Psi,
	Matrix * dual,
	cpd_constraint * con);

IType admm_cf_func(
	Matrix * A,
	Matrix * Phi,
	Matrix * Psi,
	Matrix * U,
    Matrix * mat_init, // ws->mat_init
    Matrix * auxil_mat, // ws->auxil_mat
    size_t chunk_size,
	cpd_constraint * con);

// admm API
FType admm_opt(
    IType mode,
    Matrix * mat,
    Matrix * Phi,
    FType * column_weights,
    cpd_constraint * con,
    admm_ws * ws,
    size_t block_size
);

#endif
