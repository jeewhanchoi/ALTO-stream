#include "constraints.hpp"


void constraint_lasso(
    FType * primal,
    IType nrows,
    IType ncols)
{
    FType mult = 1e-12;

    #pragma omp parallel for schedule(static)
    for (IType i = 0; i < nrows * ncols; ++i) {
        FType v = primal[i];

        if (v > mult) {
          primal[i] = v - mult;
        } else if (v < -mult) {
          primal[i] = v + mult;
        } else {
          primal[i] = 0.0;
        }
    }  
};

void constraint_nonneg(
    FType * primal, 
    IType nrows, 
    IType ncols)
{
    #pragma omp parallel for schedule(static)
    for (IType i = 0; i < nrows; ++i) {
        for (IType j = 0; j < ncols; ++j) {
            IType idx = j + (i * ncols);
            primal[idx] = (primal[idx] > 0.0) ? primal[idx] : 0.0;
        }
    }
};

cpd_constraint *  init_constraint() {
  cpd_constraint * con;
  con = (cpd_constraint *) malloc(sizeof(*con));
  
  con->func = NULL;
  con->solve_type = UNCONSTRAINED;
  
  sprintf(con->description, "UNCONSTRAINED");
  return con;
};

void free_constraint(cpd_constraint * con) {
  if (con == NULL) return;
  free(con);
};

void apply_nonneg_constraint(cpd_constraint * con) {
  memset(con->description, 0, 256);
  sprintf(con->description, "NON-NEGATIVE");
  con->solve_type = NON_NEGATIVE;
  con->func = constraint_nonneg;
};

void apply_lasso_constraint(cpd_constraint * con) {
  memset(con->description, 0, 256);
  sprintf(con->description, "LASSO (SPARSITY)");
  con->solve_type = LASSO;
  con->func = constraint_nonneg;
};