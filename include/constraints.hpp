#ifndef CONSTRAINTS_HPP_
#define CONSTRAINTS_HPP_

#include "common.hpp"

typedef enum
{
  UNCONSTRAINED,
  NON_NEGATIVE,
  LASSO,
  MAX_COL_NORM,
} con_solve_type;


typedef struct 
{
  con_solve_type solve_type;

  char description[256];

  void (* func) (FType * primal,
                            IType nrows,
                            IType ncols);
} cpd_constraint;

void constraint_lasso(FType * primal, IType nrows, IType ncols);
void constraint_nonneg(FType * primal, IType nrows, IType ncols);

cpd_constraint * init_constraint();
void free_constraint(cpd_constraint * con);
void apply_nonneg_constraint(cpd_constraint * con);
void apply_lasso_constraint(cpd_constraint * con);

#endif
