#include "admm_test.hpp"

void admm_test_(void) {
	// Do we need to fix the seed?
	Matrix * mat = rand_mat(1000, 16);

	// What are we trying to test
	// Should we create a pure function just for testing
	printf("testing admm...\n");
	TEST_ASSERT(true);
};

// void admm_base_test(void) {
// 	printf("test\n");
// 	Matrix * mat = rand_mat(100, 16);
// 	Matrix * Psi = rand_mat(100, 16);

// 	Matrix * other_mats = rand_mat(100, 16);
// 	Matrix * Phi = init_mat(16, 16);

// 	// Phi = aTa
// 	mat_aTa(other_mats, Phi);

// 	Matrix * U = init_mat(100, 16);
// 	cpd_constraint * con = init_constraint();
// 	apply_nonneg_constraint(con);

// 	// PrintMatrix("before admm", mat);
// 	admm_func(mat, Phi, Psi, U, con);
// 	// PrintMatrix("after admm", mat);
// 	TEST_ASSERT(false);
// };

// void admm_cf_test(void) {
// 	printf("test\n");
// 	Matrix * mat = rand_mat(100, 16);
// 	Matrix * Psi = rand_mat(100, 16);

// 	Matrix * other_mats = rand_mat(100, 16);
// 	Matrix * Phi = init_mat(16, 16);

// 	// Phi = aTa
// 	mat_aTa(other_mats, Phi);

// 	Matrix * U = init_mat(100, 16);
// 	cpd_constraint * con = init_constraint();
// 	apply_nonneg_constraint(con);

// 	// PrintMatrix("before admm", mat);
// 	admm_cf_func(mat, Phi, Psi, U, con);
// 	// PrintMatrix("after admm", mat);
// 	TEST_ASSERT(false);
// };
