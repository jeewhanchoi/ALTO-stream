#ifndef IMPORT_ACUTEST_H
#define IMPORT_ACUTEST_H
#include "acutest.h"
#endif

#include "matrix.hpp"
#include "sptensor.hpp"
#include "streaming_sptensor.hpp"
#include "kruskal_model.hpp"
#include "gram.hpp"
#include "mttkrp.hpp"
#include "alto.hpp"
#include "common.hpp"

// Include test related helpers and setup funcs
#include "test_helpers.hpp"

// All the test code goes here
#include "mttkrp_test.hpp"
#include "admm_test.hpp"
#include "streaming_sptensor_test.hpp"

TEST_LIST = {
	// mttkrp tests
	//  { "mttkrp", mttkrp_test},
	{ "streaming_sptensor", streaming_tensor_test },
	 // { "mttkrp_alto", mttkrp_alto_test},	 
	// admm tests
	//  { "admm vs admm_cf", admm_test_},
	// nz_slices test
   { NULL, NULL } 
};

/*
int main() {
  Matrix * r1 = rand_mat(4, 4);
  PrintMatrix("r1", r1);
}
*/
