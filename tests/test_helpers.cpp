#include "test_helpers.hpp"

#define DATASET(x) "./tests/tensors/" #x
// #define DATASET(x) DATASET_(x)

static char const * const datasets[] = {
  // DATASET(small.tns),
  // DATASET(med.tns),
  // DATASET(small4.tns),
  // DATASET(med4.tns),
  // DATASET(med5.tns)
};

// Maybe go to test_utils.cpp
SparseTensor * load_tensor(void) {
	SparseTensor* X = NULL;
	printf("\nUsing dataset: %s\n", DATASET(uber.tns));
	ImportSparseTensor(DATASET(uber.tns), TEXT_FORMAT, &X, 0);
	return X;
}
