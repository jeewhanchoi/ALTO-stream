#include "streaming_sptensor_test.hpp"

void streaming_tensor_test(void) {
	SparseTensor* X = load_tensor();
	printf("Processing Streaming Sparse Tensor\n");
	int streaming_mode = 0;
	int rank = 16;
	int max_iters = 100;
	unsigned int seed = 45;

	StreamingSparseTensor sst(X, streaming_mode);
	sst.print_tensor_info();

	// SparseTensor * t_batch = sst.next_dynamic_batch(1000);
	// SparseTensor * t_batch = sst.next_batch();

  while(!sst.last_batch()) {
	  SparseTensor * t_batch = sst.next_dynamic_batch(40000);
    // SparseTensor * t_batch = sst.next_batch();
    PrintSparseTensor(t_batch);
    PrintTensorInfo(rank, max_iters, t_batch);
    DestroySparseTensor(t_batch);
  }

  // PrintTensorInfo(16, 100, t_batch);
  TEST_ASSERT(true);
}