#define TEST_NO_MAIN
#include "acutest.h"

#include "matrix.hpp"
#include "sptensor.hpp"
#include "streaming_sptensor.hpp"
#include "kruskal_model.hpp"
#include "gram.hpp"
#include "mttkrp.hpp"
#include "alto.hpp"
#include "common.hpp"

#include "test_helpers.hpp"

void mttkrp_test(void) {
	SparseTensor* X = load_tensor();
	printf("Processing Streaming Sparse Tensor\n");
	int streaming_mode = 4;
	int rank = 16;
	int max_iters = 100;
	unsigned int seed = 45;

	StreamingSparseTensor sst(X, streaming_mode);
	sst.print_tensor_info();

	SparseTensor * t_batch = sst.next_batch();
	PrintTensorInfo(rank, max_iters, t_batch);

	Matrix ** grams;
	KruskalModel * M; // Keeps track of current factor matrices
	CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &M);
	KruskalModelRandomInit(M, (unsigned int)seed);
	omp_lock_t* writelocks;
	IType max_mode_len = 0;

	for(int i = 0; i < M->mode; i++) {
			if(max_mode_len < M->dims[i]) {
					max_mode_len = M->dims[i];
			}
	}
	writelocks = (omp_lock_t*) AlignedMalloc(sizeof(omp_lock_t) * max_mode_len);
	assert(writelocks);
	for(IType i = 0; i < max_mode_len; i++) {
			omp_init_lock(&(writelocks[i]));
	}
	mttkrp_par(t_batch, M, streaming_mode, writelocks);
	PrintFPMatrix("mttkrp result for streaming mode", M->U[streaming_mode], rank, 1);
	TEST_ASSERT(false);
};

void mttkrp_alto_test(void) {
	SparseTensor* X = load_tensor();
	printf("Processing Streaming Sparse Tensor\n");
	int streaming_mode = 4;
	int rank = 16;
	int max_iters = 100;
	unsigned int seed = 45;

	StreamingSparseTensor sst(X, streaming_mode);
	sst.print_tensor_info();

	SparseTensor * t_batch = sst.next_batch();
	PrintTensorInfo(rank, max_iters, t_batch);

	int nmodes = t_batch->nmodes;

	// Init ALTO stuff
	AltoTensor<LIType> * AT;
	AltoTensor<RLIType> * ATR;

	std::vector<std::vector<size_t>> oidx;
	FType ** ofibs = NULL;

	int num_partitions = get_num_ptrn(t_batch->nnz);
	int num_bits = get_num_bits(t_batch->dims, nmodes);

	init_salto(X, &ATR, omp_get_max_threads(), streaming_mode);      

	if (num_bits > ((int)sizeof(RLIType) * 8)) {
		printf("Full index\n");
	} else {
		printf("Not full index\n");		
	}

	update_salto(t_batch, ATR, num_partitions);
	create_da_mem(-1, rank, ATR, &ofibs);


	Matrix ** grams;
	KruskalModel * M; // Keeps track of current factor matrices
	CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &M);
	KruskalModelRandomInit(M, (unsigned int)seed);

	mttkrp_alto_par(streaming_mode, M->U, rank, ATR, NULL, ofibs, oidx);
	PrintFPMatrix("mttkrp result for streaming mode", M->U[streaming_mode], rank, 1);

	TEST_ASSERT(false);

}

