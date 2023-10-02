#ifndef STREAMING_SPTENSOR_HPP_
#define STREAMING_SPTENSOR_HPP_

#include "common.hpp"
#include "sptensor.hpp"
#include "util.hpp"
#include "sort.hpp"


class StreamingSparseTensor {
    public:
        StreamingSparseTensor(
            SparseTensor * sp,
            IType stream_mode
        );
        ~StreamingSparseTensor() {
            perm_free(_perm, _tensor->nmodes);
        };

        SparseTensor * next_batch();
        SparseTensor * next_dynamic_batch(IType nnz_threshold, IType timeslice_limit);
        void print_tensor_info();
        IType num_modes();

        bool last_batch();

        SparseTensor * full_stream();
        SparseTensor * stream_until(IType time);
        SparseTensor * stream_prev(IType previous);
        
        int _stream_mode;
        SparseTensor * _tensor;
        SparseTensor * _next_batch;
        int _batch_num;

        /* Store permutation, inverse permutation info */
        Permutation * _perm;

    private:
        IType _prev_dim[MAX_NUM_MODES];

        IType _nnz_ptr;

};

#endif // STREAMING_SPTENSOR_HPP_
