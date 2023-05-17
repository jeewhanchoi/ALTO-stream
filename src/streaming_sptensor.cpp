#include "streaming_sptensor.hpp"
#include "algorithm"
#include <vector>
#include "assert.h"
#include <iostream>

using namespace std;

// Debugging purposes
void print_sptensor(SparseTensor *sp, int nnz) {
    fprintf(stderr, "Sparse tensor has %llu nnzs\n", sp->nnz);
    
    for (int n = 0; n < sp->nmodes; ++n) {
        fprintf(stderr, "Dim #%d: %llu\n", n, sp->dims[n]);
    }
    
    for (IType n = 0; n < nnz; ++n) {
        // print dims
        fprintf(stderr, "(");
        for (int i = 0; i < sp->nmodes; ++i) {
            fprintf(stderr, "%llu ", sp->cidx[i][n]);
        }
        fprintf(stderr, ")");
        
        fprintf(stderr, " : %f\n", sp->vals[n]);
    } 
}

/**
 * Sorts sparse tensor by stream mode
 * Preprocessing for streaming tensor decomposition
 */
void tensor_sort(SparseTensor * sp, int mode) {
    int * perm = (int*)AlignedMalloc(sizeof(int) * sp->nnz);
    // Init permutation vector
    for (IType i = 0; i < sp->nnz; ++i) {
        perm[i] = i;
    }

    std::sort(perm, perm+sp->nnz, [&](const int& a, const int& b) {
        return (sp->cidx[mode][a] < sp->cidx[mode][b]);
    });

    // for (int ii = 0; ii < sp->nnz; ++ii) {
    //     printf("%d\n", perm[ii]);
    // }

    // sort sp based on permutation
    for (IType i = 0; i < sp->nnz; ++i) {
        int swp_idx = perm[i];
        while (swp_idx < i) {
            swp_idx = perm[swp_idx];
        };
        int tmp;
        float tmp_vals;
        for (int m = 0; m < sp->nmodes; ++m) {
            tmp = sp->cidx[m][swp_idx];
            sp->cidx[m][swp_idx] = sp->cidx[m][i];
            sp->cidx[m][i] = tmp;
        };
        // swap vals
        tmp_vals = sp->vals[swp_idx];
        sp->vals[swp_idx] = sp->vals[i];
        sp->vals[i] = tmp_vals;
    };

}

StreamingSparseTensor::StreamingSparseTensor(
    SparseTensor * sp,
    IType stream_mode
) : _stream_mode(stream_mode),
    _batch_num(0),
    _nnz_ptr(0)
{
    _tensor = sp; // Load tensor

    // First sort tensor based on streaming mode
    // tensor_sort(_tensor, _stream_mode);

    // Straight out of SPLATT codebase
    // Much faster and gives same results as SPLATT 
    // which helps in debugging
    tt_sort(_tensor, _stream_mode, NULL);
    // ExportSparseTensor(NULL, TEXT_FORMAT, _tensor);
    for (int m = 0; m < _tensor->nmodes; ++m) {
        _prev_dim[m] = 0;
    }
    // Allocation permutation array
    _perm = perm_alloc(_tensor->dims, _tensor->nmodes);

    // Store permutation info
    #pragma omp parallel for schedule(static, 1)
    for (int m = 0; m < _tensor->nmodes; ++m) {
        IType * const perm = _perm->perms[m];
        IType * const iperm = _perm->iperms[m];

        if (m == _stream_mode) {
            for (IType i = 0; i < _tensor->dims[m]; ++i) {
                perm[i] = i;
                iperm[i] = i;
            }
            continue;
        }

        // IDX_MAX is the initial value
        for (IType i = 0; i < _tensor->dims[m]; ++i) {
            perm[i] = IDX_MAX;
            iperm[i] = IDX_MAX;
        }

        IType seen = 0;
        IType * const inds = _tensor->cidx[m];
        for (IType n = 0; n < _tensor->nnz; ++n) {
            IType const ind = inds[n];

            if (perm[ind] == IDX_MAX) {
                perm[ind] = seen;
                iperm[seen] = ind;
                ++seen;
            }
            inds[n] = perm[ind];
        }
    }
};

SparseTensor * StreamingSparseTensor::next_batch() {
    
    IType const * const streaming_cidx = _tensor->cidx[_stream_mode];
    // If we're already at the end
    if (_nnz_ptr == _tensor->nnz) {
        return NULL;
    }

    // Find starting nnz
    IType start_nnz = _nnz_ptr;
    while ((start_nnz < _tensor->nnz) && (streaming_cidx[start_nnz] < _batch_num)) {
        ++start_nnz;
    }

    // Find ending nnz
    IType end_nnz = start_nnz;
    while ((end_nnz < _tensor->nnz) && (streaming_cidx[end_nnz] < _batch_num + 1)) {
        ++end_nnz;
    }

    IType nnz = end_nnz - start_nnz;

    // Make sure we don't have empty batches
    assert(nnz > 0);

    // We're still keeping track of the streaming mode
    SparseTensor * t_batch = AllocSparseTensor(nnz, _tensor->nmodes); 
    
    // Copy the values
    ParMemcpy(t_batch->vals, &(_tensor->vals[start_nnz]), nnz * sizeof(*(t_batch->vals)));

    #pragma omp parallel for schedule(static)
    for(IType n=0; n < nnz; ++n) {
        t_batch->cidx[_stream_mode][n] = 0;
        // Just to dump parts of tensor
        // t_batch->cidx[_stream_mode][n] = _batch_num;
    }

    // Need to figure out how to modify the dims and cidx for the batch tensors
    // SPLATT recomputes it in a unique way - should we follow?
    // SPLATT takes care of the cidx by permuting when first loading the nnzs
    // How here its enough to just copy them to the current batch
    // Adjust the size of the dimensions since 
    // it doesn't make sense to keep track of the full sized factor matrix sizes
    for (int m = 0; m < _tensor->nmodes; ++m) {
        if (m == _stream_mode) {
            t_batch->dims[_stream_mode] = 1;
            continue;
        } else {
            ParMemcpy(t_batch->cidx[m], &(_tensor->cidx[m][start_nnz]), nnz * sizeof(*(_tensor->cidx[m])));

            IType dim = 0;
            #pragma omp parallel for schedule(static) reduction(max: dim)
            for (IType n = 0; n < nnz; ++n) {
                dim = SS_MAX(dim, t_batch->cidx[m][n]);
            }
            t_batch->dims[m] = SS_MAX(dim + 1, _prev_dim[m]);
            _prev_dim[m] = t_batch->dims[m];
        }
    }

    _nnz_ptr = end_nnz;
    ++_batch_num;

    _next_batch = t_batch;

    return t_batch;
}

SparseTensor * StreamingSparseTensor::next_dynamic_batch(IType nnz_threshold, IType timeslice_limit) {
    IType _prev_batch_num = _batch_num;

    IType const * const streaming_cidx = _tensor->cidx[_stream_mode];
    // If we're already at the end
    if (_nnz_ptr == _tensor->nnz) {
        return NULL;
    }

    // Find starting nnz
    IType start_nnz = _nnz_ptr;
    while ((start_nnz < _tensor->nnz) && (streaming_cidx[start_nnz] < _batch_num)) {
        ++start_nnz;
    }

    // Find ending nnz
    IType end_nnz = start_nnz;
    vector<IType> streaming_mode_cidx;

    // printf("end_nnz: %d, start_nnz: %d batch_num: %d\n", end_nnz, start_nnz, _batch_num);

    while (end_nnz-start_nnz < nnz_threshold && end_nnz < _tensor->nnz) {
        while ((end_nnz < _tensor->nnz) && (streaming_cidx[end_nnz] < _batch_num + 1)) {
            streaming_mode_cidx.push_back(_batch_num-_prev_batch_num);
            // printf("end_nnz: %d, start_nnz: %d, streaming_cidx[end_nnz]: %d~\n", end_nnz, start_nnz, _batch_num - _prev_batch_num);
            ++end_nnz;
        }
        ++_batch_num;
        if ((_batch_num - _prev_batch_num) >= timeslice_limit) break;
    }
    IType nnz = end_nnz - start_nnz;
    assert(streaming_mode_cidx.size() == nnz);
    // Make sure we don't have empty batches
    assert(nnz > 0);

    // We're still keeping track of the streaming mode
    SparseTensor * t_batch = AllocSparseTensor(nnz, _tensor->nmodes);

    // Copy the values
    ParMemcpy(t_batch->vals, &(_tensor->vals[start_nnz]), nnz * sizeof(*(t_batch->vals)));

    // Need to figure out how to modify the dims and cidx for the batch tensors
    // SPLATT recomputes it in a unique way - should we follow?
    // SPLATT takes care of the cidx by permuting when first loading the nnzs
    // How here its enough to just copy them to the current batch
    // Adjust the size of the dimensions since
    // it doesn't make sense to keep track of the full sized factor matrix sizes
    for (int m = 0; m < _tensor->nmodes; ++m) {
        if (m == _stream_mode) {
            t_batch->dims[_stream_mode] = _batch_num - _prev_batch_num;
            ParMemcpy(t_batch->cidx[_stream_mode], streaming_mode_cidx.data(), nnz * sizeof(*(_tensor->cidx[m])));
            continue;
        } else {
            ParMemcpy(t_batch->cidx[m], &(_tensor->cidx[m][start_nnz]), nnz * sizeof(*(_tensor->cidx[m])));

            IType dim = 0;
            #pragma omp parallel for schedule(static) reduction(max: dim)
            for (IType n = 0; n < nnz; ++n) {
                dim = SS_MAX(dim, t_batch->cidx[m][n]);
            }
            t_batch->dims[m] = SS_MAX(dim + 1, _prev_dim[m]);
            _prev_dim[m] = t_batch->dims[m];
        }
    }

    _nnz_ptr = end_nnz;
    _next_batch = t_batch;
    return t_batch;
}

bool StreamingSparseTensor::last_batch()
{
    return _nnz_ptr == _tensor->nnz;
};

void StreamingSparseTensor::print_tensor_info()
{
    fprintf(stdout, "Streaming sparse tensor info\n");
    fprintf(stdout, "\tstreaming mode:          %d\n", _stream_mode);
    fprintf(stdout, "\tnumber of non-zeros:     %llu\n", _tensor->nnz);
    fprintf(stdout, "\tdimensions:              \n\t\t");
    for (int i = 0; i < _tensor->nmodes; ++i) {
        fprintf(stdout, "%llu", _tensor->dims[i]);
        if (i != _tensor->nmodes - 1) fprintf(stdout, " x ");
        else fprintf(stdout, "\n");
    }
};

SparseTensor * StreamingSparseTensor::stream_prev(IType previous)
{
    if(_batch_num == 0) {
        return NULL;
    }

    IType const start_time = (previous < _batch_num) ? _batch_num - previous : 0;

    /* find start of range */
    IType start_nnz = 0;
    while((start_nnz < _tensor->nnz) &&
            (_tensor->cidx[_stream_mode][start_nnz] < start_time)) {
        ++start_nnz;
    }

    IType const end_nnz = _nnz_ptr;
    IType const nnz = end_nnz - start_nnz;

    /* copy into new tensor */
    SparseTensor * ret = AllocSparseTensor(nnz, _tensor->nmodes);
    ParMemcpy(
        ret->vals,
        &(_tensor->vals[start_nnz]),
        nnz * sizeof(*(ret->vals)));

    /* streaming inds */
    #pragma omp parallel for schedule(static)
    for (IType x=0; x < nnz; ++x) {
        ret->cidx[_stream_mode][x] = _tensor->cidx[_stream_mode][x+start_nnz] - start_time;
    }
    ret->dims[_stream_mode] = _batch_num - start_time; /* may not be previous */

    /* the rest */
    for (IType m=0; m < _tensor->nmodes; ++m) {
        if(m == _stream_mode) {
        continue;
        }

        ret->dims[m] = _prev_dim[m];
        ParMemcpy(ret->cidx[m], &(_tensor->cidx[m][start_nnz]),
            nnz * sizeof(**(ret->cidx)));
    }

    return ret;
};
