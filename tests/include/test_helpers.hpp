#include "matrix.hpp"
#include "sptensor.hpp"
#include "streaming_sptensor.hpp"
#include "kruskal_model.hpp"
#include "gram.hpp"
#include "mttkrp.hpp"
#include "alto.hpp"
#include "common.hpp"

#define DATASET(x) "./tests/tensors/" #x

SparseTensor * load_tensor(void);