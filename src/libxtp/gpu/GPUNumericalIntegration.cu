#include <algorithm>
#include <string>
#include <votca/xtp/aoshell.h>
#include <thrust/host_vector.h>
#include <votca/xtp/gpu/GPUNumericalIntegration.cuh>

namespace votca { namespace xtp { namespace gpu {

GPUNumericalIntegration::GPUNumericalIntegration(const AOBasis& aob): _device(0), _aob(&aob){
    CUDA_API_CALL(cudaSetDevice(_device), true); // add some code here later to intelligently
                                                 // select a gpu
    
    // Later we need to also dissallow this gpu from begin used
    // by other processes (optionally?)...
    
}

}}}
