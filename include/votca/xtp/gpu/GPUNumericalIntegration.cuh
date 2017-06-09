#ifndef GPU_AO_BASIS_CUH
#define GPU_AO_BASIS_CUH

#include <votca/xtp/aobasis.h>

#include <cuda.h>
#include <vector>

#include <votca/xtp/gpu/GPUNumericalIntegrationTools.cuh>
#include <votca/xtp/aobasis.h>

namespace votca { namespace xtp { namespace gpu {

   class GPUNumericalIntegration{
   public:
       GPUNumericalIntegration(const AOBasis& aob);
       ~GPUNumericalIntegration();
   private:
       int _device;
       gpu_ao_basis _gpuAOB; 
       std::vector<int> _shellMap; 
   };
   
}}}


#endif // GPU_AO_BASIS_CUH
