#ifndef GPU_NUMERICAL_INTEGRATION
#define GPU_NUMERICAL_INTEGRATION

#include <cuda.h>
#include <vector>

#include <votca/xtp/aobasis.h>
#include <votca/xtp/gpu/GPUAOBasis.cuh>
#include <votca/xtp/gpu/GPUGridBox.cuh>

namespace votca { namespace xtp { namespace gpu {

   class GPUNumericalIntegration{
   public:
       GPUNumericalIntegration(const AOBasis& aob);
       ~GPUNumericalIntegration();
   private:
       int _device;
       const AOBasis* _aob; 
       GPUAOBasis _gpuAOB;
       GPUGridBox _gpuGridBox; 
   };
   
}}}


#endif // GPU_NUMERICAL_INTEGRATION_CUH
