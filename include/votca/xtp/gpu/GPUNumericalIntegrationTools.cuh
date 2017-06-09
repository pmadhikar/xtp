#ifndef GPU_NUMERICAL_INTEGRATION_TOOLS_CUH
#include <helper_cuda.h>
#include <iostream>

#define CUDA_API_CALL(api_call, abort){                      \
        cudaErrorCheck(api_call, __FILE__, __LINE__, abort); \
    }

inline void cudaErrorCheck(const cudaError_t e, const char* file, const int line, bool abort){
    if (e!=cudaSuccess){
        std::cerr << "Cuda error " << _cudaGetErrorEnum(e) << std::endl;               
        std::cerr << "  Error Message: " << cudaGetErrorString(e) << std::endl;        
        std::cerr << "  Occured in " << file << "(" << line << ")" <<  std::endl; 
        if (abort) exit(EXIT_FAILURE);
    }
}

struct gpu_ao_basis{
    double *sBlock;
    double *pBlock;
    double *dBlock;
    double *fBlock;
    double *gBlock;
    
    size_t nS; 
    size_t nP; 
    size_t nD; 
    size_t nF; 
    size_t nG;
    
};

typedef struct gpu_ao_basis gpu_ao_basis;


inline void allocateGPUAOBasis(gpu_ao_basis& gaob){
    CUDA_API_CALL(cudaMalloc((void **) (gaob.sBlock), gaob.nS*sizeof(double)), true);
    CUDA_API_CALL(cudaMalloc((void **) (gaob.fBlock), gaob.nP*sizeof(double)), true);
    CUDA_API_CALL(cudaMalloc((void **) (gaob.dBlock), gaob.nD*sizeof(double)), true);
    CUDA_API_CALL(cudaMalloc((void **) (gaob.fBlock), gaob.nF*sizeof(double)), true);
    CUDA_API_CALL(cudaMalloc((void **) (gaob.gBlock), gaob.nG*sizeof(double)), true);
}

inline void freeGPUAOBasis(gpu_ao_basis& gaob){
    CUDA_API_CALL(cudaFree(gaob.sBlock), true);
    CUDA_API_CALL(cudaFree(gaob.pBlock), true);
    CUDA_API_CALL(cudaFree(gaob.dBlock), true);
    CUDA_API_CALL(cudaFree(gaob.fBlock), true);
    CUDA_API_CALL(cudaFree(gaob.gBlock), true);
}
#endif // GPU_NUMERICAL_INTEGRATION_TOOLS_CUH
