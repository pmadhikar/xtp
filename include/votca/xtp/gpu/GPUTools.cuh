#ifndef GPU_TOOLS_CUH
#define GPU_TOOLS_CUH
#include <helper_cuda.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "GPUCTypes.cuh"

namespace votca { namespace xtp { namespace gpu {

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

typedef thrust::host_vector<double> thrust_vector; // vector that lives on the cpu (host)
typedef thrust::device_vector<double> gpu_vector; // vector that lives on the gpu (device)

struct thrust_R3N_vectors{
    thrust_vector x;
    thrust_vector y;
    thrust_vector z;
};

struct device_R3N_vectors{
    gpu_vector x;
    gpu_vector y;
    gpu_vector z;

    device_R3N_vectors();
    device_R3N_vectors(thrust_R3N_vectors& cpu): x(cpu.x), y(cpu.y), z(cpu.z){
    }
}; 

void Pad(thrust_vector& a, size_t p){
    size_t n = 0;
    if (a.size()%p > 0){
        n = a.size()/p + 1;
        a.resize(n, 0);
    }
}

void Pad(thrust_R3N_vectors a, size_t p){
    Pad(a.x, p);
    Pad(a.y, p);
    Pad(a.z, p);
}

}}}
#endif // GPU_TOOLS_CUH
