#ifndef GPU_CTYPES_CUH
#define GPU_CTYPES_CUH

typedef struct{
        double* array;
        size_t arraySize;
} gpuArr;

typedef struct{
    gpuArr sConts; 
    gpuArr pConts; 
    gpuArr dConts; 
    gpuArr fConts; 
    gpuArr gConts;

    gpuArr alphas;
    gpuArr powFactors;
    gpuArr expoFactors;

    gpuArr sFuncVals; 
    gpuArr pFuncVals; 
    gpuArr dFuncVals; 
    gpuArr fFuncVals; 
    gpuArr gFuncVals; 
         
} gpuAOArrs; 


typedef struct{
    double* x;
    double* y;
    double* z;

    size_t n;
} R3Nptrs; 

#endif // GPU_CTYPES_CUH
