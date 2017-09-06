#ifndef GPU_AO_BASIS_CUH
#define GPU_AO_BASIS_CUH

#include <votca/xtp/gpu/GPUTools.cuh>

namespace votca { namespace xtp { namespace gpu {

class GPUAOBasis{
public:
    thrust_vector sConts;
    thrust_vector pConts;
    thrust_vector dConts;
    thrust_vector fConts;
    thrust_vector gConts;
    thrust_vector alphas;
    thrust_vector powFactors;

    GPUAOBasis::GPUAOBasis(const AOBasis& aob){
    for (AOBasis::AOShellIterator row = aob.firstShell(); row != aob.lastShell(); row++){
        const AOShell* shell = aob.getShell(row); 
        const std::string shell_type = shell->getType();
        
        for (AOShell::GaussianIterator itr = shell->firstGaussian(); itr != shell->lastGaussian(); ++itr){
            const std::vector<double>& contractions = (*itr)->getContraction();

            alphas.push_back((*itr)->getDecay());
            powFactors.push_back((*itr)->getPowfactor());
            
            for (const char& c: shell_type){
                switch (::toupper(c)){
                case 'S':
                    sConts.push_back(contractions[0]); 
                case 'P':
                    pConts.push_back(contractions[1]);
                case 'D':
                    dConts.push_back(contractions[2]);
                case 'F':
                    fConts.push_back(contractions[3]);
                case 'G':
                    gConts.push_back(contractions[4]);
                case 'H':
                    std::cerr << "H functions not implemented at the moment!" << std::endl;
                    exit(EXIT_FAILURE);
                default:
                    std::cerr << "Shell type " << c << "unknown" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
        
    Pad(sConts, 16);
    Pad(pConts, 16);
    Pad(dConts, 16);
    Pad(fConts, 16);
    Pad(gConts, 16);
    Pad(alphas, 16);
    Pad(powFactors, 16);
        
    d_sConts = gpu_vector(sConts);
    d_pConts = gpu_vector(pConts);
    d_dConts = gpu_vector(dConts);
    d_fConts = gpu_vector(fConts);
    d_gConts = gpu_vector(gConts);
    d_alphas = gpu_vector(alphas);
        
    d_powFactors = gpu_vector(powFactors);
        
    d_expoFactors = gpu_vector(alphas.size(), 0);

    d_sFuncVals = gpu_vector(sConts.size(), 0); 
    d_pFuncVals = gpu_vector(pConts.size(), 0); 
    d_dFuncVals = gpu_vector(dConts.size(), 0); 
    d_fFuncVals = gpu_vector(fConts.size(), 0); 
    d_gFuncVals = gpu_vector(gConts.size(), 0); 

    // build the pointer struct that will be passed to the gpu kernels. 
    rawGpuArrs.sConts.array = thrust::raw_pointer_cast(&d_sConts[0]);
    rawGpuArrs.sConts.arraySize = d_sConts.size();

    rawGpuArrs.pConts.array = thrust::raw_pointer_cast(&d_pConts[0]);
    rawGpuArrs.pConts.arraySize = d_pConts.size();
        
    rawGpuArrs.dConts.array = thrust::raw_pointer_cast(&d_dConts[0]);
    rawGpuArrs.dConts.arraySize = d_dConts.size();

    rawGpuArrs.fConts.array = thrust::raw_pointer_cast(&d_fConts[0]);
    rawGpuArrs.fConts.arraySize = d_fConts.size();

    rawGpuArrs.gConts.array = thrust::raw_pointer_cast(&d_gConts[0]);
    rawGpuArrs.gConts.arraySize = d_gConts.size();

    rawGpuArrs.alphas.array = thrust::raw_pointer_cast(&d_alphas[0]);
    rawGpuArrs.alphas.arraySize = d_alphas.size();

    rawGpuArrs.powFactors.array = thrust::raw_pointer_cast(&d_powFactors[0]);
    rawGpuArrs.powFactors.arraySize = d_powFactors.size();

    rawGpuArrs.expoFactors.array = thrust::raw_pointer_cast(&d_expoFactors[0]);
    rawGpuArrs.expoFactors.arraySize = d_expoFactors.size();

    rawGpuArrs.sFuncVals.array = thrust::raw_pointer_cast(&d_sFuncVals[0]);
    rawGpuArrs.sFuncVals.arraySize = d_sFuncVals.size();

    rawGpuArrs.pFuncVals.array = thrust::raw_pointer_cast(&d_pFuncVals[0]);
    rawGpuArrs.pFuncVals.arraySize = d_pFuncVals.size();

    rawGpuArrs.dFuncVals.array = thrust::raw_pointer_cast(&d_dFuncVals[0]);
    rawGpuArrs.dFuncVals.arraySize = d_dFuncVals.size();

    rawGpuArrs.fFuncVals.array = thrust::raw_pointer_cast(&d_fFuncVals[0]);
    rawGpuArrs.fFuncVals.arraySize = d_fFuncVals.size();

    rawGpuArrs.gFuncVals.array = thrust::raw_pointer_cast(&d_gFuncVals[0]);
    rawGpuArrs.gFuncVals.arraySize = d_gFuncVals.size();
}
    
    GPUAOBasis();
    ~GPUAOBasis();

    gpuAOArrs GetRawGPUArrss(){
        return rawGpuArrs;
    }

private:
    gpuAOArrs rawGpuArrs; 

    size_t padding = 16;
    
    gpu_vector d_sConts;
    gpu_vector d_pConts;
    gpu_vector d_dConts;
    gpu_vector d_fConts;
    gpu_vector d_gConts;
    gpu_vector d_alphas;
    gpu_vector d_powFactors;

    gpu_vector d_expoFactors;

    gpu_vector d_sFuncVals; 
    gpu_vector d_pFuncVals; 
    gpu_vector d_dFuncVals; 
    gpu_vector d_fFuncVals; 
    gpu_vector d_gFuncVals; 
};




}}} 
#endif // GPU_AO_BASIS_CUH
