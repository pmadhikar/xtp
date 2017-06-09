#include <votca/xtp/gpu/GPUNumericalIntegration.cuh>
#include <votca/xtp/aoshell.h>
#include <cstdlib>

namespace votca { namespace xtp { namespace gpu {

// GPUNumericalIntegration
GPUNumericalIntegration::GPUNumericalIntegration(const AOBasis& aob):
_shellMap(aob.getNumofShells(), -1)
{
    // first check if there is a gpu and select it
    
    CUDA_API_CALL(cudaSetDevice(0), true); // add some code here later to inteligently
                                           // select a gpu
    _device = 0; 
    // Later we need to also dissallow this gpu from begin used
    // by other processes...
    unsigned int numFuncs = 0;
    unsigned int shellInd = 0;
    for (AOBasis::AOShellIterator row = aob.firstShell(); row != aob.lastShell(); row++){
        int numFunInShell = 0;
        const AOShell* shell = aob.getShell(row); 
        const std::string shell_type = shell->getType();
        for (const char& c : shell_type){
            switch (c){
            case 'S':
                numFunInShell += 1;
                _gpuAOB.nS += 1; 
            case 'P':
                numFunInShell += 3;
                _gpuAOB.nP += 1; 
            case 'D':
                numFunInShell += 5;
                _gpuAOB.nD += 1; 
            case 'F':
                numFunInShell += 7;
                _gpuAOB.nF += 1; 
            case 'G':
                numFunInShell += 9;
                _gpuAOB.nG += 1; 
            case 'H':
                std::cerr << "H functions not implemented at the moment!" << std::endl;
                exit(EXIT_FAILURE);
            default:
                std::cerr << "Single shell type " << c << "unknown" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        numFuncs += numFunInShell;
        _shellMap[shellInd] = shell->getStartIndex();

        allocateGPUAOBasis(_gpuAOB);
    }
    
}

GPUNumericalIntegration::~GPUNumericalIntegration(){
    freeGPUAOBasis(_gpuAOB);
}

}}}
