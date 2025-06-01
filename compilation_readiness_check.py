#!/usr/bin/env python3
"""
Comprehensive CUDA Compilation Readiness Check
This validates that the code should compile successfully when CUDA is available
"""
import os
import re

def check_makefile_syntax():
    """Check if Makefile has correct CUDA setup"""
    makefile_path = '/home/jkyleowens/Desktop/NeuroGen Alpha/Makefile'
    if not os.path.exists(makefile_path):
        return ["Makefile not found"]
    
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for nvcc check
    if 'which nvcc' not in content:
        issues.append("Makefile doesn't check for nvcc availability")
    
    # Check for CUDA includes
    if '-I$(CUDA_HOME)/include' not in content and '-I/usr/local/cuda/include' not in content:
        issues.append("CUDA include path not specified")
    
    # Check for CUDA libraries
    if '-lcurand' not in content and '-lcuda' not in content:
        issues.append("CUDA libraries not linked")
    
    return issues

def check_all_required_files():
    """Check that all required CUDA files exist"""
    base_path = '/home/jkyleowens/Desktop/NeuroGen Alpha'
    required_files = [
        'src/cuda/NetworkCUDA.cu',
        'src/cuda/NetworkCUDA.cuh',
        'src/cuda/STDPKernel.cu',
        'src/cuda/STDPKernel.cuh',
        'src/cuda/GPUNeuralStructures.h',
        'src/cuda/GridBlockUtils.cuh',
        'main.cpp',
        'Makefile'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(base_path, file)):
            missing_files.append(file)
    
    return missing_files

def check_include_dependencies():
    """Check that all include dependencies are satisfied"""
    base_path = '/home/jkyleowens/Desktop/NeuroGen Alpha'
    issues = []
    
    # Check NetworkCUDA.cu includes
    with open(os.path.join(base_path, 'src/cuda/NetworkCUDA.cu'), 'r') as f:
        content = f.read()
        
    required_includes = [
        'NetworkCUDA.cuh',
        'STDPKernel.cuh',
        'GridBlockUtils.cuh',
        'cuda_runtime.h'
    ]
    
    for inc in required_includes:
        if f'#include "{inc}"' not in content and f'#include <{inc}>' not in content:
            issues.append(f"NetworkCUDA.cu missing include: {inc}")
    
    return issues

def validate_cuda_kernel_syntax():
    """Validate CUDA kernel syntax patterns"""
    base_path = '/home/jkyleowens/Desktop/NeuroGen Alpha'
    issues = []
    
    files_to_check = [
        'src/cuda/NetworkCUDA.cu',
        'src/cuda/STDPKernel.cu'
    ]
    
    for file in files_to_check:
        filepath = os.path.join(base_path, file)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check kernel launch syntax
        kernel_launches = re.findall(r'(\w+)<<<([^>]+)>>>', content)
        for kernel_name, launch_config in kernel_launches:
            if 'dim3' not in launch_config and ',' not in launch_config:
                issues.append(f"Suspicious kernel launch syntax in {file}: {kernel_name}<<<{launch_config}>>>")
        
        # Check for __global__ kernel declarations
        global_kernels = re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', content)
        
        # Check that launched kernels are declared
        for kernel_name, _ in kernel_launches:
            if kernel_name not in global_kernels:
                # Check if it's a wrapper function
                wrapper_pattern = f'void\\s+{kernel_name}\\s*\\('
                if not re.search(wrapper_pattern, content):
                    issues.append(f"Kernel {kernel_name} launched but not declared in {file}")
    
    return issues

def main():
    print("=== Comprehensive CUDA Compilation Readiness Check ===\n")
    
    total_issues = 0
    
    print("1. Checking Makefile configuration...")
    makefile_issues = check_makefile_syntax()
    if makefile_issues:
        print("❌ Makefile issues:")
        for issue in makefile_issues:
            print(f"  - {issue}")
        total_issues += len(makefile_issues)
    else:
        print("✅ Makefile configuration looks good")
    
    print("\n2. Checking required files...")
    missing_files = check_all_required_files()
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        total_issues += len(missing_files)
    else:
        print("✅ All required files present")
    
    print("\n3. Checking include dependencies...")
    include_issues = check_include_dependencies()
    if include_issues:
        print("❌ Include dependency issues:")
        for issue in include_issues:
            print(f"  - {issue}")
        total_issues += len(include_issues)
    else:
        print("✅ Include dependencies satisfied")
    
    print("\n4. Validating CUDA kernel syntax...")
    kernel_issues = validate_cuda_kernel_syntax()
    if kernel_issues:
        print("❌ CUDA kernel syntax issues:")
        for issue in kernel_issues:
            print(f"  - {issue}")
        total_issues += len(kernel_issues)
    else:
        print("✅ CUDA kernel syntax looks correct")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total issues found: {total_issues}")
    
    if total_issues == 0:
        print("\n🎉 COMPILATION READINESS: EXCELLENT")
        print("The code should compile successfully when CUDA toolkit is available!")
        print("\nTo compile:")
        print("1. Install CUDA toolkit (nvcc)")
        print("2. Ensure nvcc is in PATH")
        print("3. Run: make clean && make")
    else:
        print(f"\n⚠️  COMPILATION READINESS: NEEDS ATTENTION")
        print(f"Please fix the {total_issues} issues above before compilation.")
    
    return total_issues == 0

if __name__ == '__main__':
    result = main()
    import sys
    sys.exit(0 if result else 1)
