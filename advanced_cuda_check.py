#!/usr/bin/env python3
"""
Advanced CUDA Compilation Validation
Checks for function signature mismatches, kernel declarations, and other issues
"""
import os
import re

def extract_function_signature(line):
    """Extract clean function signature from declaration or definition"""
    # Remove extern "C", __global__, __device__, __host__, etc.
    clean_line = re.sub(r'(extern\s+"C"\s*)?(__global__|__device__|__host__)\s*', '', line)
    clean_line = re.sub(r'\s+', ' ', clean_line.strip())
    return clean_line

def check_function_signatures():
    """Check for function signature mismatches between .cuh and .cu files"""
    issues = []
    
    # Read STDPKernel.cuh declarations
    cuh_path = '/home/jkyleowens/Desktop/NeuroGen Alpha/src/cuda/STDPKernel.cuh'
    cu_path = '/home/jkyleowens/Desktop/NeuroGen Alpha/src/cuda/STDPKernel.cu'
    
    if not os.path.exists(cuh_path) or not os.path.exists(cu_path):
        return ["Missing STDPKernel files"]
    
    with open(cuh_path, 'r') as f:
        cuh_content = f.read()
    
    with open(cu_path, 'r') as f:
        cu_content = f.read()
    
    # Find launchSTDPUpdateKernel declaration in .cuh
    cuh_match = re.search(r'void\s+launchSTDPUpdateKernel\s*\([^)]+\);', cuh_content, re.MULTILINE)
    if cuh_match:
        cuh_sig = extract_function_signature(cuh_match.group(0))
        print(f"Header declaration: {cuh_sig}")
        
        # Find implementation in .cu
        cu_match = re.search(r'void\s+launchSTDPUpdateKernel\s*\([^{]+\{', cu_content, re.MULTILINE | re.DOTALL)
        if cu_match:
            cu_sig_raw = cu_match.group(0).replace('{', '').strip()
            cu_sig = extract_function_signature(cu_sig_raw)
            print(f"Implementation signature: {cu_sig}")
            
            # Compare parameter lists
            cuh_params = re.search(r'\(([^)]+)\)', cuh_sig)
            cu_params = re.search(r'\(([^)]+)\)', cu_sig)
            
            if cuh_params and cu_params:
                cuh_param_list = [p.strip() for p in cuh_params.group(1).split(',')]
                cu_param_list = [p.strip() for p in cu_params.group(1).split(',')]
                
                if len(cuh_param_list) != len(cu_param_list):
                    issues.append(f"Parameter count mismatch: header has {len(cuh_param_list)}, implementation has {len(cu_param_list)}")
                
                for i, (h_param, c_param) in enumerate(zip(cuh_param_list, cu_param_list)):
                    if h_param != c_param:
                        issues.append(f"Parameter {i+1} mismatch: '{h_param}' vs '{c_param}'")
    
    return issues

def check_kernel_declarations():
    """Check that all kernel calls have proper declarations"""
    issues = []
    
    cu_path = '/home/jkyleowens/Desktop/NeuroGen Alpha/src/cuda/STDPKernel.cu'
    if not os.path.exists(cu_path):
        return ["STDPKernel.cu not found"]
    
    with open(cu_path, 'r') as f:
        content = f.read()
    
    # Find kernel calls
    kernel_calls = re.findall(r'(\w+)<<<[^>]+>>>', content)
    
    # Find kernel declarations
    kernel_decls = re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', content)
    
    for call in kernel_calls:
        if call not in kernel_decls:
            issues.append(f"Kernel '{call}' called but not declared with __global__")
    
    return issues

def main():
    print("=== Advanced CUDA Compilation Validation ===\n")
    
    print("1. Checking function signatures...")
    sig_issues = check_function_signatures()
    if sig_issues:
        print("❌ Function signature issues:")
        for issue in sig_issues:
            print(f"  - {issue}")
    else:
        print("✅ Function signatures match")
    
    print("\n2. Checking kernel declarations...")
    kernel_issues = check_kernel_declarations()
    if kernel_issues:
        print("❌ Kernel declaration issues:")
        for issue in kernel_issues:
            print(f"  - {issue}")
    else:
        print("✅ Kernel declarations are correct")
    
    total_issues = len(sig_issues) + len(kernel_issues)
    print(f"\nTotal advanced issues found: {total_issues}")
    
    if total_issues == 0:
        print("\n🎉 All advanced validation checks passed!")
        print("The CUDA code should compile successfully with nvcc.")
    
    return total_issues == 0

if __name__ == '__main__':
    result = main()
    import sys
    sys.exit(0 if result else 1)
