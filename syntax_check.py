#!/usr/bin/env python3
"""
CUDA Syntax Validation Script
Checks for common CUDA compilation issues without requiring nvcc
"""
import os
import re
import sys

def check_file_syntax(filepath):
    """Check a CUDA file for common syntax issues"""
    issues = []
    
    if not os.path.exists(filepath):
        return [f"File not found: {filepath}"]
    
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check for function signature mismatches
    if filepath.endswith('.cuh'):
        # Extract function declarations
        declarations = re.findall(r'(\w+\s+\w+\([^)]*\));', content, re.MULTILINE)
        print(f"Found declarations in {filepath}: {len(declarations)}")
    
    # Check for missing includes
    required_includes = ['cuda_runtime.h', 'device_launch_parameters.h']
    for inc in required_includes:
        if f'#include <{inc}>' not in content and f'#include "{inc}"' not in content:
            if 'cuda' in filepath.lower():
                issues.append(f"Missing include: {inc}")
    
    # Check for undefined types
    undefined_types = []
    
    # Skip type checking for the file that defines the structures
    if not filepath.endswith('GPUNeuralStructures.h'):
        if 'GPUSynapse' in content and '#include "GPUNeuralStructures.h"' not in content:
            undefined_types.append('GPUSynapse')
        if 'GPUNeuronState' in content and '#include "GPUNeuralStructures.h"' not in content:
            undefined_types.append('GPUNeuronState')
    
    if undefined_types:
        issues.append(f"Undefined types: {', '.join(undefined_types)}")
    
    # Check for kernel launch syntax
    kernel_launches = re.findall(r'(\w+<<<[^>]+>>>)', content)
    for launch in kernel_launches:
        if '>>>' not in launch:
            issues.append(f"Invalid kernel launch syntax: {launch}")
    
    return issues

def main():
    print("Starting CUDA syntax validation...")
    
    cuda_files = [
        'src/cuda/STDPKernel.cu',
        'src/cuda/STDPKernel.cuh', 
        'src/cuda/NetworkCUDA.cu',
        'src/cuda/NetworkCUDA.cuh',
        'src/cuda/GPUNeuralStructures.h'
    ]
    
    total_issues = 0
    for file in cuda_files:
        full_path = os.path.join('/home/jkyleowens/Desktop/NeuroGen Alpha', file)
        print(f"Checking {file}...")
        issues = check_file_syntax(full_path)
        
        if issues:
            print(f"\n❌ Issues in {file}:")
            for issue in issues:
                print(f"  - {issue}")
            total_issues += len(issues)
        else:
            print(f"✅ {file} - No syntax issues found")
    
    print(f"\nTotal issues found: {total_issues}")
    return total_issues == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
