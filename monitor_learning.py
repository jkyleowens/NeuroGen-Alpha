#!/usr/bin/env python3
"""
Real-time monitoring of neural network learning progress
Run this alongside the trading simulation to track performance
"""

import time
import subprocess
import json
import os
from datetime import datetime

def monitor_gpu_usage():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_used, memory_total, gpu_util = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': int(memory_used),
                'memory_total_mb': int(memory_total),
                'memory_percent': round(int(memory_used) / int(memory_total) * 100, 1),
                'gpu_utilization': int(gpu_util)
            }
    except:
        pass
    return None

def check_simulation_running():
    """Check if neural_sim is currently running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'neural_sim'], capture_output=True)
        return result.returncode == 0
    except:
        return False

def parse_log_output():
    """Parse recent log output for performance metrics"""
    # This would parse stdout or log files from the simulation
    # For now, return dummy data
    return {
        'portfolio_value': 105000.0,
        'decisions_made': 15430,
        'current_epoch': 2,
        'last_action': 'buy',
        'reward': 1250.0
    }

def main():
    print("=== Neural Network Learning Monitor ===")
    print(f"Started at: {datetime.now()}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    try:
        while True:
            # Clear screen (ANSI escape sequence)
            print("\033[2J\033[H", end="")
            
            print("=== Neural Network Learning Monitor ===")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 50)
            
            # Check if simulation is running
            sim_running = check_simulation_running()
            print(f"Simulation Status: {'🟢 RUNNING' if sim_running else '🔴 STOPPED'}")
            
            # GPU metrics
            gpu_info = monitor_gpu_usage()
            if gpu_info:
                print(f"GPU Memory: {gpu_info['memory_used_mb']} / {gpu_info['memory_total_mb']} MB ({gpu_info['memory_percent']}%)")
                print(f"GPU Utilization: {gpu_info['gpu_utilization']}%")
            else:
                print("GPU metrics unavailable")
            
            # Simulation metrics (if running)
            if sim_running:
                metrics = parse_log_output()
                print(f"Portfolio Value: ${metrics['portfolio_value']:,.2f}")
                print(f"Decisions Made: {metrics['decisions_made']:,}")
                print(f"Current Epoch: {metrics['current_epoch']}")
                print(f"Last Action: {metrics['last_action']}")
                print(f"Last Reward: {metrics['reward']:+.2f}")
            
            print("-" * 50)
            print("Neural Network Architecture:")
            print("  Input → Hidden → Output")
            print("  60 → 512 → 3 neurons")
            print("  ~40K synapses learning via STDP")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
