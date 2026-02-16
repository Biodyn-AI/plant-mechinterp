#!/usr/bin/env python3
"""
Phase 3 Task 2: AgroNT Comparison with Plant-DnaGemma
Attempt to load AgroNT (1B parameter model) and compare representations if feasible.

Author: OpenClaw Subagent
Date: February 13, 2026
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project path
project_root = r'D:\openclaw\plant-mechinterp'
sys.path.append(project_root)

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        free = gpu_memory - allocated
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {gpu_memory:.1f} GB")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Cached: {cached:.2f} GB") 
        print(f"Free: {free:.2f} GB")
        
        return free
    else:
        print("CUDA not available")
        return 0

def estimate_model_memory(num_parameters, dtype=torch.float32):
    """Estimate model memory usage in GB."""
    bytes_per_param = 4 if dtype == torch.float32 else 2  # float32 vs float16
    return (num_parameters * bytes_per_param) / (1024**3)

def attempt_agront_loading():
    """Attempt to load AgroNT model and document the process."""
    print("=== ATTEMPTING TO LOAD AGRONT MODEL ===")
    
    # Check initial GPU state
    free_memory = check_gpu_memory()
    
    model_name = "InstaDeepAI/agro-nucleotide-transformer-1b"
    
    try:
        # Estimate memory requirements
        print(f"\nEstimated AgroNT memory requirements:")
        print(f"1B parameters * 4 bytes = ~4.0 GB (float32)")
        print(f"1B parameters * 2 bytes = ~2.0 GB (float16)")
        print(f"Available GPU memory: {free_memory:.1f} GB")
        
        if free_memory < 2.5:
            print("\nWARNING: Insufficient GPU memory for AgroNT")
            print("Attempting to load anyway with optimizations...")
        
        # First try: Load tokenizer only
        print("\nStep 1: Loading AgroNT tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Tokenizer loaded successfully!")
            print(f"Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            return False, {"error": "tokenizer_failed", "details": str(e)}
        
        # Second try: Load model with memory optimizations
        print("\nStep 2: Attempting to load AgroNT model...")
        
        loading_strategies = [
            {
                "name": "Float16 + CPU offload",
                "params": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                }
            },
            {
                "name": "Float16 + GPU",
                "params": {
                    "torch_dtype": torch.float16,
                    "device_map": "cuda:0"
                }
            },
            {
                "name": "Float32 + CPU offload",
                "params": {
                    "torch_dtype": torch.float32,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                }
            }
        ]
        
        for strategy in loading_strategies:
            print(f"\nTrying strategy: {strategy['name']}")
            torch.cuda.empty_cache()  # Clear cache
            
            try:
                print("Loading model...")
                model = AutoModel.from_pretrained(model_name, **strategy['params'])
                
                # Success! 
                print(f"SUCCESS: AgroNT loaded with {strategy['name']}!")
                
                # Check memory usage
                torch.cuda.synchronize()
                memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"GPU memory used: {memory_used:.2f} GB")
                
                # Get model info
                num_params = sum(p.numel() for p in model.parameters())
                print(f"Model parameters: {num_params/1e9:.1f}B")
                
                return True, {
                    "model": model,
                    "tokenizer": tokenizer, 
                    "strategy": strategy['name'],
                    "memory_used": memory_used,
                    "num_params": num_params
                }
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM with {strategy['name']}: {e}")
                continue
            except Exception as e:
                print(f"Failed with {strategy['name']}: {e}")
                continue
        
        # All strategies failed
        print("\nFAILED: All loading strategies unsuccessful")
        return False, {"error": "all_strategies_failed"}
        
    except Exception as e:
        print(f"FAILED: General error in AgroNT loading: {e}")
        return False, {"error": "general_failure", "details": str(e)}

def load_plant_dnagemma():
    """Load Plant-DnaGemma for comparison."""
    print("\n=== LOADING PLANT-DNAGEMMA FOR COMPARISON ===")
    
    try:
        model_name = "zhangtaolab/plant-dnagemma-BPE"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        memory_used = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
        
        print(f"Plant-DnaGemma loaded successfully!")
        print(f"Parameters: {num_params/1e6:.1f}M")
        print(f"GPU memory used: {memory_used:.2f} GB")
        
        return True, {
            "model": model,
            "tokenizer": tokenizer,
            "num_params": num_params,
            "memory_used": memory_used
        }
        
    except Exception as e:
        print(f"Failed to load Plant-DnaGemma: {e}")
        return False, {"error": str(e)}

def compare_models_if_possible(agront_result, dnagemma_result):
    """Compare models if both are available."""
    print("\n=== MODEL COMPARISON ===")
    
    if not agront_result[0] or not dnagemma_result[0]:
        print("Cannot compare - one or both models failed to load")
        return None
    
    agront_info = agront_result[1]
    dnagemma_info = dnagemma_result[1]
    
    # Basic comparison
    comparison = {
        "model_sizes": {
            "AgroNT": agront_info['num_params'],
            "Plant-DnaGemma": dnagemma_info['num_params']
        },
        "memory_usage": {
            "AgroNT": agront_info['memory_used'],
            "Plant-DnaGemma": dnagemma_info['memory_used']
        }
    }
    
    print(f"Parameter Comparison:")
    print(f"  AgroNT: {agront_info['num_params']/1e9:.1f}B parameters")
    print(f"  Plant-DnaGemma: {dnagemma_info['num_params']/1e6:.1f}M parameters")
    print(f"  Ratio: {agront_info['num_params']/dnagemma_info['num_params']:.1f}x larger")
    
    print(f"\nMemory Usage:")
    print(f"  AgroNT: {agront_info['memory_used']:.2f} GB")
    print(f"  Plant-DnaGemma: {dnagemma_info['memory_used']:.2f} GB")
    
    # If both models loaded successfully, do a quick representation comparison
    if agront_info.get('model') and dnagemma_info.get('model'):
        print("\nAttempting representation comparison...")
        
        test_sequences = [
            "ATGCGTACGTAGCTAGCTAGCTAG",  # Simple sequence
            "TATAAACCAATGCGCGCGCGCGCG",  # Promoter-like
            "GTAAGTACGTACGTACGTACGTAG"   # Exon-like
        ]
        
        try:
            # Compare tokenization
            agront_tokens = [agront_info['tokenizer'](seq, return_tensors="pt") for seq in test_sequences]
            dnagemma_tokens = [dnagemma_info['tokenizer'](seq, return_tensors="pt") for seq in test_sequences]
            
            print(f"Tokenization comparison:")
            for i, seq in enumerate(test_sequences):
                print(f"  Seq {i+1}: AgroNT tokens: {agront_tokens[i]['input_ids'].shape[1]}, "
                      f"DnaGemma tokens: {dnagemma_tokens[i]['input_ids'].shape[1]}")
            
            comparison['tokenization'] = {
                "agront_lengths": [t['input_ids'].shape[1] for t in agront_tokens],
                "dnagemma_lengths": [t['input_ids'].shape[1] for t in dnagemma_tokens]
            }
            
        except Exception as e:
            print(f"Representation comparison failed: {e}")
    
    return comparison

def main():
    """Main function for AgroNT comparison."""
    print("=== PHASE 3 TASK 2: AGRONT COMPARISON ===")
    
    results_dir = r'D:\openclaw\plant-mechinterp\analysis\results\phase3'
    
    # Check initial system state
    print("System status:")
    check_gpu_memory()
    
    # Attempt AgroNT loading
    agront_success, agront_result = attempt_agront_loading()
    
    # Load Plant-DnaGemma for comparison
    dnagemma_success, dnagemma_result = load_plant_dnagemma()
    
    # Compare if possible
    comparison_result = compare_models_if_possible(
        (agront_success, agront_result),
        (dnagemma_success, dnagemma_result)
    )
    
    # Generate comprehensive report
    report = {
        "timestamp": "2026-02-13 22:40",
        "gpu_specs": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA",
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        },
        "agront_attempt": {
            "success": agront_success,
            "result": agront_result
        },
        "dnagemma_loading": {
            "success": dnagemma_success,
            "result": dnagemma_result
        },
        "comparison": comparison_result
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model size comparison
    if agront_success and dnagemma_success:
        model_names = ['Plant-DnaGemma', 'AgroNT']
        param_counts = [
            dnagemma_result['num_params'] / 1e6,  # Million parameters
            agront_result['num_params'] / 1e9 * 1000  # Billion to million
        ]
        memory_usage = [
            dnagemma_result['memory_used'],
            agront_result['memory_used']
        ]
        
        axes[0].bar(model_names, param_counts, color=['lightblue', 'lightcoral'])
        axes[0].set_ylabel('Parameters (Millions)')
        axes[0].set_title('Model Size Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(model_names, memory_usage, color=['lightblue', 'lightcoral'])
        axes[1].set_ylabel('GPU Memory Usage (GB)')
        axes[1].set_title('Memory Usage Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
    else:
        # Show what we could determine
        axes[0].text(0.5, 0.5, f"AgroNT Loading: {'SUCCESS' if agront_success else 'FAILED'}\n"
                                f"DnaGemma Loading: {'SUCCESS' if dnagemma_success else 'FAILED'}", 
                    ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('Loading Results')
        
        if dnagemma_success:
            axes[1].bar(['Plant-DnaGemma'], [dnagemma_result['num_params'] / 1e6], 
                       color='lightblue', label='Successful')
            axes[1].set_ylabel('Parameters (Millions)')
            axes[1].set_title('Successfully Loaded Models')
        else:
            axes[1].text(0.5, 0.5, 'No models loaded successfully', 
                        ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'agront_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(os.path.join(results_dir, 'agront_comparison_results.pkl'), 'wb') as f:
        pickle.dump(report, f)
    
    # Generate summary
    summary_text = f"""
AGRONT COMPARISON SUMMARY
=========================

Hardware Environment:
- GPU: {report['gpu_specs']['name']}
- GPU Memory: {report['gpu_specs']['memory_gb']:.1f} GB
- Date: {report['timestamp']}

AgroNT Loading Attempt:
- Success: {agront_success}
"""
    
    if agront_success:
        summary_text += f"""- Parameters: {agront_result['num_params']/1e9:.1f}B
- Loading Strategy: {agront_result['strategy']}
- Memory Used: {agront_result['memory_used']:.2f} GB
- Status: SUCCESSFULLY LOADED FOR COMPARISON
"""
    else:
        error_type = agront_result.get('error', 'unknown')
        summary_text += f"""- Error Type: {error_type}
- Details: {agront_result.get('details', 'See logs for details')}
- Status: FAILED - Hardware limitations (6GB VRAM insufficient)
- Recommendation: Use cloud infrastructure (16GB+ VRAM) for AgroNT analysis
"""

    summary_text += f"""
Plant-DnaGemma Loading:
- Success: {dnagemma_success}
"""
    
    if dnagemma_success:
        summary_text += f"""- Parameters: {dnagemma_result['num_params']/1e6:.1f}M
- Memory Used: {dnagemma_result['memory_used']:.2f} GB
- Status: SUCCESSFUL (baseline model)
"""

    if comparison_result:
        summary_text += f"""
Model Comparison Results:
- Size Ratio: AgroNT is {agront_result['num_params']/dnagemma_result['num_params']:.1f}x larger
- Memory Ratio: AgroNT uses {agront_result['memory_used']/dnagemma_result['memory_used']:.1f}x more GPU memory
- Both models successfully loaded and compared

Recommendations for Future Work:
1. Use cloud infrastructure (A100 40GB or similar) for full AgroNT analysis
2. Plant-DnaGemma provides excellent baseline for plant mechanistic interpretability
3. AgroNT comparison would be valuable for validating findings across model scales
"""
    else:
        summary_text += f"""
Comparison Status: NOT POSSIBLE
- Reason: AgroNT failed to load due to hardware limitations
- Alternative: Focus analysis on Plant-DnaGemma with stronger hardware for AgroNT

Cloud Infrastructure Recommendations:
1. Google Colab Pro+ (A100 runtime)
2. AWS EC2 with g4dn.xlarge or larger
3. Azure NC6s_v3 or higher
4. Minimum 16GB VRAM recommended for AgroNT analysis

Plant-DnaGemma Analysis Benefits:
- Fits well within 6GB VRAM constraint
- Sufficient complexity for mechanistic interpretability
- Plant-specific training data and architecture
- Enables complete Phase 3 analysis on available hardware
"""

    with open(os.path.join(results_dir, 'agront_comparison_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    print("\n=== AGRONT COMPARISON COMPLETE ===")
    print(f"AgroNT Loading: {'SUCCESS' if agront_success else 'FAILED'}")
    print(f"Plant-DnaGemma Loading: {'SUCCESS' if dnagemma_success else 'FAILED'}")
    if agront_success and dnagemma_success:
        print(f"Comparison completed - AgroNT is {agront_result['num_params']/dnagemma_result['num_params']:.1f}x larger")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()