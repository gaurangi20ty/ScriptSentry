"""
Evaluation utilities for continual learning memory analysis.
Loads and analyzes accuracy matrices from checkpoints and result files.
"""
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class MemoryMetrics:
    """Container for continual learning evaluation metrics."""
    accuracy_matrix: np.ndarray  # 5x5 matrix
    per_task_peak_accuracy: np.ndarray  # Max accuracy per task (before forgetting)
    per_task_final_accuracy: np.ndarray  # Final accuracy per task
    per_task_forgetting: np.ndarray  # Peak - Final (forgetting score)
    average_forgetting: float  # Mean forgetting across tasks
    knowledge_retention: float  # (1 - avg_forgetting/100) * 100
    most_forgotten_task: int  # Task index with highest forgetting
    most_retained_task: int  # Task index with lowest forgetting


def load_accuracy_matrix_from_checkpoint(
    checkpoint_path: str,
    method: str = "best",
) -> np.ndarray:
    """
    Load accuracy matrix from a PyTorch checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        method: Which matrix to load ("best", "fc", "nme")
        
    Returns:
        5x5 numpy accuracy matrix
    """
    import torch
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try the requested method first
        matrix_key = f"{method}_matrix"
        if matrix_key in ckpt:
            return ckpt[matrix_key]
        
        # Fall back to best_matrix
        if "best_matrix" in ckpt:
            return ckpt["best_matrix"]
        
        raise ValueError(f"No accuracy matrix found in checkpoint")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")


def load_accuracy_matrix_from_npy(npy_path: str) -> np.ndarray:
    """
    Load accuracy matrix from a .npy file.
    
    Args:
        npy_path: Path to .npy file
        
    Returns:
        5x5 numpy accuracy matrix
    """
    try:
        matrix = np.load(npy_path)
        if matrix.shape != (5, 5):
            raise ValueError(f"Expected 5x5 matrix, got {matrix.shape}")
        return matrix
    except Exception as e:
        raise RuntimeError(f"Failed to load npy file {npy_path}: {str(e)}")


def compute_memory_metrics(accuracy_matrix: np.ndarray) -> MemoryMetrics:
    """
    Compute continual learning memory metrics from a 5x5 accuracy matrix.
    Matrix format: rows=semester, cols=task (lower triangular with diagonal)
    
    Args:
        accuracy_matrix: 5x5 numpy array where [i,j] = accuracy on task j after semester i
        
    Returns:
        MemoryMetrics dataclass with all computed metrics
    """
    matrix = accuracy_matrix.copy()
    
    # Compute per-task peak accuracy (diagonal values - when task was first learned)
    per_task_peak = np.diag(matrix)
    
    # Compute per-task final accuracy (last row - when model has learned all tasks)
    per_task_final = matrix[-1, :]
    
    # Compute forgetting: peak - final (how much accuracy dropped on each task)
    per_task_forgetting = per_task_peak - per_task_final
    
    # Average forgetting (mean across all tasks)
    avg_forgetting = np.mean(per_task_forgetting)
    
    # Knowledge retention: percentage of knowledge retained (higher is better)
    knowledge_retention = max(0, 100.0 - avg_forgetting)
    
    # Most forgotten task (highest forgetting)
    most_forgotten_idx = int(np.argmax(per_task_forgetting))
    
    # Most retained task (lowest forgetting)
    most_retained_idx = int(np.argmin(per_task_forgetting))
    
    return MemoryMetrics(
        accuracy_matrix=matrix,
        per_task_peak_accuracy=per_task_peak,
        per_task_final_accuracy=per_task_final,
        per_task_forgetting=per_task_forgetting,
        average_forgetting=float(avg_forgetting),
        knowledge_retention=float(knowledge_retention),
        most_forgotten_task=most_forgotten_idx,
        most_retained_task=most_retained_idx,
    )


def format_heatmap_data(accuracy_matrix: np.ndarray) -> Dict:
    """
    Format accuracy matrix for Streamlit heatmap display.
    
    Args:
        accuracy_matrix: 5x5 numpy array
        
    Returns:
        Dict with formatted data for Streamlit visualization
    """
    return {
        "matrix": accuracy_matrix,
        "task_labels": [f"Task {i+1}" for i in range(5)],
        "semester_labels": [f"Sem {i+1}" for i in range(5)],
        "vmin": 0.0,
        "vmax": 100.0,
    }


def compare_methods(
    checkpoint_path: str,
    results_dir: Optional[str] = None,
) -> Dict[str, MemoryMetrics]:
    """
    Load and compare metrics across different continual learning methods.
    
    Args:
        checkpoint_path: Path to primary .pt checkpoint
        results_dir: Directory containing baseline_acc_matrix.npy, replay_acc_matrix.npy, etc.
        
    Returns:
        Dict mapping method names to MemoryMetrics
    """
    results = {}
    
    # Load from checkpoint (v4_ultimate method)
    try:
        matrix = load_accuracy_matrix_from_checkpoint(checkpoint_path)
        results["v4_ultimate"] = compute_memory_metrics(matrix)
    except Exception as e:
        print(f"Warning: Could not load checkpoint metrics: {e}")
    
    # Load from results directory if provided
    if results_dir and os.path.isdir(results_dir):
        method_files = {
            "baseline": "baseline_acc_matrix.npy",
            "replay": "replay_acc_matrix.npy",
            "ewc": "ewc_acc_matrix.npy",
            "hybrid": "hybrid_acc_matrix.npy",
        }
        
        for method_name, filename in method_files.items():
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                try:
                    matrix = load_accuracy_matrix_from_npy(filepath)
                    results[method_name] = compute_memory_metrics(matrix)
                except Exception as e:
                    print(f"Warning: Could not load {method_name} metrics: {e}")
    
    return results


def get_colorscale_value(accuracy: float) -> Tuple[int, int, int]:
    """
    Get RGB color for accuracy value (0-100).
    Green: >70%, Yellow: 40-70%, Red/Gray: <40%
    
    Args:
        accuracy: Accuracy value 0-100
        
    Returns:
        RGB tuple (r, g, b)
    """
    if accuracy > 70:  # Green
        return (76, 175, 80)
    elif accuracy >= 40:  # Yellow
        value = int(255 * ((accuracy - 40) / 30))
        return (255, 200, 0)
    else:  # Red/Gray background
        return (200, 200, 200)
