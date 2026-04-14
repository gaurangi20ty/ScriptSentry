#!/usr/bin/env python3
"""Quick test of Section 4 evaluation utilities."""

from pathlib import Path
from evaluation import (
    load_accuracy_matrix_from_checkpoint,
    compute_memory_metrics,
    compare_methods,
)

PROJECT_DIR = Path(__file__).resolve().parent

# Test loading from checkpoint
print("=" * 60)
print("SECTION 4 EVALUATION TEST")
print("=" * 60)

try:
    print("\n1. Loading accuracy matrix from checkpoint...")
    matrix = load_accuracy_matrix_from_checkpoint(
        str(PROJECT_DIR / "checkpoints" / "v4_final.pt")
    )
    print(f"✓ Matrix shape: {matrix.shape}")
    print(f"Matrix:\n{matrix}\n")
    
    print("2. Computing memory metrics...")
    metrics = compute_memory_metrics(matrix)
    print(f"✓ Knowledge Retention: {metrics.knowledge_retention:.1f}%")
    print(f"✓ Average Forgetting: {metrics.average_forgetting:.1f}%")
    print(f"✓ Most Forgotten Task: Task {metrics.most_forgotten_task + 1} ({metrics.per_task_forgetting[metrics.most_forgotten_task]:.1f}% drop)")
    print(f"✓ Most Retained Task: Task {metrics.most_retained_task + 1} ({metrics.per_task_forgetting[metrics.most_retained_task]:.1f}% drop)")
    print(f"✓ Per-task peak accuracies: {metrics.per_task_peak_accuracy}")
    print(f"✓ Per-task final accuracies: {metrics.per_task_final_accuracy}")
    
    print("\n3. Comparing methods...")
    results_dir = PROJECT_DIR / "results"
    if results_dir.exists():
        comparison = compare_methods(
            str(PROJECT_DIR / "checkpoints" / "v4_final.pt"),
            results_dir=str(results_dir),
        )
        print(f"✓ Found {len(comparison)} method(s):")
        for method_name, mtrx in comparison.items():
            print(f"  - {method_name}: {mtrx.knowledge_retention:.1f}% retention")
    else:
        print(f"  Results directory not found: {results_dir}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
