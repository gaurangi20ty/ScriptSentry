"""
ScriptSentry — Phase 1: Baseline Training
===========================================
Sequential training across 5 semesters with NO continual-learning
protection.  This script demonstrates **catastrophic forgetting**:
the model's accuracy on earlier semesters collapses as new ones
are learned.

Outputs
-------
• results/phase1_forgetting_curve.png   – Semester-1 accuracy drop
• results/phase1_accuracy_heatmap.png   – Full 5×5 accuracy matrix
• results/phase1_all_semesters.png      – Every semester over time
• results/baseline_acc_matrix.npy       – Raw numpy matrix
• checkpoints/baseline_after_semN.pt    – Model after each semester

Usage:
    python train.py                     # default (5 epochs / semester)
    python train.py --epochs 10         # more epochs
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_all_semesters
from models import WriterClassifier, ReplayBuffer, EWC


# ================================================================
# Config
# ================================================================
DEVICE = torch.device("cpu")
NUM_SEMESTERS = 5
WRITERS_PER_SEM = 30
TOTAL_WRITERS = NUM_SEMESTERS * WRITERS_PER_SEM  # 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS_DEFAULT = 15

PROJECT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PROJECT, "results")
CKPTS = os.path.join(PROJECT, "checkpoints")


def ensure_dirs():
    """Create output directories."""
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CKPTS, exist_ok=True)


# ================================================================
# Training helpers
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, active_classes=None):
    """
    Train for one epoch.

    Args:
        active_classes: list of int, if provided only these class logits
                        are used for loss computation (masked softmax).

    Returns:
        avg_loss  : float
        accuracy  : float  (0-100)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(images)

        # Mask logits: set inactive classes to -inf so they don't
        # affect softmax / cross-entropy. Focuses gradient on active classes.
        if active_classes is not None:
            mask = torch.full_like(out, float('-inf'))
            mask[:, active_classes] = out[:, active_classes]
            out_masked = mask
        else:
            out_masked = out

        loss = criterion(out_masked, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += out_masked.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, seen_classes=None):
    """
    Evaluate accuracy on a DataLoader.

    Args:
        seen_classes: list of int, if provided only consider these
                      class logits when finding argmax (fair eval).

    Returns:
        accuracy : float  (0-100)
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out = model(images)

        if seen_classes is not None:
            mask = torch.full_like(out, float('-inf'))
            mask[:, seen_classes] = out[:, seen_classes]
            preds = mask.argmax(1)
        else:
            preds = out.argmax(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total if total else 0.0


# ================================================================
# Plotting
# ================================================================
def plot_forgetting_curve(acc_matrix):
    """
    Semester-1 accuracy dropping over time = catastrophic forgetting.
    Saved to results/phase1_forgetting_curve.png
    """
    n = acc_matrix.shape[0]
    xs = list(range(1, n + 1))
    ys = acc_matrix[:, 0]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, "ro-", linewidth=2.5, markersize=10,
             label="Semester 1 Students")

    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=11, fontweight="bold")

    plt.xlabel("After Training on Semester …", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Catastrophic Forgetting — Semester 1 Students", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS, "phase1_forgetting_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_accuracy_heatmap(acc_matrix):
    """
    5×5 heatmap of the accuracy matrix.
    Saved to results/phase1_accuracy_heatmap.png
    """
    n = acc_matrix.shape[0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        acc_matrix, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=[f"Sem {i+1}" for i in range(n)],
        yticklabels=[f"After Sem {i+1}" for i in range(n)],
        vmin=0, vmax=100, linewidths=0.5,
        annot_kws={"size": 12, "fontweight": "bold"},
    )
    plt.xlabel("Evaluated on Semester", fontsize=13)
    plt.ylabel("Trained Through Semester", fontsize=13)
    plt.title("Accuracy Matrix — Baseline (No Continual Learning)", fontsize=14)
    plt.tight_layout()

    path = os.path.join(RESULTS, "phase1_accuracy_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_all_semesters(acc_matrix):
    """
    Accuracy of every semester tracked from the moment it was first learned.
    Saved to results/phase1_all_semesters.png
    """
    n = acc_matrix.shape[0]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]

    plt.figure(figsize=(12, 7))
    for sem in range(n):
        xs = list(range(sem + 1, n + 1))
        ys = acc_matrix[sem:, sem]
        plt.plot(xs, ys, "o-", color=colors[sem % len(colors)],
                 linewidth=2, markersize=8,
                 label=f"Semester {sem+1}")

    plt.xlabel("After Training on Semester …", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.title("All Semesters — Baseline", fontsize=14)
    plt.xticks(range(1, n + 1))
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    path = os.path.join(RESULTS, "phase1_all_semesters.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ================================================================
# Phase 1 — Baseline (catastrophic forgetting)
# ================================================================
def train_baseline(epochs_per_sem=5):
    """
    Train sequentially on 5 semesters.  No replay, no EWC.

    Returns:
        acc_matrix : np.ndarray  (5, 5)
        model      : WriterClassifier
    """
    ensure_dirs()

    banner = (
        "\n" + "=" * 60 + "\n"
        "  ScriptSentry — Phase 1: Baseline Training\n"
        "  (Demonstrating Catastrophic Forgetting)\n"
        + "=" * 60
    )
    print(banner)

    # ---- data ----
    semesters = load_all_semesters(NUM_SEMESTERS, WRITERS_PER_SEM, BATCH_SIZE)

    # ---- model ----
    print(f"\nInitializing WriterClassifier ({TOTAL_WRITERS} classes) …")
    model = WriterClassifier(num_writers=TOTAL_WRITERS).to(DEVICE)
    print(f"  Trainable params : {model.count_trainable():,}")
    print(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss()

    # ---- bookkeeping ----
    acc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    sem_times = []
    total_t0 = time.time()
    seen_classes = []   # cumulative list of class indices seen so far

    # ---- sequential training ----
    for si in range(NUM_SEMESTERS):
        sn = si + 1
        train_loader = semesters[si][0]
        test_loader = semesters[si][1]
        sem_sids = semesters[si][2]
        label_map = semesters[si][3]

        # Active classes for THIS semester (the 30 writer labels)
        active_classes = sorted([label_map[sid] for sid in sem_sids])
        # Add to cumulative seen classes
        for c in active_classes:
            if c not in seen_classes:
                seen_classes.append(c)
        seen_classes_sorted = sorted(seen_classes)

        print(f"\n{'─' * 55}")
        print(f"  SEMESTER {sn}  ({len(sem_sids)} students)")
        print(f"  Active classes: {len(active_classes)} | "
              f"Total seen: {len(seen_classes_sorted)}")
        print(f"{'─' * 55}")

        # Fresh optimizer each semester — prevents stale Adam momentum
        # from sabotaging learning on new writers
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_sem, eta_min=1e-5
        )

        t0 = time.time()
        for ep in range(1, epochs_per_sem + 1):
            loss, tacc = train_one_epoch(model, train_loader,
                                         optimizer, criterion,
                                         active_classes=active_classes)
            scheduler.step()
            print(f"  Epoch {ep}/{epochs_per_sem}  "
                  f"Loss: {loss:.4f}  Train Acc: {tacc:.1f}%")

        elapsed = time.time() - t0
        sem_times.append(elapsed)
        print(f"  ⏱  {elapsed:.1f}s")

        # ---- evaluate on every semester seen so far ----
        # Use seen_classes so argmax only considers writers the model
        # has been trained on (fair comparison)
        print(f"\n  Evaluation after Semester {sn}:")
        for ej in range(NUM_SEMESTERS):
            acc = evaluate(model, semesters[ej][1],
                           seen_classes=seen_classes_sorted)
            acc_matrix[si][ej] = acc

            tag = ""
            if ej == si:
                tag = " ← just learned"
            elif ej < si:
                drop = acc_matrix[ej][ej] - acc
                if drop > 5:
                    tag = f"  ▼ dropped {drop:.1f}%"
            else:
                tag = "  (not yet learned)"
            print(f"    Sem {ej+1}: {acc:5.1f}%{tag}")

        # ---- checkpoint ----
        ckpt = os.path.join(CKPTS, f"baseline_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "acc_matrix": acc_matrix.copy(),
        }, ckpt)

    total_time = time.time() - total_t0

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1 RESULTS — CATASTROPHIC FORGETTING")
    print("=" * 60)

    # ---- pretty accuracy matrix ----
    hdr = "               " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    print(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{acc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        print(f"  After Sem {i+1} :  {row}")

    # ---- key stats ----
    sem1_init = acc_matrix[0][0]
    sem1_final = acc_matrix[-1][0]
    forgetting = sem1_init - sem1_final
    avg_final = np.mean(acc_matrix[-1])

    print(f"\n  Semester 1 — after its own training  : {sem1_init:.1f}%")
    print(f"  Semester 1 — after ALL 5 semesters   : {sem1_final:.1f}%")
    print(f"  Forgetting (Sem 1)                   : {forgetting:.1f}%")
    print(f"  Average final accuracy               : {avg_final:.1f}%")
    print(f"  Total training time                  : {total_time:.1f}s")

    if forgetting > 10:
        print(f"\n  >>> CATASTROPHIC FORGETTING DETECTED <<<")
        print(f"  >>> Semester 1 lost {forgetting:.1f}% accuracy — "
              f"the model forgot those students! <<<")

    # ---- plots ----
    print("\nGenerating plots …")
    plot_forgetting_curve(acc_matrix)
    plot_accuracy_heatmap(acc_matrix)
    plot_all_semesters(acc_matrix)

    npy_path = os.path.join(RESULTS, "baseline_acc_matrix.npy")
    np.save(npy_path, acc_matrix)
    print(f"  Saved → {npy_path}")

    print(f"\n{'=' * 60}")
    print(f"  Phase 1 complete!  Check the results/ folder for plots.")
    print(f"{'=' * 60}\n")

    return acc_matrix, model


# ================================================================
# Phase 2 — Experience Replay
# ================================================================
def plot_replay_heatmap(acc_matrix):
    """5×5 heatmap for the replay accuracy matrix."""
    n = acc_matrix.shape[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        acc_matrix, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=[f"Sem {i+1}" for i in range(n)],
        yticklabels=[f"After Sem {i+1}" for i in range(n)],
        vmin=0, vmax=100, linewidths=0.5,
        annot_kws={"size": 12, "fontweight": "bold"},
    )
    plt.xlabel("Evaluated on Semester", fontsize=13)
    plt.ylabel("Trained Through Semester", fontsize=13)
    plt.title("Accuracy Matrix — Experience Replay", fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase2_accuracy_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_replay_forgetting(acc_matrix):
    """Semester-1 accuracy over time for replay method."""
    n = acc_matrix.shape[0]
    xs = list(range(1, n + 1))
    ys = acc_matrix[:, 0]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, "bo-", linewidth=2.5, markersize=10,
             label="Semester 1 (Replay)")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=11, fontweight="bold")
    plt.xlabel("After Training on Semester …", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Semester 1 Retention — Experience Replay", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase2_forgetting_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_replay_all_semesters(acc_matrix):
    """All semesters over time for replay method."""
    n = acc_matrix.shape[0]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
    plt.figure(figsize=(12, 7))
    for sem in range(n):
        xs = list(range(sem + 1, n + 1))
        ys = acc_matrix[sem:, sem]
        plt.plot(xs, ys, "o-", color=colors[sem % len(colors)],
                 linewidth=2, markersize=8,
                 label=f"Semester {sem+1}")
    plt.xlabel("After Training on Semester …", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.title("All Semesters — Experience Replay", fontsize=14)
    plt.xticks(range(1, n + 1))
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase2_all_semesters.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_comparison_forgetting(baseline_matrix, replay_matrix):
    """
    Side-by-side comparison: Semester 1 accuracy over time.
    Baseline vs Replay.
    """
    n = baseline_matrix.shape[0]
    xs = list(range(1, n + 1))

    plt.figure(figsize=(12, 7))

    # Baseline
    ys_base = baseline_matrix[:, 0]
    plt.plot(xs, ys_base, "ro--", linewidth=2.5, markersize=10,
             label="Baseline (No Protection)", alpha=0.8)
    for x, y in zip(xs, ys_base):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(-15, -15),
                     ha="center", fontsize=10, color="red")

    # Replay
    ys_replay = replay_matrix[:, 0]
    plt.plot(xs, ys_replay, "bs-", linewidth=2.5, markersize=10,
             label="Experience Replay", alpha=0.8)
    for x, y in zip(xs, ys_replay):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(15, 10),
                     ha="center", fontsize=10, color="blue")

    plt.xlabel("After Training on Semester …", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Catastrophic Forgetting: Baseline vs Experience Replay", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS, "comparison_baseline_vs_replay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_comparison_avg_accuracy(baseline_matrix, replay_matrix):
    """
    Bar chart: average final accuracy — Baseline vs Replay.
    """
    base_avg = np.mean(baseline_matrix[-1])
    replay_avg = np.mean(replay_matrix[-1])

    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Baseline", "Experience Replay"],
                   [base_avg, replay_avg],
                   color=["#e74c3c", "#3498db"], width=0.5, edgecolor="black")
    for bar, val in zip(bars, [base_avg, replay_avg]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", fontsize=14, fontweight="bold")

    plt.ylabel("Average Final Accuracy (%)", fontsize=13)
    plt.title("Average Accuracy After All 5 Semesters", fontsize=14)
    plt.ylim(0, max(base_avg, replay_avg) + 15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS, "comparison_avg_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def train_replay(epochs_per_sem=10, buffer_size=500, samples_per_class=5):
    """
    Phase 2: Train with Experience Replay.

    After each semester, store representative samples in a replay buffer.
    When training on a new semester, mix old buffered samples with new data
    so the model reviews previous students while learning new ones.

    Args:
        epochs_per_sem    : int  epochs per semester
        buffer_size       : int  max replay buffer capacity
        samples_per_class : int  samples to store per writer

    Returns:
        acc_matrix : np.ndarray  (5, 5)
        model      : WriterClassifier
    """
    ensure_dirs()

    banner = (
        "\n" + "=" * 60 + "\n"
        "  ScriptSentry — Phase 2: Experience Replay\n"
        f"  Buffer size={buffer_size}, samples/class={samples_per_class}\n"
        + "=" * 60
    )
    print(banner)

    # ---- data ----
    semesters = load_all_semesters(NUM_SEMESTERS, WRITERS_PER_SEM, BATCH_SIZE)

    # ---- model ----
    print(f"\nInitializing WriterClassifier ({TOTAL_WRITERS} classes) …")
    model = WriterClassifier(num_writers=TOTAL_WRITERS).to(DEVICE)
    print(f"  Trainable params : {model.count_trainable():,}")
    print(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(max_size=buffer_size)

    # ---- bookkeeping ----
    acc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    sem_times = []
    total_t0 = time.time()
    seen_classes = []

    # ---- sequential training with replay ----
    for si in range(NUM_SEMESTERS):
        sn = si + 1
        train_loader = semesters[si][0]
        sem_sids = semesters[si][2]
        label_map = semesters[si][3]

        # Active classes
        active_classes = sorted([label_map[sid] for sid in sem_sids])
        for c in active_classes:
            if c not in seen_classes:
                seen_classes.append(c)
        seen_classes_sorted = sorted(seen_classes)

        # Combine new data with replay buffer
        if si == 0:
            combined_loader = train_loader
        else:
            combined_loader = replay_buffer.get_combined_loader(
                train_loader, batch_size=BATCH_SIZE
            )

        buf_stats = replay_buffer.get_stats()
        print(f"\n{'─' * 55}")
        print(f"  SEMESTER {sn}  ({len(sem_sids)} students)")
        print(f"  Active classes: {len(active_classes)} | "
              f"Total seen: {len(seen_classes_sorted)}")
        print(f"  Replay buffer: {buf_stats['size']} samples, "
              f"{buf_stats.get('classes', 0)} classes")
        if si > 0:
            print(f"  Combined training set: "
                  f"{len(combined_loader.dataset)} samples")
        print(f"{'─' * 55}")

        # Fresh optimizer
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_sem, eta_min=1e-5
        )

        t0 = time.time()
        for ep in range(1, epochs_per_sem + 1):
            # Train on combined data but mask to all seen classes
            loss, tacc = train_one_epoch(
                model, combined_loader, optimizer, criterion,
                active_classes=seen_classes_sorted
            )
            scheduler.step()
            print(f"  Epoch {ep}/{epochs_per_sem}  "
                  f"Loss: {loss:.4f}  Train Acc: {tacc:.1f}%")

        elapsed = time.time() - t0
        sem_times.append(elapsed)
        print(f"  ⏱  {elapsed:.1f}s")

        # ---- Add this semester's samples to replay buffer ----
        replay_buffer.add_task_samples(train_loader,
                                       n_per_class=samples_per_class)
        buf_stats = replay_buffer.get_stats()
        print(f"  Buffer updated: {buf_stats['size']} samples, "
              f"{buf_stats.get('classes', 0)} classes")

        # ---- evaluate on every semester ----
        print(f"\n  Evaluation after Semester {sn}:")
        for ej in range(NUM_SEMESTERS):
            acc = evaluate(model, semesters[ej][1],
                           seen_classes=seen_classes_sorted)
            acc_matrix[si][ej] = acc

            tag = ""
            if ej == si:
                tag = " ← just learned"
            elif ej < si:
                drop = acc_matrix[ej][ej] - acc
                if drop > 5:
                    tag = f"  ▼ dropped {drop:.1f}%"
                elif drop < -2:
                    tag = f"  ▲ improved {-drop:.1f}%"
                else:
                    tag = "  ≈ retained"
            else:
                tag = "  (not yet learned)"
            print(f"    Sem {ej+1}: {acc:5.1f}%{tag}")

        # ---- checkpoint ----
        ckpt = os.path.join(CKPTS, f"replay_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "acc_matrix": acc_matrix.copy(),
            "method": "replay",
            "buffer_size": buffer_size,
        }, ckpt)

    total_time = time.time() - total_t0

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2 RESULTS — EXPERIENCE REPLAY")
    print("=" * 60)

    hdr = "               " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    print(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{acc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        print(f"  After Sem {i+1} :  {row}")

    sem1_init = acc_matrix[0][0]
    sem1_final = acc_matrix[-1][0]
    forgetting = sem1_init - sem1_final
    avg_final = np.mean(acc_matrix[-1])

    print(f"\n  Semester 1 — after its own training  : {sem1_init:.1f}%")
    print(f"  Semester 1 — after ALL 5 semesters   : {sem1_final:.1f}%")
    print(f"  Forgetting (Sem 1)                   : {forgetting:.1f}%")
    print(f"  Average final accuracy               : {avg_final:.1f}%")
    print(f"  Buffer size                          : {len(replay_buffer)}")
    print(f"  Total training time                  : {total_time:.1f}s")

    if forgetting < 10:
        print(f"\n  >>> REPLAY IS WORKING! <<<")
        print(f"  >>> Semester 1 only lost {forgetting:.1f}% — "
              f"much less forgetting! <<<")

    # ---- Phase 2 plots ----
    print("\nGenerating Phase 2 plots …")
    plot_replay_heatmap(acc_matrix)
    plot_replay_forgetting(acc_matrix)
    plot_replay_all_semesters(acc_matrix)

    npy_path = os.path.join(RESULTS, "replay_acc_matrix.npy")
    np.save(npy_path, acc_matrix)
    print(f"  Saved → {npy_path}")

    # ---- Comparison plots (if baseline results exist) ----
    baseline_path = os.path.join(RESULTS, "baseline_acc_matrix.npy")
    if os.path.isfile(baseline_path):
        print("\nGenerating comparison plots …")
        baseline_matrix = np.load(baseline_path)
        plot_comparison_forgetting(baseline_matrix, acc_matrix)
        plot_comparison_avg_accuracy(baseline_matrix, acc_matrix)
    else:
        print("\n  ℹ Run Phase 1 first to generate comparison plots.")

    print(f"\n{'=' * 60}")
    print(f"  Phase 2 complete!  Check the results/ folder for plots.")
    print(f"{'=' * 60}\n")

    return acc_matrix, model


# ================================================================
# Phase 3 — Elastic Weight Consolidation (EWC)
# ================================================================
def plot_ewc_heatmap(acc_matrix, lambda_val):
    """5x5 heatmap for the EWC accuracy matrix."""
    n = acc_matrix.shape[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        acc_matrix, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=[f"Sem {i+1}" for i in range(n)],
        yticklabels=[f"After Sem {i+1}" for i in range(n)],
        vmin=0, vmax=100, linewidths=0.5,
        annot_kws={"size": 12, "fontweight": "bold"},
    )
    plt.xlabel("Evaluated on Semester", fontsize=13)
    plt.ylabel("Trained Through Semester", fontsize=13)
    plt.title(f"Accuracy Matrix \u2014 EWC (\u03bb={lambda_val})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase3_accuracy_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_ewc_forgetting(acc_matrix):
    """Semester-1 accuracy over time for EWC method."""
    n = acc_matrix.shape[0]
    xs = list(range(1, n + 1))
    ys = acc_matrix[:, 0]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, "g^-", linewidth=2.5, markersize=10,
             label="Semester 1 (EWC)")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=11, fontweight="bold")
    plt.xlabel("After Training on Semester \u2026", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Semester 1 Retention \u2014 EWC", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase3_forgetting_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_ewc_all_semesters(acc_matrix):
    """All semesters over time for EWC method."""
    n = acc_matrix.shape[0]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
    plt.figure(figsize=(12, 7))
    for sem in range(n):
        xs = list(range(sem + 1, n + 1))
        ys = acc_matrix[sem:, sem]
        plt.plot(xs, ys, "o-", color=colors[sem % len(colors)],
                 linewidth=2, markersize=8,
                 label=f"Semester {sem+1}")
    plt.xlabel("After Training on Semester \u2026", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.title("All Semesters \u2014 EWC", fontsize=14)
    plt.xticks(range(1, n + 1))
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase3_all_semesters.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_three_way_forgetting(baseline_matrix, replay_matrix, ewc_matrix):
    """
    Three-way comparison: Semester 1 accuracy over time.
    Baseline vs Replay vs EWC.
    """
    n = baseline_matrix.shape[0]
    xs = list(range(1, n + 1))

    plt.figure(figsize=(13, 7))

    # Baseline
    ys = baseline_matrix[:, 0]
    plt.plot(xs, ys, "ro--", linewidth=2.5, markersize=10,
             label="Baseline (No Protection)", alpha=0.8)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(-20, -15),
                     ha="center", fontsize=9, color="red")

    # Replay
    ys = replay_matrix[:, 0]
    plt.plot(xs, ys, "bs-", linewidth=2.5, markersize=10,
             label="Experience Replay", alpha=0.8)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(20, 10),
                     ha="center", fontsize=9, color="blue")

    # EWC
    ys = ewc_matrix[:, 0]
    plt.plot(xs, ys, "g^-", linewidth=2.5, markersize=10,
             label="EWC", alpha=0.8)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 15),
                     ha="center", fontsize=9, color="green")

    plt.xlabel("After Training on Semester \u2026", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Catastrophic Forgetting: Baseline vs Replay vs EWC", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS, "comparison_three_way_forgetting.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_three_way_avg_accuracy(baseline_matrix, replay_matrix, ewc_matrix):
    """
    Bar chart: average final accuracy \u2014 Baseline vs Replay vs EWC.
    """
    base_avg = np.mean(baseline_matrix[-1])
    replay_avg = np.mean(replay_matrix[-1])
    ewc_avg = np.mean(ewc_matrix[-1])

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        ["Baseline", "Replay", "EWC"],
        [base_avg, replay_avg, ewc_avg],
        color=["#e74c3c", "#3498db", "#27ae60"],
        width=0.5, edgecolor="black",
    )
    for bar, val in zip(bars, [base_avg, replay_avg, ewc_avg]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", fontsize=14, fontweight="bold")

    plt.ylabel("Average Final Accuracy (%)", fontsize=13)
    plt.title("Average Accuracy After All 5 Semesters", fontsize=14)
    plt.ylim(0, max(base_avg, replay_avg, ewc_avg) + 15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS, "comparison_three_way_avg.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_three_way_heatmaps(baseline_matrix, replay_matrix, ewc_matrix):
    """Side-by-side-by-side heatmaps for all three methods."""
    n = baseline_matrix.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    titles = ["Baseline (No Protection)", "Experience Replay", "EWC"]
    matrices = [baseline_matrix, replay_matrix, ewc_matrix]

    for ax, mat, title in zip(axes, matrices, titles):
        sns.heatmap(
            mat, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=[f"Sem {i+1}" for i in range(n)],
            yticklabels=[f"After Sem {i+1}" for i in range(n)],
            vmin=0, vmax=100, linewidths=0.5,
            annot_kws={"size": 10, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Evaluated on Semester", fontsize=11)
        ax.set_ylabel("Trained Through Semester", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.suptitle("Accuracy Matrices \u2014 All Methods Compared", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(RESULTS, "comparison_three_way_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved \u2192 {path}")


def train_ewc(epochs_per_sem=10, ewc_lambda=1000):
    """
    Phase 3: Train with Elastic Weight Consolidation.

    After each semester, the Fisher Information Matrix is computed to
    estimate parameter importance.  During subsequent training, an EWC
    penalty discourages large changes to important parameters.

    Args:
        epochs_per_sem : int   epochs per semester
        ewc_lambda     : float EWC penalty weight

    Returns:
        acc_matrix : np.ndarray  (5, 5)
        model      : WriterClassifier
    """
    ensure_dirs()

    banner = (
        "\n" + "=" * 60 + "\n"
        "  ScriptSentry \u2014 Phase 3: Elastic Weight Consolidation\n"
        f"  \u03bb = {ewc_lambda}\n"
        + "=" * 60
    )
    print(banner)

    # ---- data ----
    semesters = load_all_semesters(NUM_SEMESTERS, WRITERS_PER_SEM, BATCH_SIZE)

    # ---- model ----
    print(f"\nInitializing WriterClassifier ({TOTAL_WRITERS} classes) \u2026")
    model = WriterClassifier(num_writers=TOTAL_WRITERS).to(DEVICE)
    print(f"  Trainable params : {model.count_trainable():,}")
    print(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    ewc = EWC(model, lambda_=ewc_lambda)

    # ---- bookkeeping ----
    acc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    sem_times = []
    ewc_losses_log = []   # track EWC penalty magnitude per epoch
    total_t0 = time.time()
    seen_classes = []

    # ---- sequential training with EWC ----
    for si in range(NUM_SEMESTERS):
        sn = si + 1
        train_loader = semesters[si][0]
        sem_sids = semesters[si][2]
        label_map = semesters[si][3]

        # Active classes
        active_classes = sorted([label_map[sid] for sid in sem_sids])
        for c in active_classes:
            if c not in seen_classes:
                seen_classes.append(c)
        seen_classes_sorted = sorted(seen_classes)

        sep = '\u2500' * 55
        print(f"\n{sep}")
        print(f"  SEMESTER {sn}  ({len(sem_sids)} students)")
        print(f"  Active classes: {len(active_classes)} | "
              f"Total seen: {len(seen_classes_sorted)}")
        print(f"  EWC consolidated tasks: {ewc.num_tasks()}")
        print(sep)

        # Fresh optimizer
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_sem, eta_min=1e-5
        )

        t0 = time.time()
        for ep in range(1, epochs_per_sem + 1):
            # Custom training loop with EWC penalty
            model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_ewc_pen = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(images)

                # Mask logits to active classes
                if active_classes is not None:
                    mask_t = torch.full_like(out, float('-inf'))
                    mask_t[:, active_classes] = out[:, active_classes]
                    out_masked = mask_t
                else:
                    out_masked = out

                ce_loss = criterion(out_masked, labels)
                ewc_pen = ewc.penalty(model)
                loss = ce_loss + ewc_pen

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_ce += ce_loss.item() * labels.size(0)
                total_ewc_pen += ewc_pen.item() * labels.size(0)
                correct += out_masked.argmax(1).eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / total
            avg_ce = total_ce / total
            avg_ewc = total_ewc_pen / total
            acc = 100.0 * correct / total
            ewc_losses_log.append(avg_ewc)

            print(f"  Epoch {ep}/{epochs_per_sem}  "
                  f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f} + "
                  f"EWC: {avg_ewc:.4f})  Acc: {acc:.1f}%")
            scheduler.step()

        elapsed = time.time() - t0
        sem_times.append(elapsed)
        print(f"  \u23f1  {elapsed:.1f}s")

        # ---- Consolidate: compute Fisher & store params ----
        print(f"  Computing Fisher Information Matrix \u2026")
        fisher_t0 = time.time()
        ewc.consolidate(model, train_loader, active_classes=active_classes)
        print(f"  Fisher computed in {time.time() - fisher_t0:.1f}s  "
              f"(tasks stored: {ewc.num_tasks()})")

        # ---- evaluate on every semester ----
        print(f"\n  Evaluation after Semester {sn}:")
        for ej in range(NUM_SEMESTERS):
            acc = evaluate(model, semesters[ej][1],
                           seen_classes=seen_classes_sorted)
            acc_matrix[si][ej] = acc

            tag = ""
            if ej == si:
                tag = " \u2190 just learned"
            elif ej < si:
                drop = acc_matrix[ej][ej] - acc
                if drop > 5:
                    tag = f"  \u25bc dropped {drop:.1f}%"
                elif drop < -2:
                    tag = f"  \u25b2 improved {-drop:.1f}%"
                else:
                    tag = "  \u2248 retained"
            else:
                tag = "  (not yet learned)"
            print(f"    Sem {ej+1}: {acc:5.1f}%{tag}")

        # ---- checkpoint ----
        ckpt = os.path.join(CKPTS, f"ewc_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "acc_matrix": acc_matrix.copy(),
            "method": "ewc",
            "ewc_lambda": ewc_lambda,
        }, ckpt)

    total_time = time.time() - total_t0

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print(f"  PHASE 3 RESULTS \u2014 EWC (\u03bb={ewc_lambda})")
    print("=" * 60)

    hdr = "               " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    print(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{acc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        print(f"  After Sem {i+1} :  {row}")

    sem1_init = acc_matrix[0][0]
    sem1_final = acc_matrix[-1][0]
    forgetting = sem1_init - sem1_final
    avg_final = np.mean(acc_matrix[-1])

    print(f"\n  Semester 1 \u2014 after its own training  : {sem1_init:.1f}%")
    print(f"  Semester 1 \u2014 after ALL 5 semesters   : {sem1_final:.1f}%")
    print(f"  Forgetting (Sem 1)                   : {forgetting:.1f}%")
    print(f"  Average final accuracy               : {avg_final:.1f}%")
    print(f"  EWC \u03bb                                : {ewc_lambda}")
    print(f"  Total training time                  : {total_time:.1f}s")

    # ---- Phase 3 plots ----
    print("\nGenerating Phase 3 plots \u2026")
    plot_ewc_heatmap(acc_matrix, ewc_lambda)
    plot_ewc_forgetting(acc_matrix)
    plot_ewc_all_semesters(acc_matrix)

    npy_path = os.path.join(RESULTS, "ewc_acc_matrix.npy")
    np.save(npy_path, acc_matrix)
    print(f"  Saved \u2192 {npy_path}")

    # ---- Three-way comparison plots (if both baseline & replay exist) ----
    baseline_path = os.path.join(RESULTS, "baseline_acc_matrix.npy")
    replay_path = os.path.join(RESULTS, "replay_acc_matrix.npy")
    if os.path.isfile(baseline_path) and os.path.isfile(replay_path):
        print("\nGenerating three-way comparison plots \u2026")
        baseline_matrix = np.load(baseline_path)
        replay_matrix = np.load(replay_path)
        plot_three_way_forgetting(baseline_matrix, replay_matrix, acc_matrix)
        plot_three_way_avg_accuracy(baseline_matrix, replay_matrix, acc_matrix)
        plot_three_way_heatmaps(baseline_matrix, replay_matrix, acc_matrix)
    else:
        print("\n  \u2139 Run Phases 1 & 2 first to generate three-way comparison plots.")

    print(f"\n{'=' * 60}")
    print(f"  Phase 3 complete!  Check the results/ folder for plots.")
    print(f"{'=' * 60}\n")

    return acc_matrix, model


# ================================================================
# Phase 3b — Hybrid: Replay + EWC
# ================================================================
def plot_hybrid_heatmap(acc_matrix, ewc_lambda):
    """5x5 heatmap for the hybrid accuracy matrix."""
    n = acc_matrix.shape[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        acc_matrix, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=[f"Sem {i+1}" for i in range(n)],
        yticklabels=[f"After Sem {i+1}" for i in range(n)],
        vmin=0, vmax=100, linewidths=0.5,
        annot_kws={"size": 12, "fontweight": "bold"},
    )
    plt.xlabel("Evaluated on Semester", fontsize=13)
    plt.ylabel("Trained Through Semester", fontsize=13)
    plt.title(f"Accuracy Matrix \u2014 Hybrid Replay+EWC (\u03bb={ewc_lambda})",
              fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase3b_accuracy_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_hybrid_forgetting(acc_matrix):
    """Semester-1 accuracy over time for hybrid method."""
    n = acc_matrix.shape[0]
    xs = list(range(1, n + 1))
    ys = acc_matrix[:, 0]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, "m^-", linewidth=2.5, markersize=10,
             label="Semester 1 (Hybrid)")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=11, fontweight="bold")
    plt.xlabel("After Training on Semester \u2026", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Semester 1 Retention \u2014 Hybrid (Replay + EWC)", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase3b_forgetting_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_hybrid_all_semesters(acc_matrix):
    """All semesters over time for hybrid method."""
    n = acc_matrix.shape[0]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
    plt.figure(figsize=(12, 7))
    for sem in range(n):
        xs = list(range(sem + 1, n + 1))
        ys = acc_matrix[sem:, sem]
        plt.plot(xs, ys, "o-", color=colors[sem % len(colors)],
                 linewidth=2, markersize=8,
                 label=f"Semester {sem+1}")
    plt.xlabel("After Training on Semester \u2026", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.title("All Semesters \u2014 Hybrid (Replay + EWC)", fontsize=14)
    plt.xticks(range(1, n + 1))
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS, "phase3b_all_semesters.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_four_way_forgetting(base, replay, ewc, hybrid):
    """Semester 1 retention: Baseline vs Replay vs EWC vs Hybrid."""
    n = base.shape[0]
    xs = list(range(1, n + 1))
    plt.figure(figsize=(14, 7))

    for mat, label, style, color, offset in [
        (base,   "Baseline",       "ro--", "red",    (-20, -15)),
        (replay, "Replay",         "bs-",  "blue",   (20, 10)),
        (ewc,    "EWC",            "g^-",  "green",  (-20, 12)),
        (hybrid, "Hybrid (R+EWC)", "mD-",  "purple", (20, -12)),
    ]:
        ys = mat[:, 0]
        plt.plot(xs, ys, style, linewidth=2.5, markersize=10,
                 label=label, alpha=0.8)
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.1f}%", (x, y),
                         textcoords="offset points", xytext=offset,
                         ha="center", fontsize=8, color=color)

    plt.xlabel("After Training on Semester \u2026", fontsize=13)
    plt.ylabel("Accuracy on Semester 1 (%)", fontsize=13)
    plt.title("Catastrophic Forgetting: All Methods Compared", fontsize=14)
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_four_way_forgetting.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_four_way_avg(base, replay, ewc, hybrid):
    """Bar chart: average final accuracy \u2014 all four methods."""
    vals = [np.mean(m[-1]) for m in [base, replay, ewc, hybrid]]
    labels = ["Baseline", "Replay", "EWC", "Hybrid\n(R+EWC)"]
    colors = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, vals, color=colors, width=0.5, edgecolor="black")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", fontsize=14, fontweight="bold")

    plt.ylabel("Average Final Accuracy (%)", fontsize=13)
    plt.title("Average Accuracy After All 5 Semesters", fontsize=14)
    plt.ylim(0, max(vals) + 15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_four_way_avg.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved \u2192 {path}")


def plot_four_way_heatmaps(base, replay, ewc, hybrid):
    """2x2 grid of heatmaps for all four methods."""
    n = base.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    titles = ["Baseline", "Replay", "EWC", "Hybrid (Replay + EWC)"]
    matrices = [base, replay, ewc, hybrid]

    for ax, mat, title in zip(axes.flatten(), matrices, titles):
        sns.heatmap(
            mat, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=[f"Sem {i+1}" for i in range(n)],
            yticklabels=[f"After Sem {i+1}" for i in range(n)],
            vmin=0, vmax=100, linewidths=0.5,
            annot_kws={"size": 10, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Evaluated on Semester", fontsize=11)
        ax.set_ylabel("Trained Through Semester", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.suptitle("Accuracy Matrices \u2014 All Methods Compared", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_four_way_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved \u2192 {path}")


def train_hybrid(epochs_per_sem=15, buffer_size=1500, samples_per_class=10,
                 ewc_lambda=1000):
    """
    Phase 3b: Hybrid — Replay + EWC combined.

    Combines Experience Replay (buffered rehearsal of old samples) with
    EWC (Fisher-based regularization penalty) for maximum protection
    against catastrophic forgetting.

    Args:
        epochs_per_sem    : int   epochs per semester
        buffer_size       : int   max replay buffer capacity
        samples_per_class : int   samples to store per writer
        ewc_lambda        : float EWC penalty weight

    Returns:
        acc_matrix : np.ndarray (5, 5)
        model      : WriterClassifier
    """
    ensure_dirs()

    banner = (
        "\n" + "=" * 60 + "\n"
        "  ScriptSentry \u2014 Phase 3b: Hybrid (Replay + EWC)\n"
        f"  Buffer={buffer_size}, samples/class={samples_per_class}, "
        f"\u03bb={ewc_lambda}\n"
        + "=" * 60
    )
    print(banner)

    # ---- data ----
    semesters = load_all_semesters(NUM_SEMESTERS, WRITERS_PER_SEM, BATCH_SIZE)

    # ---- model ----
    print(f"\nInitializing WriterClassifier ({TOTAL_WRITERS} classes) \u2026")
    model = WriterClassifier(num_writers=TOTAL_WRITERS).to(DEVICE)
    print(f"  Trainable params : {model.count_trainable():,}")
    print(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(max_size=buffer_size)
    ewc = EWC(model, lambda_=ewc_lambda)

    # ---- bookkeeping ----
    acc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    sem_times = []
    total_t0 = time.time()
    seen_classes = []

    # ---- sequential training with replay + EWC ----
    for si in range(NUM_SEMESTERS):
        sn = si + 1
        train_loader = semesters[si][0]
        sem_sids = semesters[si][2]
        label_map = semesters[si][3]

        # Active classes
        active_classes = sorted([label_map[sid] for sid in sem_sids])
        for c in active_classes:
            if c not in seen_classes:
                seen_classes.append(c)
        seen_classes_sorted = sorted(seen_classes)

        # Combine new data with replay buffer
        if si == 0:
            combined_loader = train_loader
        else:
            combined_loader = replay_buffer.get_combined_loader(
                train_loader, batch_size=BATCH_SIZE
            )

        buf_stats = replay_buffer.get_stats()
        sep = '\u2500' * 55
        print(f"\n{sep}")
        print(f"  SEMESTER {sn}  ({len(sem_sids)} students)")
        print(f"  Active classes: {len(active_classes)} | "
              f"Total seen: {len(seen_classes_sorted)}")
        print(f"  Replay buffer: {buf_stats['size']} samples, "
              f"{buf_stats.get('classes', 0)} classes")
        print(f"  EWC consolidated tasks: {ewc.num_tasks()}")
        if si > 0:
            print(f"  Combined training set: "
                  f"{len(combined_loader.dataset)} samples")
        print(sep)

        # Fresh optimizer
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_sem, eta_min=1e-5
        )

        t0 = time.time()
        for ep in range(1, epochs_per_sem + 1):
            # Custom training loop: replay data + EWC penalty
            model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_ewc_pen = 0.0
            correct = 0
            total = 0

            for images, labels in combined_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(images)

                # Mask logits to all seen classes
                mask_t = torch.full_like(out, float('-inf'))
                mask_t[:, seen_classes_sorted] = out[:, seen_classes_sorted]
                out_masked = mask_t

                ce_loss = criterion(out_masked, labels)
                ewc_pen = ewc.penalty(model)
                loss = ce_loss + ewc_pen

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_ce += ce_loss.item() * labels.size(0)
                total_ewc_pen += ewc_pen.item() * labels.size(0)
                correct += out_masked.argmax(1).eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / total
            avg_ce = total_ce / total
            avg_ewc = total_ewc_pen / total
            acc = 100.0 * correct / total

            print(f"  Epoch {ep}/{epochs_per_sem}  "
                  f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f} + "
                  f"EWC: {avg_ewc:.4f})  Acc: {acc:.1f}%")
            scheduler.step()

        elapsed = time.time() - t0
        sem_times.append(elapsed)
        print(f"  \u23f1  {elapsed:.1f}s")

        # ---- Add to replay buffer ----
        replay_buffer.add_task_samples(train_loader,
                                       n_per_class=samples_per_class)
        buf_stats = replay_buffer.get_stats()
        print(f"  Buffer updated: {buf_stats['size']} samples, "
              f"{buf_stats.get('classes', 0)} classes")

        # ---- Consolidate EWC: compute Fisher & store params ----
        print(f"  Computing Fisher Information Matrix \u2026")
        fisher_t0 = time.time()
        ewc.consolidate(model, train_loader, active_classes=active_classes)
        print(f"  Fisher computed in {time.time() - fisher_t0:.1f}s  "
              f"(tasks stored: {ewc.num_tasks()})")

        # ---- evaluate on every semester ----
        print(f"\n  Evaluation after Semester {sn}:")
        for ej in range(NUM_SEMESTERS):
            acc = evaluate(model, semesters[ej][1],
                           seen_classes=seen_classes_sorted)
            acc_matrix[si][ej] = acc

            tag = ""
            if ej == si:
                tag = " \u2190 just learned"
            elif ej < si:
                drop = acc_matrix[ej][ej] - acc
                if drop > 5:
                    tag = f"  \u25bc dropped {drop:.1f}%"
                elif drop < -2:
                    tag = f"  \u25b2 improved {-drop:.1f}%"
                else:
                    tag = "  \u2248 retained"
            else:
                tag = "  (not yet learned)"
            print(f"    Sem {ej+1}: {acc:5.1f}%{tag}")

        # ---- checkpoint ----
        ckpt = os.path.join(CKPTS, f"hybrid_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "acc_matrix": acc_matrix.copy(),
            "method": "hybrid",
            "buffer_size": buffer_size,
            "ewc_lambda": ewc_lambda,
        }, ckpt)

    total_time = time.time() - total_t0

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print(f"  PHASE 3b RESULTS \u2014 HYBRID Replay+EWC (\u03bb={ewc_lambda})")
    print("=" * 60)

    hdr = "               " + "  ".join(
        f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    print(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(
            f"{acc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        print(f"  After Sem {i+1} :  {row}")

    sem1_init = acc_matrix[0][0]
    sem1_final = acc_matrix[-1][0]
    forgetting = sem1_init - sem1_final
    avg_final = np.mean(acc_matrix[-1])

    print(f"\n  Semester 1 \u2014 after its own training  : {sem1_init:.1f}%")
    print(f"  Semester 1 \u2014 after ALL 5 semesters   : {sem1_final:.1f}%")
    print(f"  Forgetting (Sem 1)                   : {forgetting:.1f}%")
    print(f"  Average final accuracy               : {avg_final:.1f}%")
    print(f"  Buffer size                          : {len(replay_buffer)}")
    print(f"  EWC \u03bb                                : {ewc_lambda}")
    print(f"  Total training time                  : {total_time:.1f}s")

    # ---- Phase 3b plots ----
    print("\nGenerating Phase 3b plots \u2026")
    plot_hybrid_heatmap(acc_matrix, ewc_lambda)
    plot_hybrid_forgetting(acc_matrix)
    plot_hybrid_all_semesters(acc_matrix)

    npy_path = os.path.join(RESULTS, "hybrid_acc_matrix.npy")
    np.save(npy_path, acc_matrix)
    print(f"  Saved \u2192 {npy_path}")

    # ---- Four-way comparison (if all other results exist) ----
    base_path = os.path.join(RESULTS, "baseline_acc_matrix.npy")
    replay_path = os.path.join(RESULTS, "replay_acc_matrix.npy")
    ewc_path = os.path.join(RESULTS, "ewc_acc_matrix.npy")
    if (os.path.isfile(base_path) and os.path.isfile(replay_path)
            and os.path.isfile(ewc_path)):
        print("\nGenerating four-way comparison plots \u2026")
        base_m = np.load(base_path)
        replay_m = np.load(replay_path)
        ewc_m = np.load(ewc_path)
        plot_four_way_forgetting(base_m, replay_m, ewc_m, acc_matrix)
        plot_four_way_avg(base_m, replay_m, ewc_m, acc_matrix)
        plot_four_way_heatmaps(base_m, replay_m, ewc_m, acc_matrix)
    else:
        print("\n  \u2139 Run Phases 1, 2 & 3 first for four-way comparison.")

    print(f"\n{'=' * 60}")
    print(f"  Phase 3b complete!  Check the results/ folder for plots.")
    print(f"{'=' * 60}\n")

    return acc_matrix, model


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ScriptSentry Training — Phase 1 (Baseline), Phase 2 (Replay), Phase 3 (EWC), Phase 4 (Hybrid)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training phase: 1=Baseline, 2=Replay, 3=EWC, 4=Hybrid (default: 1)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT,
                        help=f"Epochs per semester (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--buffer-size", type=int, default=1500,
                        help="Replay buffer max size (Phase 2/4, default: 1500)")
    parser.add_argument("--samples-per-class", type=int, default=10,
                        help="Samples to store per writer (Phase 2/4, default: 10)")
    parser.add_argument("--ewc-lambda", type=float, default=5000,
                        help="EWC penalty weight (Phase 3, default: 5000)")
    parser.add_argument("--hybrid-lambda", type=float, default=1000,
                        help="EWC penalty weight for hybrid (Phase 4, default: 1000)")
    args = parser.parse_args()

    if args.phase == 1:
        train_baseline(epochs_per_sem=args.epochs)
    elif args.phase == 2:
        train_replay(
            epochs_per_sem=args.epochs,
            buffer_size=args.buffer_size,
            samples_per_class=args.samples_per_class,
        )
    elif args.phase == 3:
        train_ewc(
            epochs_per_sem=args.epochs,
            ewc_lambda=args.ewc_lambda,
        )
    elif args.phase == 4:
        train_hybrid(
            epochs_per_sem=args.epochs,
            buffer_size=args.buffer_size,
            samples_per_class=args.samples_per_class,
            ewc_lambda=args.hybrid_lambda,
        )
