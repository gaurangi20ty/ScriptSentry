"""
ScriptSentry — Image Testing Script
=====================================
Load any saved checkpoint and test handwriting images against it.

Modes:
  1. Single image test    — predict writer for one image
  2. Folder test          — predict writer for all images in a folder
  3. Interactive mode      — keep entering image paths
  4. Test dataset samples  — grab random samples from the dataset and verify

Usage:
    python test_image.py --image path/to/image.png
    python test_image.py --image path/to/image.png --checkpoint baseline_after_sem3.pt
    python test_image.py --folder path/to/folder/
    python test_image.py --interactive
    python test_image.py --dataset-test --semester 2
    python test_image.py --dataset-test --all-semesters
"""

import os
import sys
import argparse
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import models

from models import WriterClassifier
from dataset import (
    get_test_transform, build_student_dataset, load_all_semesters,
    WORDS_DIR, WORDS_TXT
)

# ================================================================
# Config
# ================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DEVICE = torch.device("cpu")

NUM_SEMESTERS = 5
WRITERS_PER_SEM = 30
TOTAL_WRITERS = NUM_SEMESTERS * WRITERS_PER_SEM  # 150


class CosineLinear(nn.Module):
    """Cosine-similarity classification head used by v4 checkpoints."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.sigma = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.sigma * F.linear(x_norm, w_norm)


class WriterClassifierV4(nn.Module):
    """ResNet-50 + projector + cosine classifier used by kaggle_ultimate_v4.py."""
    def __init__(self, num_writers=150):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = CosineLinear(512, num_writers)

    def get_features(self, x):
        raw = self.backbone(x)
        return self.projector(raw)

    def forward(self, x):
        feat = self.get_features(x)
        return self.classifier(feat)


# ================================================================
# Helpers
# ================================================================
def list_checkpoints():
    """List all available checkpoint files."""
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    return ckpts


def load_model(checkpoint_path):
    """
    Load a WriterClassifier from a checkpoint file.

    Returns:
        model      : WriterClassifier (eval mode)
        ckpt_data  : dict with metadata (semester, acc_matrix, etc.)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Auto-detect model family from checkpoint keys.
    is_v4 = (
        "projector.0.weight" in state_dict
        or "classifier.sigma" in state_dict
        or any(k.startswith("backbone.layer1.0.conv3") for k in state_dict)
    )

    if is_v4:
        model = WriterClassifierV4(num_writers=TOTAL_WRITERS).to(DEVICE)
        model_name = "WriterClassifierV4 (ResNet-50 + cosine head)"
    else:
        model = WriterClassifier(num_writers=TOTAL_WRITERS).to(DEVICE)
        model_name = "WriterClassifier (ResNet-18)"

    model.load_state_dict(state_dict)
    model.eval()

    print(f"  ✓ Loaded checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"    Model type: {model_name}")
    print(f"    Trained through semester: {ckpt.get('semester', '?')}")
    return model, ckpt


def get_student_id_map():
    """
    Build the same student ID mapping used during training.

    Returns:
        idx_to_sid : dict {int_label: student_id_string}
        sid_to_idx : dict {student_id_string: int_label}
        semester_groups : list of lists, each containing student IDs for that semester
    """
    student_samples, _ = build_student_dataset()

    total_needed = NUM_SEMESTERS * WRITERS_PER_SEM
    by_count = sorted(student_samples, key=lambda s: len(student_samples[s]),
                      reverse=True)
    selected = sorted(by_count[:total_needed])

    sid_to_idx = {sid: idx for idx, sid in enumerate(selected)}
    idx_to_sid = {idx: sid for sid, idx in sid_to_idx.items()}

    # Group by semester
    semester_groups = []
    for s in range(NUM_SEMESTERS):
        start = s * WRITERS_PER_SEM
        end = start + WRITERS_PER_SEM
        semester_groups.append(selected[start:end])

    return idx_to_sid, sid_to_idx, semester_groups


def predict_image(model, image_path, transform, idx_to_sid, top_k=5, seen_classes=None):
    """
    Run prediction on a single image.

    Args:
        model       : WriterClassifier in eval mode
        image_path  : str path to image file
        transform   : torchvision transform
        idx_to_sid  : dict mapping class index to student ID
        top_k       : number of top predictions to return
        seen_classes: list of class indices to restrict predictions to

    Returns:
        dict with prediction results
    """
    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        # Mask to only seen classes if provided
        if seen_classes is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, seen_classes] = logits[:, seen_classes]
            logits = mask

        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

    results = {
        "image": os.path.basename(image_path),
        "predictions": []
    }

    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        sid = idx_to_sid.get(idx, f"unknown_{idx}")

        # Determine which semester this student belongs to
        sem = (idx // WRITERS_PER_SEM) + 1

        results["predictions"].append({
            "rank": i + 1,
            "student_id": sid,
            "class_idx": idx,
            "confidence": prob * 100,
            "semester": sem,
        })

    return results


def print_prediction(result, verbose=True):
    """Pretty-print prediction results."""
    if "error" in result:
        print(f"  ✗ {result['error']}")
        return

    print(f"\n  Image: {result['image']}")
    print(f"  {'─' * 50}")

    top = result["predictions"][0]
    conf = top["confidence"]

    if conf > 65:
        verdict = "HIGH CONFIDENCE ✅"
    elif conf > 40:
        verdict = "MODERATE CONFIDENCE ⚠️"
    else:
        verdict = "LOW CONFIDENCE ❌"

    print(f"  Predicted Writer : {top['student_id']}  (Semester {top['semester']})")
    print(f"  Confidence       : {conf:.1f}%  — {verdict}")

    if verbose and len(result["predictions"]) > 1:
        print(f"\n  Top-{len(result['predictions'])} Predictions:")
        for p in result["predictions"]:
            bar = "█" * int(p["confidence"] / 2) + "░" * (50 - int(p["confidence"] / 2))
            print(f"    #{p['rank']}  {p['student_id']:>12s}  (Sem {p['semester']})  "
                  f"{p['confidence']:5.1f}%  {bar}")


# ================================================================
# Test Modes
# ================================================================
def test_single_image(args, model, idx_to_sid, seen_classes):
    """Test a single image file."""
    transform = get_test_transform()
    result = predict_image(model, args.image, transform, idx_to_sid,
                           top_k=args.top_k, seen_classes=seen_classes)
    print_prediction(result)


def test_folder(args, model, idx_to_sid, seen_classes):
    """Test all images in a folder."""
    transform = get_test_transform()
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")

    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(args.folder, ext)))

    if not images:
        print(f"  ✗ No images found in {args.folder}")
        return

    print(f"\n  Testing {len(images)} images from: {args.folder}")
    print(f"  {'=' * 55}")

    for img_path in sorted(images):
        result = predict_image(model, img_path, transform, idx_to_sid,
                               top_k=args.top_k, seen_classes=seen_classes)
        print_prediction(result, verbose=False)


def test_interactive(args, model, idx_to_sid, seen_classes):
    """Interactive mode — keep entering image paths."""
    transform = get_test_transform()

    print("\n  Interactive Testing Mode")
    print("  Type an image path and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            path = input("  Image path > ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if path.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break

        if not path:
            continue

        if not os.path.isfile(path):
            print(f"  ✗ File not found: {path}")
            continue

        result = predict_image(model, path, transform, idx_to_sid,
                               top_k=args.top_k, seen_classes=seen_classes)
        print_prediction(result)
        print()


def test_dataset_samples(args, model, idx_to_sid, sid_to_idx, semester_groups, seen_classes):
    """
    Grab random samples from the actual dataset and test them.
    Shows whether the model correctly identifies the writer.
    """
    transform = get_test_transform()
    student_samples, _ = build_student_dataset()

    semesters_to_test = []
    if args.all_semesters:
        semesters_to_test = list(range(1, NUM_SEMESTERS + 1))
    elif args.semester:
        semesters_to_test = [args.semester]
    else:
        semesters_to_test = list(range(1, NUM_SEMESTERS + 1))

    n_samples = args.n_samples

    print(f"\n  Dataset Sample Test — {n_samples} random samples per semester")
    print(f"  {'=' * 60}")

    total_correct = 0
    total_tested = 0

    for sem_num in semesters_to_test:
        if sem_num > NUM_SEMESTERS:
            continue

        sem_sids = semester_groups[sem_num - 1]
        print(f"\n  ── Semester {sem_num} ({len(sem_sids)} students) ──")

        sem_correct = 0
        sem_total = 0

        for sid in sem_sids:
            if sid not in student_samples:
                continue

            paths = student_samples[sid]
            # Pick random samples
            sample_paths = random.sample(paths, min(n_samples, len(paths)))

            for img_path in sample_paths:
                result = predict_image(model, img_path, transform, idx_to_sid,
                                       top_k=1, seen_classes=seen_classes)

                if "error" in result:
                    continue

                predicted_sid = result["predictions"][0]["student_id"]
                confidence = result["predictions"][0]["confidence"]
                correct = predicted_sid == sid

                sem_correct += int(correct)
                sem_total += 1

                status = "✅" if correct else "❌"
                print(f"    {status}  True: {sid:>12s}  Pred: {predicted_sid:>12s}  "
                      f"Conf: {confidence:5.1f}%")

        if sem_total > 0:
            acc = 100.0 * sem_correct / sem_total
            print(f"\n    Semester {sem_num} Accuracy: {sem_correct}/{sem_total} = {acc:.1f}%")
            total_correct += sem_correct
            total_tested += sem_total

    if total_tested > 0:
        overall = 100.0 * total_correct / total_tested
        print(f"\n  {'=' * 60}")
        print(f"  Overall: {total_correct}/{total_tested} = {overall:.1f}%")


# ================================================================
# Main
# ================================================================
def main():
    """Main entry point for the testing script."""
    parser = argparse.ArgumentParser(
        description="ScriptSentry — Test handwriting images against trained checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_image.py --image sample.png
  python test_image.py --image sample.png --checkpoint baseline_after_sem3.pt
  python test_image.py --folder ./test_images/
  python test_image.py --interactive
  python test_image.py --dataset-test --semester 1
  python test_image.py --dataset-test --all-semesters --n-samples 2
  python test_image.py --list-checkpoints
        """)

    # What to test
    parser.add_argument("--image", type=str, help="Path to a single image to test")
    parser.add_argument("--folder", type=str, help="Path to folder of images to test")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--dataset-test", action="store_true",
                        help="Test random samples from the dataset itself")

    # Dataset test options
    parser.add_argument("--semester", type=int, default=None,
                        help="Which semester to test (1-5)")
    parser.add_argument("--all-semesters", action="store_true",
                        help="Test all semesters")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of random samples per student (default: 1)")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (e.g. baseline_after_sem5.pt). "
                             "Default: latest checkpoint.")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List all available checkpoints and exit")

    # Output
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to show (default: 5)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ScriptSentry — Image Testing")
    print("=" * 60)

    # ---- List checkpoints ----
    if args.list_checkpoints:
        ckpts = list_checkpoints()
        if not ckpts:
            print("  No checkpoints found in checkpoints/")
        else:
            print(f"\n  Available checkpoints ({len(ckpts)}):")
            for c in ckpts:
                size_mb = os.path.getsize(c) / (1024 * 1024)
                print(f"    • {os.path.basename(c)}  ({size_mb:.1f} MB)")
        return

    # ---- Validate args ----
    if not (args.image or args.folder or args.interactive or args.dataset_test):
        parser.print_help()
        print("\n  ✗ Specify --image, --folder, --interactive, or --dataset-test")
        return

    # ---- Resolve checkpoint ----
    if args.checkpoint:
        ckpt_path = os.path.join(CKPT_DIR, args.checkpoint)
        if not os.path.isfile(ckpt_path):
            # Maybe they gave a full path
            ckpt_path = args.checkpoint
    else:
        # Default: latest checkpoint (highest semester number)
        ckpts = list_checkpoints()
        if not ckpts:
            print("  ✗ No checkpoints found. Run train.py first.")
            return
        ckpt_path = ckpts[-1]  # last = latest

    # ---- Load model ----
    print(f"\n  Loading model …")
    model, ckpt_data = load_model(ckpt_path)
    trained_semester = ckpt_data.get("semester", NUM_SEMESTERS)

    # ---- Build student ID mapping ----
    print("  Building student ID mapping …")
    idx_to_sid, sid_to_idx, semester_groups = get_student_id_map()
    print(f"  Total students: {len(idx_to_sid)}")
    print(f"  Semesters trained: {trained_semester}")

    # Compute seen classes (up to trained semester)
    seen_classes = []
    for s in range(trained_semester):
        for sid in semester_groups[s]:
            seen_classes.append(sid_to_idx[sid])
    seen_classes = sorted(seen_classes)
    print(f"  Seen classes: {len(seen_classes)}")

    # ---- Run test mode ----
    if args.image:
        test_single_image(args, model, idx_to_sid, seen_classes)
    elif args.folder:
        test_folder(args, model, idx_to_sid, seen_classes)
    elif args.interactive:
        test_interactive(args, model, idx_to_sid, seen_classes)
    elif args.dataset_test:
        test_dataset_samples(args, model, idx_to_sid, sid_to_idx,
                             semester_groups, seen_classes)

    print()


if __name__ == "__main__":
    main()
