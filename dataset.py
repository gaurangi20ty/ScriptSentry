
import os
import glob
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# ================================================================
# Paths (relative to this file)
# ================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "iam")
IAM_WORDS_DIR = os.path.join(DATA_DIR, "iam_words")
WORDS_DIR = os.path.join(IAM_WORDS_DIR, "words")
WORDS_TXT = os.path.join(IAM_WORDS_DIR, "words.txt")
FORMS_TXT = os.path.join(IAM_WORDS_DIR, "forms.txt")


# ================================================================
# Image Transforms
# ================================================================
class PadToSquare:
    """Pad an image to a square (white background) to preserve aspect ratio."""

    def __call__(self, img):
        """Pad image so width == height, centered."""
        w, h = img.size
        side = max(w, h)
        # white fill value depends on mode
        fill = 255 if img.mode in ("L", "1") else (255, 255, 255)
        new_img = Image.new(img.mode, (side, side), fill)
        new_img.paste(img, ((side - w) // 2, (side - h) // 2))
        return new_img


def get_train_transform():
    """Training transform with stronger augmentation for 224x224."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])


def get_test_transform():
    """Test transform: clean resize and normalize (no augmentation)."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ================================================================
# IAM File Parsers
# ================================================================
def parse_forms_txt(forms_path):
    """
    Parse IAM forms.txt → {form_id: writer_id}.

    File format (lines starting with # are comments):
        a01-000u 000 2 prt 6 5 ...
    """
    mapping = {}
    with open(forms_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    print(f"  forms.txt  → {len(mapping)} forms, "
          f"{len(set(mapping.values()))} unique writers")
    return mapping


def parse_words_txt(words_path):
    """
    Parse IAM words.txt → list of (word_id, form_id).

    Only keeps words with segmentation result 'ok'.
    File format:
        a01-000u-00-00 ok 154 1 408 768 27 51 AT A
    """
    entries = []
    with open(words_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            word_id = parts[0]
            seg = parts[1]
            if seg != "ok":
                continue
            # word_id = a01-000u-00-00 → form_id = a01-000u
            tokens = word_id.split("-")
            if len(tokens) >= 2:
                form_id = f"{tokens[0]}-{tokens[1]}"
                entries.append((word_id, form_id))
    print(f"  words.txt  → {len(entries)} valid word entries")
    return entries


def find_images(words_dir):
    """
    Recursively find all .png images under the words directory.
    Returns {word_id: absolute_path} regardless of subfolder layout.
    """
    print(f"  Scanning images in {words_dir} ...")
    img_map = {}
    for path in glob.iglob(os.path.join(words_dir, "**", "*.png"), recursive=True):
        img_map[Path(path).stem] = path
    print(f"  Found {len(img_map)} .png images")
    return img_map


# ================================================================
# Build Student Dataset
# ================================================================
def build_student_dataset(data_dir=None):
    """
    Build {student_id: [image_paths, ...]} from IAM data.

    Strategy:
      • If forms.txt exists → group by real writer_id  (best)
      • Otherwise           → group by form_id         (fallback)

    Returns:
        student_samples : dict  {student_id: [paths]}
        using_writers   : bool  True if real writer IDs used
    """
    wd = os.path.join(data_dir, "iam_words", "words") if data_dir else WORDS_DIR
    wt = os.path.join(data_dir, "iam_words", "words.txt") if data_dir else WORDS_TXT
    ft = os.path.join(data_dir, "iam_words", "forms.txt") if data_dir else FORMS_TXT

    # --- Validate required files ---
    if not os.path.isdir(wd):
        raise FileNotFoundError(
            f"Missing 'words/' directory at:\n  {wd}\n\n"
            "Download from Kaggle:\n"
            "  https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database\n"
            "Extract and place the 'words' folder inside data/iam/"
        )
    if not os.path.isfile(wt):
        raise FileNotFoundError(
            f"Missing 'words.txt' at:\n  {wt}\n\n"
            "It should be included in the Kaggle download.\n"
            "Place it inside data/iam/"
        )

    # --- Step 1: find images ---
    img_map = find_images(wd)
    if not img_map:
        raise RuntimeError(f"No .png images found under {wd}")

    # --- Step 2: parse words.txt ---
    entries = parse_words_txt(wt)

    # --- Step 3: writer mapping ---
    using_writers = os.path.isfile(ft)
    if using_writers:
        print("  ✓ Using real writer IDs from forms.txt")
        form_to_writer = parse_forms_txt(ft)
    else:
        print("  ℹ forms.txt not found — using form IDs as student identifiers")
        print("    (each exam form = one student; works fine for the demo)")
        form_to_writer = None

    # --- Step 4: group by student ---
    student_samples = defaultdict(list)
    matched = 0

    for word_id, form_id in entries:
        if word_id not in img_map:
            continue
        if using_writers:
            if form_id not in form_to_writer:
                continue
            sid = f"w{form_to_writer[form_id]}"
        else:
            sid = form_id          # form as student
        student_samples[sid].append(img_map[word_id])
        matched += 1

    print(f"  Matched {matched} images → {len(student_samples)} students")

    # --- Step 5: filter by minimum sample count ---
    MIN_SAMPLES = 30
    student_samples = {
        s: paths for s, paths in student_samples.items()
        if len(paths) >= MIN_SAMPLES
    }
    print(f"  After filter (>={MIN_SAMPLES} samples): {len(student_samples)} students")
    return dict(student_samples), using_writers


# ================================================================
# PyTorch Dataset
# ================================================================
class WriterDataset(Dataset):
    """
    PyTorch Dataset: each item is a (word_image_tensor, label) pair
    where label is the integer class index for the student/writer.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths : list[str]   absolute paths to word images
            labels      : list[int]   integer class labels
            transform   : callable    torchvision transform pipeline
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load image, apply transform, return (tensor, label)."""
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.new("L", (128, 128), 255)
        if self.transform:
            img = self.transform(img)
        return img, label


# ================================================================
# Semester Loaders
# ================================================================
def load_semester_data(
    semester_id,
    writers_per_semester=20,
    batch_size=32,
    student_samples=None,
    selected_students=None,
    data_dir=None,
):
    """
    Build train and test DataLoaders for one semester.

    Args:
        semester_id        : int  1-based semester number
        writers_per_semester: int  students per semester
        batch_size         : int  DataLoader batch size
        student_samples    : dict pre-loaded {sid: [paths]}  (avoids re-parse)
        selected_students  : list ordered student IDs
        data_dir           : str  path to IAM data root

    Returns:
        train_loader  : DataLoader
        test_loader   : DataLoader
        semester_sids : list[str]  student IDs in this semester
        label_map     : dict       {sid: int} global label mapping
    """
    if student_samples is None:
        student_samples, _ = build_student_dataset(data_dir)
    if selected_students is None:
        selected_students = sorted(student_samples.keys())

    # Global label map — consistent across all semesters
    label_map = {sid: idx for idx, sid in enumerate(selected_students)}

    # Slice this semester's students
    start = (semester_id - 1) * writers_per_semester
    end = start + writers_per_semester
    sem_sids = selected_students[start:end]

    # Gather paths + labels
    paths, labels = [], []
    for sid in sem_sids:
        for p in student_samples.get(sid, []):
            paths.append(p)
            labels.append(label_map[sid])

    # Reproducible 80/20 train / test split
    paired = list(zip(paths, labels))
    random.Random(42 + semester_id).shuffle(paired)
    cut = int(len(paired) * 0.8)
    train_pairs, test_pairs = paired[:cut], paired[cut:]

    def _unzip(pairs):
        if pairs:
            return list(zip(*pairs))
        return [], []

    tr_paths, tr_labels = _unzip(train_pairs)
    te_paths, te_labels = _unzip(test_pairs)

    train_ds = WriterDataset(list(tr_paths), list(tr_labels), get_train_transform())
    test_ds = WriterDataset(list(te_paths), list(te_labels), get_test_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    print(f"  Semester {semester_id}: {len(sem_sids)} students | "
          f"{len(train_pairs)} train / {len(test_pairs)} test")
    return train_loader, test_loader, sem_sids, label_map


def load_all_semesters(
    num_semesters=5,
    writers_per_semester=20,
    batch_size=32,
    data_dir=None,
):
    """
    Load every semester's data in one call.

    Picks the top students by sample count so each class has
    plenty of training images.

    Returns:
        list of (train_loader, test_loader, student_ids, label_map)
    """
    print("\n" + "=" * 60)
    print("  Loading IAM Handwriting Dataset")
    print("=" * 60)

    student_samples, real_writers = build_student_dataset(data_dir)

    total_needed = num_semesters * writers_per_semester
    by_count = sorted(student_samples, key=lambda s: len(student_samples[s]),
                      reverse=True)

    available = min(len(by_count), total_needed)
    if available < total_needed:
        num_semesters = available // writers_per_semester
        total_needed = num_semesters * writers_per_semester
        print(f"  Adjusted to {num_semesters} semesters "
              f"({available} students available)")

    selected = sorted(by_count[:total_needed])
    lo = len(student_samples[by_count[total_needed - 1]])
    hi = len(student_samples[by_count[0]])
    print(f"\n  Selected top {total_needed} students "
          f"(samples per student: {lo}–{hi})")
    print(f"  IDs: {selected[:3]} … {selected[-3:]}\n")

    semesters = []
    for sid in range(1, num_semesters + 1):
        data = load_semester_data(
            sid, writers_per_semester, batch_size,
            student_samples=student_samples,
            selected_students=selected,
            data_dir=data_dir,
        )
        semesters.append(data)
    return semesters


# ================================================================
# Standalone Validation
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ScriptSentry — Dataset Validation")
    print("=" * 60)

    try:
        semesters = load_all_semesters(5, 20, 32)

        print("\n--- Sample Batch Check ---")
        for i, (tr, te, sids, lm) in enumerate(semesters):
            imgs, labs = next(iter(tr))
            print(f"  Semester {i+1}: batch {imgs.shape}, "
                  f"labels {labs[:5].tolist()}")

        print("\n  ✓ Dataset ready! Run  python train.py  to start Phase 1.\n")

    except FileNotFoundError as e:
        print(f"\n  ✗ {e}\n")
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
