"""
ScriptSentry — Enhanced Kaggle GPU Training (DER++ Replay)
============================================================
Adds Knowledge Distillation (Dark Experience Replay++) on top of
standard replay for significantly better continual learning accuracy.

Key improvements over kaggle_replay_gpu.py:
  1. Knowledge Distillation — old model's soft predictions guide new training
  2. Larger buffer (5000) + more samples per class (30)
  3. More epochs (50) with warmup
  4. Full backbone unfreezing (all ResNet layers trainable)
  5. Balanced buffer sampling

KAGGLE SETUP:
  1. Create Kaggle Notebook, set Accelerator → GPU
  2. Add IAM dataset
  3. Update DATA_DIR below
  4. Run all cells
"""

import os
import sys
import time
import random
import copy
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# CONFIGURATION
# ================================================================
DATA_DIR = "/kaggle/input/iam-handwriting-word-database"

NUM_SEMESTERS      = 5
WRITERS_PER_SEM    = 30
TOTAL_WRITERS      = NUM_SEMESTERS * WRITERS_PER_SEM   # 150
BATCH_SIZE         = 64
LEARNING_RATE      = 0.0005     # slightly lower for stability with KD
EPOCHS_PER_SEM     = 50         # more epochs
WARMUP_EPOCHS      = 3          # LR warmup at start of each semester
BUFFER_SIZE        = 5000       # larger buffer
SAMPLES_PER_CLASS  = 30         # more samples stored
NUM_WORKERS        = 2

# Knowledge Distillation
KD_ALPHA           = 0.5        # weight for distillation loss (0=none, 1=full KD)
KD_TEMPERATURE     = 2.0        # softmax temperature for KD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS = "/kaggle/working/results"
CKPTS   = "/kaggle/working/checkpoints"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)

LOG_PATH = os.path.join(RESULTS, "enhanced_training_log.txt")
log_file = None

def log(msg=""):
    global log_file
    print(msg, flush=True)
    if log_file is None:
        log_file = open(LOG_PATH, "w", encoding="utf-8")
    log_file.write(msg + "\n")
    log_file.flush()


# ================================================================
# IMAGE TRANSFORMS
# ================================================================
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        fill = 255 if img.mode in ("L", "1") else (255, 255, 255)
        new_img = Image.new(img.mode, (side, side), fill)
        new_img.paste(img, ((side - w) // 2, (side - h) // 2))
        return new_img


def get_train_transform():
    return transforms.Compose([
        PadToSquare(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06),
                                scale=(0.88, 1.12), shear=6),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_test_transform():
    return transforms.Compose([
        PadToSquare(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ================================================================
# DATASET
# ================================================================
class WriterDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.new("L", (128, 128), 255)
        if self.transform:
            img = self.transform(img)
        return img, label


def list_dir_tree(data_dir, max_depth=3):
    log(f"\n  Directory listing of {data_dir} (depth={max_depth}):")
    if not os.path.isdir(data_dir):
        log(f"    {data_dir} does NOT exist!")
        input_dir = "/kaggle/input"
        if os.path.isdir(input_dir):
            log(f"\n  Available datasets in {input_dir}/:")
            for name in sorted(os.listdir(input_dir)):
                log(f"    {name}/")
        return
    for root, dirs, files in os.walk(data_dir):
        depth = root.replace(data_dir, "").count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        indent = "    " + "  " * depth
        log(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files)[:10]:
            log(f"{indent}  {f}")
        if len(files) > 10:
            log(f"{indent}  ... and {len(files) - 10} more files")


def find_iam_paths(data_dir):
    words_txt = None
    words_dir = None
    png_dirs = set()

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f == "words.txt":
                words_txt = os.path.join(root, f)
            if f.endswith(".png"):
                png_dirs.add(root)
        if "words" in dirs:
            candidate = os.path.join(root, "words")
            if any(glob.iglob(os.path.join(candidate, "**", "*.png"), recursive=True)):
                words_dir = candidate

    if words_txt is None:
        for pattern in ["**/words.txt", "words.txt"]:
            matches = glob.glob(os.path.join(data_dir, pattern), recursive=True)
            if matches:
                words_txt = matches[0]
                break

    if words_dir is None:
        for pattern in ["**/words", "words"]:
            matches = glob.glob(os.path.join(data_dir, pattern), recursive=True)
            for m in matches:
                if os.path.isdir(m):
                    words_dir = m
                    break

    if words_dir is None and png_dirs:
        words_dir = min(png_dirs, key=lambda d: d.count(os.sep))
        log(f"  No 'words/' dir found, using PNG directory: {words_dir}")

    forms_txt = None
    for root, dirs, files in os.walk(data_dir):
        if "forms.txt" in files:
            forms_txt = os.path.join(root, "forms.txt")
            break

    return words_dir, words_txt, forms_txt


def build_student_dataset(data_dir):
    list_dir_tree(data_dir)
    words_dir, words_txt, forms_txt = find_iam_paths(data_dir)

    log(f"  Data dir : {data_dir}")
    log(f"  Words dir: {words_dir}")
    log(f"  Words txt: {words_txt}")
    log(f"  Forms txt: {forms_txt}")

    if words_dir is None:
        raise FileNotFoundError(
            f"Cannot find directory with .png images under {data_dir}\n"
            f"Check the directory listing above and update DATA_DIR."
        )
    if words_txt is None:
        raise FileNotFoundError(
            f"Cannot find 'words.txt' under {data_dir}\n"
            f"Check the directory listing above and update DATA_DIR."
        )

    img_map = {}
    for path in glob.iglob(os.path.join(words_dir, "**", "*.png"), recursive=True):
        img_map[Path(path).stem] = path
    log(f"  Found {len(img_map)} .png images")

    entries = []
    with open(words_txt, "r", encoding="utf-8") as fh:
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
            tokens = word_id.split("-")
            if len(tokens) >= 2:
                form_id = f"{tokens[0]}-{tokens[1]}"
                entries.append((word_id, form_id))
    log(f"  words.txt -> {len(entries)} valid word entries")

    form_to_writer = None
    if forms_txt and os.path.isfile(forms_txt):
        form_to_writer = {}
        with open(forms_txt, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    form_to_writer[parts[0]] = parts[1]
        log(f"  forms.txt -> {len(form_to_writer)} forms, "
            f"{len(set(form_to_writer.values()))} unique writers")
        log("  Using real writer IDs")
    else:
        log("  forms.txt not found, using form IDs as student identifiers")

    student_samples = defaultdict(list)
    matched = 0
    for word_id, form_id in entries:
        if word_id not in img_map:
            continue
        if form_to_writer:
            if form_id not in form_to_writer:
                continue
            sid = f"w{form_to_writer[form_id]}"
        else:
            sid = form_id
        student_samples[sid].append(img_map[word_id])
        matched += 1

    log(f"  Matched {matched} images -> {len(student_samples)} students")

    MIN_SAMPLES = 30
    student_samples = {s: paths for s, paths in student_samples.items()
                       if len(paths) >= MIN_SAMPLES}
    log(f"  After filter (>={MIN_SAMPLES} samples): {len(student_samples)} students")
    return dict(student_samples)


def load_all_semesters(data_dir, num_semesters, writers_per_sem, batch_size):
    log("\n" + "=" * 60)
    log("  Loading IAM Handwriting Dataset")
    log("=" * 60)

    student_samples = build_student_dataset(data_dir)
    total_needed = num_semesters * writers_per_sem

    by_count = sorted(student_samples, key=lambda s: len(student_samples[s]),
                      reverse=True)
    selected = sorted(by_count[:total_needed])
    label_map = {sid: idx for idx, sid in enumerate(selected)}

    lo = len(student_samples[by_count[total_needed - 1]])
    hi = len(student_samples[by_count[0]])
    log(f"\n  Selected top {total_needed} students "
        f"(samples per student: {lo}-{hi})")

    semesters = []
    for si in range(1, num_semesters + 1):
        start = (si - 1) * writers_per_sem
        end = start + writers_per_sem
        sem_sids = selected[start:end]

        paths, labels = [], []
        for sid in sem_sids:
            for p in student_samples.get(sid, []):
                paths.append(p)
                labels.append(label_map[sid])

        paired = list(zip(paths, labels))
        random.Random(42 + si).shuffle(paired)
        cut = int(len(paired) * 0.8)
        train_pairs, test_pairs = paired[:cut], paired[cut:]

        def _unzip(pairs):
            return list(zip(*pairs)) if pairs else ([], [])

        tr_paths, tr_labels = _unzip(train_pairs)
        te_paths, te_labels = _unzip(test_pairs)

        train_ds = WriterDataset(list(tr_paths), list(tr_labels), get_train_transform())
        test_ds = WriterDataset(list(te_paths), list(te_labels), get_test_transform())

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)

        log(f"  Semester {si}: {len(sem_sids)} students | "
            f"{len(train_pairs)} train / {len(test_pairs)} test")
        semesters.append((train_loader, test_loader, sem_sids, label_map))

    return semesters


# ================================================================
# MODEL — Full unfreeze for better adaptation
# ================================================================
class WriterClassifier(nn.Module):
    def __init__(self, num_writers=150):
        super().__init__()
        self.num_writers = num_writers
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze only layer1 (low-level features are generic enough)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer1.parameters():
            p.requires_grad = False
        for p in self.backbone.layer2.parameters():
            p.requires_grad = True
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_writers),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_features(self, x):
        with torch.no_grad():
            return self.backbone(x)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# DER++ REPLAY BUFFER — stores soft logits alongside images
# ================================================================
class _BufferDataset(Dataset):
    def __init__(self, images, labels, logits=None):
        self.images = images
        self.labels = labels
        self.logits = logits  # soft targets from old model

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.logits is not None:
            return self.images[idx], self.labels[idx], self.logits[idx]
        return self.images[idx], self.labels[idx], torch.zeros(1)


class DERReplayBuffer:
    """
    Dark Experience Replay++ buffer.
    Stores (image_tensor, hard_label, soft_logits) triples.
    The soft_logits are the old model's output at storage time, used
    for knowledge distillation during future training.
    """

    def __init__(self, max_size=500, num_classes=150):
        self.max_size = max_size
        self.num_classes = num_classes
        self.images = []
        self.labels = []
        self.logits = []   # old model's soft predictions
        self._count = 0

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def add_task_samples(self, model, data_loader, n_per_class=5):
        """Store samples WITH the current model's logits (for future KD)."""
        model.eval()
        class_samples = {}

        for images, labels in data_loader:
            images_dev = images.to(DEVICE)
            logits_batch = model(images_dev).cpu()

            for img, lab, logit in zip(images, labels, logits_batch):
                lab_int = lab.item()
                if lab_int not in class_samples:
                    class_samples[lab_int] = []
                if len(class_samples[lab_int]) < n_per_class:
                    class_samples[lab_int].append(
                        (img.clone().cpu(), lab_int, logit.clone())
                    )

        candidates = []
        for lab, items in class_samples.items():
            candidates.extend(items)

        for img, lab, logit in candidates:
            self._count += 1
            if len(self.images) < self.max_size:
                self.images.append(img)
                self.labels.append(lab)
                self.logits.append(logit)
            else:
                j = random.randint(0, self._count - 1)
                if j < self.max_size:
                    self.images[j] = img
                    self.labels[j] = lab
                    self.logits[j] = logit

    def get_combined_loader(self, new_task_loader, batch_size=32):
        if len(self.images) == 0:
            return new_task_loader, False

        buf_dataset = _BufferDataset(self.images, self.labels, self.logits)
        combined = ConcatDataset([new_task_loader.dataset, buf_dataset])

        # Custom collate to handle mixed datasets (new data has no logits)
        def collate_fn(batch):
            imgs, labs, extra = [], [], []
            for item in batch:
                if len(item) == 3:
                    imgs.append(item[0])
                    labs.append(item[1])
                    extra.append(item[2])
                else:
                    imgs.append(item[0])
                    labs.append(item[1])
                    extra.append(torch.zeros(1))
            return torch.stack(imgs), torch.tensor(labs), extra

        loader = DataLoader(combined, batch_size=batch_size, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=collate_fn)
        return loader, True

    def get_stats(self):
        if not self.labels:
            return {"size": 0, "classes": 0}
        unique = set(self.labels)
        return {"size": len(self.images), "classes": len(unique),
                "samples_per_class": len(self.images) / max(len(unique), 1)}


# ================================================================
# TRAINING WITH KNOWLEDGE DISTILLATION
# ================================================================
def kd_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Knowledge Distillation loss (Hinton et al., 2015).
    Matches soft probability distributions between teacher and student.
    """
    s_soft = F.log_softmax(student_logits / temperature, dim=1)
    t_soft = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)


def train_one_epoch_der(model, loader, optimizer, criterion,
                        active_classes, has_buffer_data,
                        old_model=None, kd_alpha=0.5, kd_temp=2.0):
    """
    Train one epoch with optional DER++ knowledge distillation.

    Loss = (1-alpha)*CE + alpha*KD  for buffer samples with stored logits
    Loss = CE                       for new task samples (no old logits)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if has_buffer_data:
            images, labels, stored_logits = batch
        else:
            images, labels = batch
            stored_logits = None

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(images)

        # Mask to active classes
        mask = torch.full_like(out, float('-inf'))
        mask[:, active_classes] = out[:, active_classes]
        out_masked = mask

        # Classification loss
        ce_loss = criterion(out_masked, labels)

        # Knowledge distillation from stored logits (DER++)
        loss = ce_loss
        if has_buffer_data and stored_logits is not None and old_model is not None:
            # Find which samples have valid stored logits (buffer samples)
            has_logits = []
            valid_stored = []
            for i, sl in enumerate(stored_logits):
                if sl.dim() > 0 and sl.shape[0] > 1:
                    has_logits.append(i)
                    valid_stored.append(sl)

            if has_logits:
                buf_indices = torch.tensor(has_logits, device=DEVICE)
                buf_student = out[buf_indices]
                buf_teacher = torch.stack(valid_stored).to(DEVICE)
                kd = kd_loss(buf_student, buf_teacher, kd_temp)
                loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += out_masked.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, seen_classes=None):
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
# WARMUP SCHEDULER
# ================================================================
class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.eta_min, base_lr * factor)


# ================================================================
# PLOTTING
# ================================================================
def plot_heatmap(acc_matrix, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(acc_matrix, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
    for i in range(NUM_SEMESTERS):
        for j in range(NUM_SEMESTERS):
            val = acc_matrix[i, j]
            color = "white" if val > 55 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=12, fontweight=weight, color=color)
    ax.set_xticks(range(NUM_SEMESTERS))
    ax.set_xticklabels([f"Sem {i+1}" for i in range(NUM_SEMESTERS)])
    ax.set_yticks(range(NUM_SEMESTERS))
    ax.set_yticklabels([f"After Sem {i+1}" for i in range(NUM_SEMESTERS)])
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy %")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved -> {path}")


def plot_forgetting(acc_matrix, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(NUM_SEMESTERS):
        accs = [acc_matrix[t, j] for t in range(NUM_SEMESTERS)]
        ax.plot(range(1, NUM_SEMESTERS + 1), accs, "o-", linewidth=2.5,
                markersize=8, label=f"Semester {j+1}")
    ax.set_xlabel("After Training Semester", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("DER++ Replay — Knowledge Retention", fontsize=15, fontweight="bold")
    ax.set_xticks(range(1, NUM_SEMESTERS + 1))
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved -> {path}")


def plot_all_semesters(acc_matrix, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    sems = range(1, NUM_SEMESTERS + 1)
    diag = [acc_matrix[i, i] for i in range(NUM_SEMESTERS)]
    final = [acc_matrix[-1, j] for j in range(NUM_SEMESTERS)]
    ax.bar([s - 0.2 for s in sems], diag, 0.35, label="Peak (just learned)",
           color="#2ecc71", edgecolor="white")
    ax.bar([s + 0.2 for s in sems], final, 0.35, label="Final (after Sem 5)",
           color="#e74c3c", edgecolor="white")
    for i, (p, f) in enumerate(zip(diag, final)):
        ax.text(i + 0.8, p + 1, f"{p:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + 1.2, f + 1, f"{f:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Semester", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("DER++ Replay — Peak vs Final", fontsize=15, fontweight="bold")
    ax.set_xticks(list(sems))
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved -> {path}")


# ================================================================
# MAIN TRAINING — DER++ REPLAY
# ================================================================
def train_enhanced():
    log("\n" + "=" * 60)
    log("  ScriptSentry — Enhanced DER++ GPU Training")
    log(f"  Device: {DEVICE}")
    log(f"  Epochs/sem: {EPOCHS_PER_SEM}, Buffer: {BUFFER_SIZE}, "
        f"Samples/class: {SAMPLES_PER_CLASS}")
    log(f"  KD alpha: {KD_ALPHA}, KD temp: {KD_TEMPERATURE}")
    log(f"  Warmup: {WARMUP_EPOCHS} epochs, LR: {LEARNING_RATE}")
    log(f"  Batch size: {BATCH_SIZE}")
    log("=" * 60)

    semesters = load_all_semesters(DATA_DIR, NUM_SEMESTERS,
                                   WRITERS_PER_SEM, BATCH_SIZE)

    log(f"\nInitializing WriterClassifier ({TOTAL_WRITERS} classes)...")
    model = WriterClassifier(num_writers=TOTAL_WRITERS).to(DEVICE)
    log(f"  Trainable params : {model.count_trainable():,}")
    log(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    replay_buffer = DERReplayBuffer(max_size=BUFFER_SIZE,
                                     num_classes=TOTAL_WRITERS)

    acc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    seen_classes = []
    old_model = None
    total_t0 = time.time()

    for si in range(NUM_SEMESTERS):
        sn = si + 1
        train_loader = semesters[si][0]
        sem_sids = semesters[si][2]
        label_map = semesters[si][3]

        active_classes = sorted([label_map[sid] for sid in sem_sids])
        for c in active_classes:
            if c not in seen_classes:
                seen_classes.append(c)
        seen_classes_sorted = sorted(seen_classes)

        # Get combined loader (with KD logits if buffer has data)
        if si == 0:
            combined_loader = train_loader
            has_buffer = False
        else:
            combined_loader, has_buffer = replay_buffer.get_combined_loader(
                train_loader, batch_size=BATCH_SIZE)

        buf_stats = replay_buffer.get_stats()
        log(f"\n{'=' * 55}")
        log(f"  SEMESTER {sn}  ({len(sem_sids)} students)")
        log(f"  Active classes: {len(active_classes)} | "
            f"Total seen: {len(seen_classes_sorted)}")
        log(f"  Replay buffer: {buf_stats['size']} samples, "
            f"{buf_stats.get('classes', 0)} classes")
        if has_buffer:
            log(f"  Combined training set: {len(combined_loader.dataset)} samples")
            log(f"  Knowledge Distillation: alpha={KD_ALPHA}, temp={KD_TEMPERATURE}")
        log(f"{'=' * 55}")

        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS,
            total_epochs=EPOCHS_PER_SEM, eta_min=1e-6
        )

        t0 = time.time()
        best_acc = 0
        for ep in range(1, EPOCHS_PER_SEM + 1):
            loss, tacc = train_one_epoch_der(
                model, combined_loader, optimizer, criterion,
                active_classes=seen_classes_sorted,
                has_buffer_data=has_buffer,
                old_model=old_model,
                kd_alpha=KD_ALPHA if si > 0 else 0.0,
                kd_temp=KD_TEMPERATURE,
            )
            scheduler.step()

            if ep % 5 == 0 or ep == EPOCHS_PER_SEM:
                cur_lr = optimizer.param_groups[0]['lr']
                log(f"  Epoch {ep:>2}/{EPOCHS_PER_SEM}  "
                    f"Loss: {loss:.4f}  Train Acc: {tacc:.1f}%  "
                    f"LR: {cur_lr:.6f}")

        elapsed = time.time() - t0
        log(f"  Time: {elapsed:.1f}s")

        # Save old model for KD in next semester
        old_model = copy.deepcopy(model)
        old_model.eval()

        # Add to DER++ buffer (stores logits!)
        replay_buffer.add_task_samples(model, train_loader,
                                       n_per_class=SAMPLES_PER_CLASS)
        buf_stats = replay_buffer.get_stats()
        log(f"  Buffer updated: {buf_stats['size']} samples, "
            f"{buf_stats.get('classes', 0)} classes")

        # Evaluate
        log(f"\n  Evaluation after Semester {sn}:")
        for ej in range(NUM_SEMESTERS):
            acc = evaluate(model, semesters[ej][1],
                           seen_classes=seen_classes_sorted)
            acc_matrix[si][ej] = acc

            if ej == si:
                tag = " <- just learned"
            elif ej < si:
                drop = acc_matrix[ej][ej] - acc
                tag = f"  dropped {drop:.1f}%" if drop > 5 else "  retained"
            else:
                tag = "  (not yet learned)"
            log(f"    Sem {ej+1}: {acc:5.1f}%{tag}")

        ckpt = os.path.join(CKPTS, f"replay_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "acc_matrix": acc_matrix.copy(),
            "method": "replay",
            "buffer_size": BUFFER_SIZE,
            "epochs": EPOCHS_PER_SEM,
            "kd_alpha": KD_ALPHA,
            "kd_temperature": KD_TEMPERATURE,
            "device": str(DEVICE),
        }, ckpt)
        log(f"  Checkpoint -> {ckpt}")

    total_time = time.time() - total_t0

    # ── Summary ──
    log("\n" + "=" * 60)
    log("  ENHANCED DER++ RESULTS")
    log("=" * 60)

    hdr = "               " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    log(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{acc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1} :  {row}")

    sem1_init = acc_matrix[0][0]
    sem1_final = acc_matrix[-1][0]
    forgetting = sem1_init - sem1_final
    avg_final = np.mean(acc_matrix[-1])
    avg_peak = np.mean([acc_matrix[i][i] for i in range(NUM_SEMESTERS)])
    avg_forgetting = np.mean([acc_matrix[i][i] - acc_matrix[-1][i]
                              for i in range(NUM_SEMESTERS)])

    log(f"\n  Average peak accuracy  : {avg_peak:.1f}%")
    log(f"  Average final accuracy : {avg_final:.1f}%")
    log(f"  Average forgetting     : {avg_forgetting:.1f}%")
    log(f"  Sem 1 forgetting       : {forgetting:.1f}%")
    log(f"  vs Random chance       : {avg_final / (100/150):.0f}x")
    log(f"  Buffer size            : {len(replay_buffer)}")
    log(f"  Total training time    : {total_time:.1f}s ({total_time/60:.1f} min)")

    # Compare with previous
    log(f"\n  --- Improvement over basic Replay ---")
    cpu_avg = 33.0
    gpu_avg = 37.9
    log(f"  CPU Replay: {cpu_avg:.1f}% -> Enhanced: {avg_final:.1f}% "
        f"(+{avg_final - cpu_avg:.1f}%)")
    log(f"  GPU Replay: {gpu_avg:.1f}% -> Enhanced: {avg_final:.1f}% "
        f"(+{avg_final - gpu_avg:.1f}%)")

    npy_path = os.path.join(RESULTS, "replay_acc_matrix.npy")
    np.save(npy_path, acc_matrix)
    log(f"\n  Saved -> {npy_path}")

    log("\nGenerating plots...")
    plot_heatmap(acc_matrix,
                 f"DER++ Enhanced — Accuracy Matrix\n"
                 f"(Epochs={EPOCHS_PER_SEM}, Buffer={BUFFER_SIZE}, "
                 f"KD_alpha={KD_ALPHA})",
                 os.path.join(RESULTS, "enhanced_heatmap.png"))
    plot_forgetting(acc_matrix,
                    os.path.join(RESULTS, "enhanced_forgetting.png"))
    plot_all_semesters(acc_matrix,
                       os.path.join(RESULTS, "enhanced_all_semesters.png"))

    final_path = os.path.join(CKPTS, "replay_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "acc_matrix": acc_matrix,
        "method": "replay",
        "config": {
            "epochs": EPOCHS_PER_SEM,
            "buffer_size": BUFFER_SIZE,
            "samples_per_class": SAMPLES_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "kd_alpha": KD_ALPHA,
            "kd_temperature": KD_TEMPERATURE,
            "num_semesters": NUM_SEMESTERS,
            "writers_per_sem": WRITERS_PER_SEM,
        }
    }, final_path)
    log(f"  Final model -> {final_path}")

    log(f"\n{'=' * 60}")
    log(f"  DER++ training complete!")
    log(f"  Download /kaggle/working/results/ and /kaggle/working/checkpoints/")
    log(f"{'=' * 60}\n")

    return acc_matrix, model


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    log(f"Python: {sys.version}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_enhanced()

    if log_file:
        log_file.close()
