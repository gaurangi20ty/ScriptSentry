"""
ScriptSentry — ResNet-34 Replay Ablation (Kaggle GPU)
======================================================
Identical to GPU Replay V1 except backbone is ResNet-34 instead of
ResNet-18. Same FC head, same replay buffer, same hyperparameters.

PURPOSE: Isolate the effect of backbone capacity.
  - If gain is small (~2-3%), proves the bottleneck is classifier bias
  - Justifies why V3 needs NME + cosine classifier, not just a bigger model

KAGGLE SETUP:
  1. Create Kaggle Notebook, set Accelerator → GPU T4 x2 or P100
  2. Add IAM dataset
  3. Update DATA_DIR below
  4. Run all cells
"""

import os
import sys
import time
import random
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# CONFIGURATION — Same as V1 for fair comparison
# ================================================================
DATA_DIR = "/kaggle/input/iam-handwriting-word-database"

NUM_SEMESTERS      = 5
WRITERS_PER_SEM    = 30
TOTAL_WRITERS      = NUM_SEMESTERS * WRITERS_PER_SEM   # 150
BATCH_SIZE         = 64
LEARNING_RATE      = 0.001       # same as V1
EPOCHS_PER_SEM     = 30          # same as V1
BUFFER_SIZE        = 3000        # same as V1
SAMPLES_PER_CLASS  = 20          # same as V1
NUM_WORKERS        = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS = "/kaggle/working/results"
CKPTS   = "/kaggle/working/checkpoints"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)

LOG_PATH = os.path.join(RESULTS, "resnet34_replay_log.txt")
log_file = None

def log(msg=""):
    global log_file
    print(msg, flush=True)
    if log_file is None:
        log_file = open(LOG_PATH, "w", encoding="utf-8")
    log_file.write(msg + "\n")
    log_file.flush()


# ================================================================
# IMAGE TRANSFORMS (identical to V1)
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
    return transforms.Compose([
        PadToSquare(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ================================================================
# DATASET (identical to V1)
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
# MODEL — ResNet-34 + same FC head as V1
# ================================================================
class WriterClassifier34(nn.Module):
    """
    Only change vs V1: ResNet-34 backbone instead of ResNet-18.
    Same FC head (512 -> 256 -> ReLU -> Dropout -> 150).
    Same frozen/unfrozen layers.
    """
    def __init__(self, num_writers=150):
        super().__init__()
        self.num_writers = num_writers
        # >>> THE ONLY CHANGE: resnet34 instead of resnet18 <<<
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Same freeze strategy as V1
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer2.parameters():
            p.requires_grad = True
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        in_features = self.backbone.fc.in_features  # 512 (same as ResNet-18)
        self.backbone.fc = nn.Identity()

        # Same FC head as V1
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_writers),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# REPLAY BUFFER (identical to V1)
# ================================================================
class _BufferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ReplayBuffer:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.images = []
        self.labels = []
        self._count = 0

    def __len__(self):
        return len(self.images)

    def add_task_samples(self, data_loader, n_per_class=5):
        class_samples = {}
        for images, labels in data_loader:
            for img, lab in zip(images, labels):
                lab_int = lab.item()
                if lab_int not in class_samples:
                    class_samples[lab_int] = []
                if len(class_samples[lab_int]) < n_per_class:
                    class_samples[lab_int].append(img.clone().cpu())

        candidates = []
        for lab, imgs in class_samples.items():
            for img in imgs:
                candidates.append((img, lab))

        for img, lab in candidates:
            self._count += 1
            if len(self.images) < self.max_size:
                self.images.append(img)
                self.labels.append(lab)
            else:
                j = random.randint(0, self._count - 1)
                if j < self.max_size:
                    self.images[j] = img
                    self.labels[j] = lab

    def get_combined_loader(self, new_task_loader, batch_size=32):
        if len(self.images) == 0:
            return new_task_loader
        buf_dataset = _BufferDataset(self.images, self.labels)
        combined = ConcatDataset([new_task_loader.dataset, buf_dataset])
        return DataLoader(combined, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

    def get_stats(self):
        if not self.labels:
            return {"size": 0, "classes": 0}
        unique = set(self.labels)
        return {"size": len(self.images), "classes": len(unique),
                "samples_per_class": len(self.images) / max(len(unique), 1)}


# ================================================================
# TRAINING (identical to V1)
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, active_classes=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)

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
    ax.set_title("ResNet-34 Replay — Knowledge Retention",
                 fontsize=15, fontweight="bold")
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


def plot_peak_vs_final(acc_matrix, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    sems = range(1, NUM_SEMESTERS + 1)
    diag = [acc_matrix[i, i] for i in range(NUM_SEMESTERS)]
    final = [acc_matrix[-1, j] for j in range(NUM_SEMESTERS)]
    ax.bar([s - 0.2 for s in sems], diag, 0.35, label="Peak (just learned)",
           color="#2ecc71", edgecolor="white")
    ax.bar([s + 0.2 for s in sems], final, 0.35, label="Final (after Sem 5)",
           color="#e74c3c", edgecolor="white")
    for i, (p, f) in enumerate(zip(diag, final)):
        ax.text(i + 0.8, p + 1, f"{p:.0f}%", ha="center", fontsize=9,
                fontweight="bold")
        ax.text(i + 1.2, f + 1, f"{f:.0f}%", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("Semester", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("ResNet-34 Replay — Peak vs Final Accuracy",
                 fontsize=15, fontweight="bold")
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
# MAIN TRAINING LOOP
# ================================================================
def train_replay():
    log("\n" + "=" * 60)
    log("  ScriptSentry — ResNet-34 Replay (Ablation)")
    log(f"  Device: {DEVICE}")
    log(f"  Backbone: ResNet-34 (vs ResNet-18 in V1)")
    log(f"  Classifier: Same FC head as V1")
    log(f"  Epochs/sem: {EPOCHS_PER_SEM}, Buffer: {BUFFER_SIZE}, "
        f"Samples/class: {SAMPLES_PER_CLASS}")
    log(f"  Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    log("=" * 60)

    semesters = load_all_semesters(DATA_DIR, NUM_SEMESTERS,
                                   WRITERS_PER_SEM, BATCH_SIZE)

    log(f"\nInitializing WriterClassifier34 ({TOTAL_WRITERS} classes)...")
    model = WriterClassifier34(num_writers=TOTAL_WRITERS).to(DEVICE)
    log(f"  Trainable params : {model.count_trainable():,}")
    log(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(max_size=BUFFER_SIZE)

    acc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    seen_classes = []
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

        if si == 0:
            combined_loader = train_loader
        else:
            combined_loader = replay_buffer.get_combined_loader(
                train_loader, batch_size=BATCH_SIZE)

        buf_stats = replay_buffer.get_stats()
        log(f"\n{'=' * 55}")
        log(f"  SEMESTER {sn}  ({len(sem_sids)} students)")
        log(f"  Active classes: {len(active_classes)} | "
            f"Total seen: {len(seen_classes_sorted)}")
        log(f"  Replay buffer: {buf_stats['size']} samples, "
            f"{buf_stats.get('classes', 0)} classes")
        if si > 0:
            log(f"  Combined training set: {len(combined_loader.dataset)} samples")
        log(f"{'=' * 55}")

        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_PER_SEM, eta_min=1e-5
        )

        t0 = time.time()
        for ep in range(1, EPOCHS_PER_SEM + 1):
            loss, tacc = train_one_epoch(
                model, combined_loader, optimizer, criterion,
                active_classes=seen_classes_sorted)
            scheduler.step()
            log(f"  Epoch {ep}/{EPOCHS_PER_SEM}  "
                f"Loss: {loss:.4f}  Train Acc: {tacc:.1f}%")

        elapsed = time.time() - t0
        log(f"  Time: {elapsed:.1f}s")

        replay_buffer.add_task_samples(train_loader,
                                       n_per_class=SAMPLES_PER_CLASS)
        buf_stats = replay_buffer.get_stats()
        log(f"  Buffer updated: {buf_stats['size']} samples, "
            f"{buf_stats.get('classes', 0)} classes")

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

        ckpt = os.path.join(CKPTS, f"r34_replay_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "acc_matrix": acc_matrix.copy(),
            "method": "resnet34_replay",
            "buffer_size": BUFFER_SIZE,
            "epochs": EPOCHS_PER_SEM,
            "device": str(DEVICE),
        }, ckpt)
        log(f"  Checkpoint -> {ckpt}")

    total_time = time.time() - total_t0

    # ── Summary ──
    log("\n" + "=" * 60)
    log("  RESNET-34 REPLAY RESULTS")
    log("=" * 60)

    hdr = "               " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    log(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{acc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1} :  {row}")

    avg_final = np.mean(acc_matrix[-1])
    avg_peak = np.mean([acc_matrix[i][i] for i in range(NUM_SEMESTERS)])
    avg_forgetting = np.mean([acc_matrix[i][i] - acc_matrix[-1][i]
                              for i in range(NUM_SEMESTERS)])

    log(f"\n  Average peak accuracy  : {avg_peak:.1f}%")
    log(f"  Average final accuracy : {avg_final:.1f}%")
    log(f"  Average forgetting     : {avg_forgetting:.1f}%")
    log(f"  vs Random chance       : {avg_final / (100/150):.0f}x")
    log(f"  Buffer size            : {len(replay_buffer)}")
    log(f"  Total training time    : {total_time:.1f}s ({total_time/60:.1f} min)")

    log(f"\n  --- Comparison ---")
    log(f"  ResNet-18 Replay (V1) : 37.9% avg final")
    log(f"  ResNet-34 Replay      : {avg_final:.1f}% avg final")
    log(f"  Difference            : {avg_final - 37.9:+.1f}%")
    log(f"  (If small, confirms classifier bias is the real bottleneck)")

    # Save
    npy_path = os.path.join(RESULTS, "r34_replay_acc_matrix.npy")
    np.save(npy_path, acc_matrix)
    log(f"\n  Saved -> {npy_path}")

    log("\nGenerating plots...")
    plot_heatmap(acc_matrix,
                 f"ResNet-34 Replay — Accuracy Matrix\n"
                 f"(Epochs={EPOCHS_PER_SEM}, Buffer={BUFFER_SIZE})",
                 os.path.join(RESULTS, "r34_replay_heatmap.png"))
    plot_forgetting(acc_matrix,
                    os.path.join(RESULTS, "r34_replay_forgetting.png"))
    plot_peak_vs_final(acc_matrix,
                       os.path.join(RESULTS, "r34_replay_peak_vs_final.png"))

    final_path = os.path.join(CKPTS, "r34_replay_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "acc_matrix": acc_matrix,
        "method": "resnet34_replay",
        "config": {
            "model": "resnet34_fc",
            "epochs": EPOCHS_PER_SEM,
            "buffer_size": BUFFER_SIZE,
            "samples_per_class": SAMPLES_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
        }
    }, final_path)
    log(f"  Final model -> {final_path}")

    log(f"\n{'=' * 60}")
    log(f"  ResNet-34 Replay ablation complete!")
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

    train_replay()

    if log_file:
        log_file.close()
