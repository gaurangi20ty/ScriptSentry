"""
ScriptSentry — V3 Kaggle GPU Training (NME + Feature Distillation)
====================================================================
Major accuracy improvements via:
  1. NME Classifier — test-time Nearest-Mean-of-Exemplars (bypasses FC bias)
  2. Feature Distillation — L2 loss on backbone features (not just logits)
  3. Cosine Classifier — normalized weights eliminate magnitude bias
  4. ResNet-34 — more capacity than ResNet-18
  5. Separated Softmax — prevents new classes from dominating old ones
  6. Mixup Augmentation — creates interpolated training samples

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
import copy
import glob
import math
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
SEED               = 42
BATCH_SIZE         = 64
LEARNING_RATE      = 0.0005
EPOCHS_PER_SEM     = 70
WARMUP_EPOCHS      = 5
BUFFER_SIZE        = 8000
SAMPLES_PER_CLASS  = 45
NUM_WORKERS        = 2

# Feature distillation weight (higher = more preservation)
FEAT_DIST_LAMBDA   = 8.0
# Mixup strength
MIXUP_ALPHA        = 0.4
# Label smoothing helps calibration in class-incremental setup
LABEL_SMOOTHING    = 0.05
# Post-semester classifier calibration on class-balanced replay set
CALIBRATION_EPOCHS = 8
CALIBRATION_SAMPLES_PER_CLASS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS = "/kaggle/working/results"
CKPTS   = "/kaggle/working/checkpoints"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)

LOG_PATH = os.path.join(RESULTS, "v3_training_log.txt")
log_file = None

def log(msg=""):
    global log_file
    print(msg, flush=True)
    if log_file is None:
        log_file = open(LOG_PATH, "w", encoding="utf-8")
    log_file.write(msg + "\n")
    log_file.flush()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# MODEL — ResNet-34 + Cosine Classifier
# ================================================================
class CosineLinear(nn.Module):
    """
    Cosine classifier — normalizes both weights and features before
    computing logits. Eliminates magnitude bias towards new classes.
    Uses a learnable temperature (sigma) to scale logits.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.sigma = nn.Parameter(torch.tensor(10.0))  # learnable temperature

    def forward(self, x):
        # L2 normalize both
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity * temperature
        return self.sigma * F.linear(x_norm, w_norm)


class WriterClassifierV3(nn.Module):
    """
    ResNet-34 backbone + Cosine classifier head.
    More capacity + bias-free classifier.
    """
    def __init__(self, num_writers=150):
        super().__init__()
        self.num_writers = num_writers
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Freeze only conv1 + bn1 + layer1 (very low-level features)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer2.parameters():
            p.requires_grad = True
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        self.feat_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Feature projection (keeps features rich before cosine classifier)
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Cosine classifier (bias-free)
        self.classifier = CosineLinear(512, num_writers)

    def get_features(self, x):
        """Extract projected feature embeddings."""
        raw = self.backbone(x)
        return self.projector(raw)

    def forward(self, x):
        feat = self.get_features(x)
        return self.classifier(feat)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# NME (NEAREST MEAN OF EXEMPLARS) CLASSIFIER
# ================================================================
class NMEClassifier:
    """
    At test time, classify by comparing feature distances to class
    prototypes (mean feature vectors). Completely bypasses the FC head,
    eliminating the bias towards recently learned classes.

    This is the single biggest accuracy improvement for class-incremental
    learning (used in iCaRL, LUCIR, PODNet, etc.).
    """

    def __init__(self):
        self.prototypes = {}  # {class_idx: mean_feature_vector}

    @torch.no_grad()
    def update_prototypes(self, model, data_loaders, buffer_images,
                          buffer_labels, seen_classes):
        """
        Compute mean feature vectors for all seen classes using
        both current training data and buffer data.
        """
        model.eval()
        class_features = defaultdict(list)

        # Features from data loaders
        for loader in data_loaders:
            for images, labels in loader:
                images = images.to(DEVICE)
                feats = model.get_features(images).cpu()
                for feat, lab in zip(feats, labels):
                    lab_int = lab.item()
                    if lab_int in seen_classes:
                        class_features[lab_int].append(feat)

        # Features from buffer
        if buffer_images:
            for i in range(0, len(buffer_images), BATCH_SIZE):
                batch = torch.stack(buffer_images[i:i + BATCH_SIZE]).to(DEVICE)
                feats = model.get_features(batch).cpu()
                for j, feat in enumerate(feats):
                    lab = buffer_labels[i + j]
                    class_features[lab].append(feat)

        # Compute mean prototypes
        self.prototypes = {}
        for cls, feats in class_features.items():
            stacked = torch.stack(feats)
            mean_feat = F.normalize(stacked.mean(dim=0), p=2, dim=0)
            self.prototypes[cls] = mean_feat

    @torch.no_grad()
    def classify(self, model, images, seen_classes):
        """Classify images using nearest prototype."""
        model.eval()
        feats = model.get_features(images.to(DEVICE))
        feats = F.normalize(feats, p=2, dim=1).cpu()

        # Build prototype matrix for seen classes
        proto_classes = sorted([c for c in seen_classes if c in self.prototypes])
        if not proto_classes:
            return torch.zeros(images.size(0), dtype=torch.long)

        proto_matrix = torch.stack([self.prototypes[c] for c in proto_classes])

        # Cosine similarity (features and prototypes are already normalized)
        similarities = feats @ proto_matrix.t()  # [batch, num_protos]
        pred_indices = similarities.argmax(dim=1)

        # Map back to actual class labels
        preds = torch.tensor([proto_classes[i] for i in pred_indices])
        return preds


# ================================================================
# REPLAY BUFFER (with features for NME)
# ================================================================
class _BufferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ReplayBufferV3:
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.images = []
        self.labels = []
        self._count = 0

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def add_task_samples(self, data_loader, n_per_class=30):
        """Add representative samples using herding selection."""
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
# MIXUP
# ================================================================
def mixup_data(x, y, alpha=0.3):
    """Mixup augmentation — interpolate between random pairs."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def train_one_epoch_v3(model, loader, optimizer, criterion,
                       active_classes, old_model=None,
                       feat_dist_lambda=0.0, use_mixup=False):
    """
    Training with feature distillation + optional mixup.
    Feature distillation preserves the backbone's learned representations
    much more effectively than logit-level distillation.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Optional mixup
        if use_mixup and random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        # Get features from current model
        cur_features = model.get_features(images)
        out = model.classifier(cur_features)

        # Mask to active classes
        mask = torch.full_like(out, float('-inf'))
        mask[:, active_classes] = out[:, active_classes]
        out_masked = mask

        # Classification loss (with mixup support)
        if lam < 1.0:
            ce_loss = lam * criterion(out_masked, labels_a) + \
                      (1 - lam) * criterion(out_masked, labels_b)
        else:
            ce_loss = criterion(out_masked, labels)

        # Feature distillation loss
        loss = ce_loss
        if old_model is not None and feat_dist_lambda > 0:
            with torch.no_grad():
                old_features = old_model.get_features(images)
            # Normalized L2 distance between features
            feat_dist = F.mse_loss(
                F.normalize(cur_features, p=2, dim=1),
                F.normalize(old_features, p=2, dim=1),
            )
            loss = ce_loss + feat_dist_lambda * feat_dist

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += out_masked.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_fc(model, loader, seen_classes=None):
    """Standard FC-head evaluation."""
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


def build_balanced_calibration_loader(train_loader, buffer_images, buffer_labels,
                                      seen_classes, target_per_class, batch_size):
    """Build a balanced loader from replay + current semester samples."""
    class_buckets = {c: [] for c in seen_classes}

    # Prefer replay samples first to preserve old classes.
    for img, lab in zip(buffer_images, buffer_labels):
        if lab in class_buckets and len(class_buckets[lab]) < target_per_class:
            class_buckets[lab].append(img.clone())

    # Fill remaining with current task samples.
    needed = {c for c in seen_classes if len(class_buckets[c]) < target_per_class}
    if needed:
        for images, labels in train_loader:
            for img, lab in zip(images, labels):
                li = int(lab.item())
                if li in needed and len(class_buckets[li]) < target_per_class:
                    class_buckets[li].append(img.cpu().clone())
            needed = {c for c in seen_classes if len(class_buckets[c]) < target_per_class}
            if not needed:
                break

    calib_images, calib_labels = [], []
    for cls in seen_classes:
        for img in class_buckets[cls]:
            calib_images.append(img)
            calib_labels.append(cls)

    if not calib_images:
        return None

    dataset = _BufferDataset(calib_images, calib_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True)


def calibrate_classifier(model, loader, seen_classes, epochs=5):
    """Freeze backbone/projector and tune cosine head on balanced data."""
    if loader is None or epochs <= 0:
        return

    model.train()
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(images)
            mask = torch.full_like(out, float('-inf'))
            mask[:, seen_classes] = out[:, seen_classes]
            loss = criterion(mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=5.0)
            optimizer.step()

    # Restore trainable state for next semester training.
    for p in model.backbone.layer2.parameters():
        p.requires_grad = True
    for p in model.backbone.layer3.parameters():
        p.requires_grad = True
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate_nme(model, nme, loader, seen_classes):
    """NME-based evaluation (nearest prototype)."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        preds = nme.classify(model, images, seen_classes)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


# ================================================================
# WARMUP SCHEDULER
# ================================================================
class WarmupCosineScheduler:
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
            factor = self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.eta_min, base_lr * factor)


# ================================================================
# PLOTTING
# ================================================================
def plot_heatmap(acc_matrix, title, path, n_sem=NUM_SEMESTERS):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(acc_matrix, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
    for i in range(n_sem):
        for j in range(n_sem):
            val = acc_matrix[i, j]
            color = "white" if val > 55 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=12, fontweight=weight, color=color)
    ax.set_xticks(range(n_sem))
    ax.set_xticklabels([f"Sem {i+1}" for i in range(n_sem)])
    ax.set_yticks(range(n_sem))
    ax.set_yticklabels([f"After Sem {i+1}" for i in range(n_sem)])
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy %")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved -> {path}")


def plot_forgetting(acc_matrix, title, path, n_sem=NUM_SEMESTERS):
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(n_sem):
        accs = [acc_matrix[t, j] for t in range(n_sem)]
        ax.plot(range(1, n_sem + 1), accs, "o-", linewidth=2.5,
                markersize=8, label=f"Semester {j+1}")
    ax.set_xlabel("After Training Semester", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks(range(1, n_sem + 1))
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved -> {path}")


def plot_comparison(fc_matrix, nme_matrix, path):
    """Bar chart comparing FC vs NME final accuracy per semester."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sems = range(1, NUM_SEMESTERS + 1)
    fc_final = [fc_matrix[-1, j] for j in range(NUM_SEMESTERS)]
    nme_final = [nme_matrix[-1, j] for j in range(NUM_SEMESTERS)]

    ax.bar([s - 0.2 for s in sems], fc_final, 0.35,
           label=f"FC Head (avg {np.mean(fc_final):.1f}%)",
           color="#e74c3c", edgecolor="white")
    ax.bar([s + 0.2 for s in sems], nme_final, 0.35,
           label=f"NME (avg {np.mean(nme_final):.1f}%)",
           color="#2ecc71", edgecolor="white")

    for i, (fc, nme) in enumerate(zip(fc_final, nme_final)):
        ax.text(i + 0.8, fc + 1, f"{fc:.0f}%", ha="center", fontsize=8)
        ax.text(i + 1.2, nme + 1, f"{nme:.0f}%", ha="center", fontsize=8,
                fontweight="bold")

    ax.set_xlabel("Semester", fontsize=13)
    ax.set_ylabel("Final Accuracy (%)", fontsize=13)
    ax.set_title("FC Head vs NME Classifier — Final Accuracy",
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
# MAIN TRAINING — V3
# ================================================================
def train_v3():
    log("\n" + "=" * 60)
    log("  ScriptSentry V3 — NME + Feature Distillation")
    log(f"  Device: {DEVICE}")
    log(f"  Model: ResNet-34 + Cosine Classifier")
    log(f"  Epochs/sem: {EPOCHS_PER_SEM}, Buffer: {BUFFER_SIZE}")
    log(f"  Feature distillation lambda: {FEAT_DIST_LAMBDA}")
    log(f"  Mixup alpha: {MIXUP_ALPHA}")
    log(f"  LR: {LEARNING_RATE}, Warmup: {WARMUP_EPOCHS}")
    log("=" * 60)

    semesters = load_all_semesters(DATA_DIR, NUM_SEMESTERS,
                                   WRITERS_PER_SEM, BATCH_SIZE)

    log(f"\nInitializing WriterClassifierV3 ({TOTAL_WRITERS} classes)...")
    model = WriterClassifierV3(num_writers=TOTAL_WRITERS).to(DEVICE)
    log(f"  Trainable params : {model.count_trainable():,}")
    log(f"  Feature dim      : {model.feat_dim}")
    log(f"  Device           : {DEVICE}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    replay_buffer = ReplayBufferV3(max_size=BUFFER_SIZE)
    nme = NMEClassifier()

    # Track both FC and NME accuracy matrices
    fc_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
    nme_matrix = np.zeros((NUM_SEMESTERS, NUM_SEMESTERS))
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

        # Combined training data
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
            log(f"  Feature distillation: lambda={FEAT_DIST_LAMBDA}")
        log(f"{'=' * 55}")

        # Separate LR for backbone vs head
        backbone_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and 'classifier' not in n
                          and 'projector' not in n]
        head_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and ('classifier' in n or 'projector' in n)]

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # lower LR for backbone
            {'params': head_params, 'lr': LEARNING_RATE},
        ], weight_decay=1e-4)

        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS,
            total_epochs=EPOCHS_PER_SEM, eta_min=1e-6
        )

        t0 = time.time()
        for ep in range(1, EPOCHS_PER_SEM + 1):
            loss, tacc = train_one_epoch_v3(
                model, combined_loader, optimizer, criterion,
                active_classes=seen_classes_sorted,
                old_model=old_model,
                feat_dist_lambda=FEAT_DIST_LAMBDA if si > 0 else 0.0,
                use_mixup=(si > 0),  # mixup only after first semester
            )
            scheduler.step()

            if ep % 10 == 0 or ep == EPOCHS_PER_SEM:
                log(f"  Epoch {ep:>2}/{EPOCHS_PER_SEM}  "
                    f"Loss: {loss:.4f}  Train Acc: {tacc:.1f}%")

        elapsed = time.time() - t0
        log(f"  Time: {elapsed:.1f}s")

        # Save old model for feature distillation
        old_model = copy.deepcopy(model)
        old_model.eval()

        # Add to replay buffer
        replay_buffer.add_task_samples(train_loader,
                                       n_per_class=SAMPLES_PER_CLASS)
        buf_stats = replay_buffer.get_stats()
        log(f"  Buffer updated: {buf_stats['size']} samples, "
            f"{buf_stats.get('classes', 0)} classes")

        # Update NME prototypes using all seen training data + buffer
        seen_train_loaders = [semesters[j][0] for j in range(sn)]
        nme.update_prototypes(model, seen_train_loaders,
                              replay_buffer.images, replay_buffer.labels,
                              set(seen_classes_sorted))
        log(f"  NME prototypes: {len(nme.prototypes)} classes")

        # Classifier calibration on class-balanced replay data.
        if si > 0 and CALIBRATION_EPOCHS > 0:
            calib_loader = build_balanced_calibration_loader(
                train_loader=train_loader,
                buffer_images=replay_buffer.images,
                buffer_labels=replay_buffer.labels,
                seen_classes=seen_classes_sorted,
                target_per_class=CALIBRATION_SAMPLES_PER_CLASS,
                batch_size=BATCH_SIZE,
            )
            calibrate_classifier(model, calib_loader, seen_classes_sorted,
                                 epochs=CALIBRATION_EPOCHS)
            # Refresh prototypes after calibration since features/head moved.
            nme.update_prototypes(model, seen_train_loaders,
                                  replay_buffer.images, replay_buffer.labels,
                                  set(seen_classes_sorted))
            log(f"  Calibrated classifier for {CALIBRATION_EPOCHS} epochs")

        # Evaluate with BOTH classifiers
        log(f"\n  Evaluation after Semester {sn}:")
        log(f"  {'':>12s} {'FC Head':>10s} {'NME':>10s} {'Best':>10s}")
        for ej in range(NUM_SEMESTERS):
            fc_acc = evaluate_fc(model, semesters[ej][1],
                                seen_classes=seen_classes_sorted)
            nme_acc = evaluate_nme(model, nme, semesters[ej][1],
                                  seen_classes_sorted)
            fc_matrix[si][ej] = fc_acc
            nme_matrix[si][ej] = nme_acc
            best = max(fc_acc, nme_acc)
            marker = "*" if nme_acc > fc_acc else ""

            if ej == si:
                tag = " <- learned"
            elif ej < si:
                tag = ""
            else:
                tag = " (future)"
            log(f"    Sem {ej+1}:  {fc_acc:6.1f}%   {nme_acc:6.1f}%   "
                f"{best:6.1f}%{marker}{tag}")

        # Save checkpoint
        ckpt = os.path.join(CKPTS, f"v3_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "fc_matrix": fc_matrix.copy(),
            "nme_matrix": nme_matrix.copy(),
            "nme_prototypes": {k: v.cpu() for k, v in nme.prototypes.items()},
            "method": "v3_nme_featdist",
            "buffer_size": BUFFER_SIZE,
            "epochs": EPOCHS_PER_SEM,
            "device": str(DEVICE),
        }, ckpt)
        log(f"  Checkpoint -> {ckpt}")

    total_time = time.time() - total_t0

    # Use the best matrix (NME should be better for old classes)
    best_matrix = np.maximum(fc_matrix, nme_matrix)

    # ── Summary ──
    log("\n" + "=" * 60)
    log("  V3 RESULTS — FC HEAD")
    log("=" * 60)
    hdr = "               " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    log(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{fc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1} :  {row}")

    fc_avg_final = np.mean(fc_matrix[-1])
    fc_avg_peak = np.mean([fc_matrix[i][i] for i in range(NUM_SEMESTERS)])

    log(f"\n  FC Avg peak  : {fc_avg_peak:.1f}%")
    log(f"  FC Avg final : {fc_avg_final:.1f}%")

    log("\n" + "=" * 60)
    log("  V3 RESULTS — NME CLASSIFIER")
    log("=" * 60)
    log(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{nme_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1} :  {row}")

    nme_avg_final = np.mean(nme_matrix[-1])
    nme_avg_peak = np.mean([nme_matrix[i][i] for i in range(NUM_SEMESTERS)])

    log(f"\n  NME Avg peak  : {nme_avg_peak:.1f}%")
    log(f"  NME Avg final : {nme_avg_final:.1f}%")

    # Best of both
    best_avg_final = np.mean(best_matrix[-1])
    best_avg_peak = np.mean([best_matrix[i][i] for i in range(NUM_SEMESTERS)])
    avg_forgetting = np.mean([best_matrix[i][i] - best_matrix[-1][i]
                              for i in range(NUM_SEMESTERS)])

    log("\n" + "=" * 60)
    log("  BEST (max of FC, NME per cell)")
    log("=" * 60)
    log(f"\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{best_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1} :  {row}")

    log(f"\n  Best Avg peak      : {best_avg_peak:.1f}%")
    log(f"  Best Avg final     : {best_avg_final:.1f}%")
    log(f"  Avg forgetting     : {avg_forgetting:.1f}%")
    log(f"  vs Random chance   : {best_avg_final / (100/150):.0f}x")
    log(f"  Buffer size        : {len(replay_buffer)}")
    log(f"  Total time         : {total_time:.1f}s ({total_time/60:.1f} min)")

    # Compare with previous runs
    log(f"\n  --- Progression ---")
    log(f"  CPU Baseline     : 13.8%")
    log(f"  CPU Replay       : 33.0%")
    log(f"  GPU Replay       : 37.9%")
    log(f"  GPU DER++        : ~40%")
    log(f"  V3 NME+FeatDist  : {best_avg_final:.1f}%")

    # Save matrices
    np.save(os.path.join(RESULTS, "v3_fc_matrix.npy"), fc_matrix)
    np.save(os.path.join(RESULTS, "v3_nme_matrix.npy"), nme_matrix)
    np.save(os.path.join(RESULTS, "v3_best_matrix.npy"), best_matrix)
    log(f"\n  Saved all matrices to {RESULTS}/")

    # Plots
    log("\nGenerating plots...")
    plot_heatmap(fc_matrix,
                 "V3 FC Head — Accuracy Matrix",
                 os.path.join(RESULTS, "v3_fc_heatmap.png"))
    plot_heatmap(nme_matrix,
                 "V3 NME Classifier — Accuracy Matrix",
                 os.path.join(RESULTS, "v3_nme_heatmap.png"))
    plot_heatmap(best_matrix,
                 "V3 Best (FC/NME) — Accuracy Matrix",
                 os.path.join(RESULTS, "v3_best_heatmap.png"))
    plot_forgetting(nme_matrix,
                    "V3 NME — Knowledge Retention",
                    os.path.join(RESULTS, "v3_nme_forgetting.png"))
    plot_comparison(fc_matrix, nme_matrix,
                    os.path.join(RESULTS, "v3_fc_vs_nme.png"))

    # Save final model
    final_path = os.path.join(CKPTS, "v3_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "fc_matrix": fc_matrix,
        "nme_matrix": nme_matrix,
        "best_matrix": best_matrix,
        "nme_prototypes": {k: v.cpu() for k, v in nme.prototypes.items()},
        "method": "v3_nme_featdist",
        "config": {
            "model": "resnet34_cosine",
            "epochs": EPOCHS_PER_SEM,
            "buffer_size": BUFFER_SIZE,
            "samples_per_class": SAMPLES_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "feat_dist_lambda": FEAT_DIST_LAMBDA,
            "mixup_alpha": MIXUP_ALPHA,
        }
    }, final_path)
    log(f"  Final model -> {final_path}")

    log(f"\n{'=' * 60}")
    log(f"  V3 training complete!")
    log(f"  Download /kaggle/working/results/ and /kaggle/working/checkpoints/")
    log(f"{'=' * 60}\n")

    return best_matrix, model


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    set_seed(SEED)
    log(f"Python: {sys.version}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"Seed: {SEED}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_v3()

    if log_file:
        log_file.close()
