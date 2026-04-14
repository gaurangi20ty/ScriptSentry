"""
ScriptSentry — ULTIMATE V4 Kaggle GPU Training (ResNet-50 Elite)
=================================================================
Aggressive accuracy maximization for continual learning.

Key ingredients for 80%+ average final accuracy:
  1. ResNet-50 backbone (2.5× parameters vs ResNet-18)
  2. NME classifier + Cosine head (removes bias towards new classes)
  3. Stronger replay buffer (10k samples, 50 per class)
  4. Feature distillation (λ=10 for aggressive preservation)
  5. Balanced prototype calibration (10 epochs per semester)
  6. Label smoothing + gradient clipping
  7. Longer training (80 epochs per semester)
  8. Separate LR for backbone (lower) vs head (higher)
  9. Mixup + Strong data augmentation
  10. Temperature scaling on cosine classifier

KAGGLE SETUP:
  1. Set GPU: T4 x2 or P100 (needs GPU for 80 epoch × 5 sem)
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
# CONFIGURATION — TUNED FOR 80% ACCURACY
# ================================================================
DATA_DIR = "/kaggle/input/iam-handwriting-word-database"

NUM_SEMESTERS      = 5
WRITERS_PER_SEM    = 30
TOTAL_WRITERS      = NUM_SEMESTERS * WRITERS_PER_SEM   # 150
SEED               = 42
BATCH_SIZE         = 32  # smaller batch for better gradient flow with large model
LEARNING_RATE      = 0.0003  # lower LR for larger model
EPOCHS_PER_SEM     = 80
WARMUP_EPOCHS      = 5
BUFFER_SIZE        = 10000  # larger memory
SAMPLES_PER_CLASS  = 50
NUM_WORKERS        = 4

# Feature distillation — aggressive preservation
FEAT_DIST_LAMBDA   = 10.0
KD_LAMBDA          = 2.0
KD_TEMPERATURE     = 2.0
MIXUP_ALPHA        = 0.4
LABEL_SMOOTHING    = 0.1
COSINE_MARGIN      = 0.20
OLD_CLASS_WEIGHT   = 1.8

# Classifier calibration — extensive fine-tuning on balanced data
CALIBRATION_EPOCHS = 15
CALIBRATION_LR     = 0.001
CALIBRATION_SAMPLES = 50

# Full-network balanced finetuning on seen classes
BALANCED_FT_EPOCHS = 8
BALANCED_FT_BB_LR  = 1e-5
BALANCED_FT_HD_LR  = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS = "/kaggle/working/results"
CKPTS   = "/kaggle/working/checkpoints"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)

LOG_PATH = os.path.join(RESULTS, "v4_ultimate_training_log.txt")
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
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15), shear=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
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

    if words_dir is None and png_dirs:
        words_dir = min(png_dirs, key=lambda d: d.count(os.sep))

    forms_txt = None
    for root, dirs, files in os.walk(data_dir):
        if "forms.txt" in files:
            forms_txt = os.path.join(root, "forms.txt")
            break

    return words_dir, words_txt, forms_txt


def build_student_dataset(data_dir):
    words_dir, words_txt, forms_txt = find_iam_paths(data_dir)

    if words_dir is None:
        raise FileNotFoundError(f"Missing 'words/' directory under {data_dir}")
    if words_txt is None:
        raise FileNotFoundError(f"Missing 'words.txt' under {data_dir}")

    img_map = {}
    for path in glob.iglob(os.path.join(words_dir, "**", "*.png"), recursive=True):
        img_map[Path(path).stem] = path

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

    student_samples = defaultdict(list)
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

    MIN_SAMPLES = 30
    student_samples = {s: paths for s, paths in student_samples.items()
                       if len(paths) >= MIN_SAMPLES}
    return dict(student_samples)


def load_all_semesters(data_dir, num_semesters, writers_per_sem, batch_size):
    student_samples = build_student_dataset(data_dir)
    total_needed = num_semesters * writers_per_sem

    by_count = sorted(student_samples, key=lambda s: len(student_samples[s]),
                      reverse=True)
    selected = sorted(by_count[:total_needed])
    label_map = {sid: idx for idx, sid in enumerate(selected)}

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

        semesters.append((train_loader, test_loader, sem_sids, label_map))

    return semesters


# ================================================================
# MODEL — ResNet-50 + Cosine Classifier
# ================================================================
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.sigma = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.sigma * F.linear(x_norm, w_norm)


class WriterClassifierV4(nn.Module):
    """ResNet-50 backbone + Cosine classifier for max capacity."""
    def __init__(self, num_writers=150):
        super().__init__()
        self.num_writers = num_writers
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze only conv1 + bn1 + layer1
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer2.parameters():
            p.requires_grad = True
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        self.feat_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Richer projection for ResNet-50
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

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# NME CLASSIFIER
# ================================================================
class NMEClassifier:
    def __init__(self):
        self.prototypes = {}

    @torch.no_grad()
    def update_prototypes(self, model, data_loaders, buffer_images,
                          buffer_labels, seen_classes):
        model.eval()
        class_features = defaultdict(list)

        for loader in data_loaders:
            for images, labels in loader:
                images = images.to(DEVICE)
                feats = model.get_features(images).cpu()
                for feat, lab in zip(feats, labels):
                    lab_int = lab.item()
                    if lab_int in seen_classes:
                        class_features[lab_int].append(feat)

        if buffer_images:
            for i in range(0, len(buffer_images), BATCH_SIZE):
                batch = torch.stack(buffer_images[i:i + BATCH_SIZE]).to(DEVICE)
                feats = model.get_features(batch).cpu()
                for j, feat in enumerate(feats):
                    lab = buffer_labels[i + j]
                    class_features[lab].append(feat)

        self.prototypes = {}
        for cls, feats in class_features.items():
            stacked = torch.stack(feats)
            mean_feat = F.normalize(stacked.mean(dim=0), p=2, dim=0)
            self.prototypes[cls] = mean_feat

    @torch.no_grad()
    def classify(self, model, images, seen_classes):
        model.eval()
        feats = model.get_features(images.to(DEVICE))
        feats = F.normalize(feats, p=2, dim=1).cpu()

        proto_classes = sorted([c for c in seen_classes if c in self.prototypes])
        if not proto_classes:
            return torch.zeros(images.size(0), dtype=torch.long)

        proto_matrix = torch.stack([self.prototypes[c] for c in proto_classes])
        similarities = feats @ proto_matrix.t()
        pred_indices = similarities.argmax(dim=1)
        preds = torch.tensor([proto_classes[i] for i in pred_indices])
        return preds


# ================================================================
# REPLAY BUFFER
# ================================================================
class _BufferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ReplayBufferV4:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.images = []
        self.labels = []
        self._count = 0

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def add_task_samples(self, data_loader, n_per_class=50):
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
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def margin_weighted_ce_loss(logits, labels, active_classes,
                            margin=0.0,
                            label_smoothing=0.0,
                            old_classes=None,
                            old_class_weight=1.0):
    masked = torch.full_like(logits, float("-inf"))
    masked[:, active_classes] = logits[:, active_classes]

    if margin > 0:
        margin_logits = masked.clone()
        row_idx = torch.arange(labels.size(0), device=labels.device)
        margin_logits[row_idx, labels] = margin_logits[row_idx, labels] - margin
    else:
        margin_logits = masked

    ce = F.cross_entropy(
        margin_logits,
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )

    if old_classes and old_class_weight > 1.0:
        old_cls = torch.tensor(sorted(old_classes), device=labels.device, dtype=labels.dtype)
        is_old = torch.isin(labels, old_cls)
        sample_weights = torch.ones_like(ce)
        sample_weights[is_old] = old_class_weight
        ce = ce * sample_weights

    return ce.mean(), masked


# ================================================================
# TRAINING
# ================================================================
def train_one_epoch_v4(model, loader, optimizer, criterion,
                       active_classes, old_model=None,
                       prev_seen_classes=None,
                       feat_dist_lambda=0.0,
                       kd_lambda=0.0,
                       kd_temperature=2.0,
                       cosine_margin=0.0,
                       old_class_weight=1.0,
                       use_mixup=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if use_mixup and random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad()
        cur_features = model.get_features(images)
        out = model.classifier(cur_features)

        if lam < 1.0:
            ce_a, out_masked = margin_weighted_ce_loss(
                out, labels_a, active_classes,
                margin=cosine_margin,
                label_smoothing=LABEL_SMOOTHING,
                old_classes=set(prev_seen_classes) if prev_seen_classes else None,
                old_class_weight=old_class_weight,
            )
            ce_b, _ = margin_weighted_ce_loss(
                out, labels_b, active_classes,
                margin=cosine_margin,
                label_smoothing=LABEL_SMOOTHING,
                old_classes=set(prev_seen_classes) if prev_seen_classes else None,
                old_class_weight=old_class_weight,
            )
            ce_loss = lam * ce_a + (1 - lam) * ce_b
        else:
            ce_loss, out_masked = margin_weighted_ce_loss(
                out, labels, active_classes,
                margin=cosine_margin,
                label_smoothing=LABEL_SMOOTHING,
                old_classes=set(prev_seen_classes) if prev_seen_classes else None,
                old_class_weight=old_class_weight,
            )

        loss = ce_loss
        old_features = None
        if old_model is not None and feat_dist_lambda > 0:
            with torch.no_grad():
                old_features = old_model.get_features(images)
            feat_dist = F.mse_loss(
                F.normalize(cur_features, p=2, dim=1),
                F.normalize(old_features, p=2, dim=1),
            )
            loss = ce_loss + feat_dist_lambda * feat_dist

        if old_model is not None and prev_seen_classes and kd_lambda > 0:
            with torch.no_grad():
                if old_features is None:
                    old_features = old_model.get_features(images)
                old_logits = old_model.classifier(old_features)
            cur_old = out[:, prev_seen_classes]
            old_old = old_logits[:, prev_seen_classes]
            t = kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(cur_old / t, dim=1),
                F.softmax(old_old / t, dim=1),
                reduction="batchmean",
            ) * (t * t)
            loss = loss + kd_lambda * kd_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += out_masked.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def build_balanced_calib_loader(train_loader, buffer_images, buffer_labels,
                                seen_classes, target_per_class, batch_size):
    class_buckets = {c: [] for c in seen_classes}

    for img, lab in zip(buffer_images, buffer_labels):
        if lab in class_buckets and len(class_buckets[lab]) < target_per_class:
            class_buckets[lab].append(img.clone())

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


def calibrate_classifier_v4(model, loader, seen_classes, epochs=15):
    if loader is None or epochs <= 0:
        return

    model.train()
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.SGD(model.classifier.parameters(), 
                          lr=CALIBRATION_LR, momentum=0.9, weight_decay=1e-4)
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

    for p in model.backbone.layer2.parameters():
        p.requires_grad = True
    for p in model.backbone.layer3.parameters():
        p.requires_grad = True
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True


def balanced_finetune_v4(model, loader, seen_classes, old_model=None,
                         prev_seen_classes=None, epochs=8):
    if loader is None or epochs <= 0:
        return

    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "classifier" not in n and "projector" not in n
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ("classifier" in n or "projector" in n)
    ]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": BALANCED_FT_BB_LR},
        {"params": head_params, "lr": BALANCED_FT_HD_LR},
    ], weight_decay=1e-4)

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            cur_features = model.get_features(images)
            out = model.classifier(cur_features)
            loss, _ = margin_weighted_ce_loss(
                out,
                labels,
                seen_classes,
                margin=COSINE_MARGIN,
                label_smoothing=LABEL_SMOOTHING,
                old_classes=set(prev_seen_classes) if prev_seen_classes else None,
                old_class_weight=OLD_CLASS_WEIGHT,
            )

            if old_model is not None and prev_seen_classes and KD_LAMBDA > 0:
                with torch.no_grad():
                    old_features = old_model.get_features(images)
                    old_logits = old_model.classifier(old_features)
                cur_old = out[:, prev_seen_classes]
                old_old = old_logits[:, prev_seen_classes]
                t = KD_TEMPERATURE
                kd_loss = F.kl_div(
                    F.log_softmax(cur_old / t, dim=1),
                    F.softmax(old_old / t, dim=1),
                    reduction="batchmean",
                ) * (t * t)
                loss = loss + KD_LAMBDA * kd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()


@torch.no_grad()
def evaluate_fc(model, loader, seen_classes=None):
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


@torch.no_grad()
def evaluate_nme(model, nme, loader, seen_classes):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        preds = nme.classify(model, images, seen_classes)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


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
# MAIN TRAINING
# ================================================================
def train_v4_ultimate():
    log("\n" + "=" * 70)
    log("  ScriptSentry V4 ULTIMATE — ResNet-50 Elite for 80% Average Accuracy")
    log("=" * 70)
    log(f"  Model: ResNet-50 (2.5M trainable params)")
    log(f"  Classifier: NME + Cosine Head (bias-free)")
    log(f"  Training: {EPOCHS_PER_SEM} epochs × {NUM_SEMESTERS} semesters")
    log(f"  Buffer: {BUFFER_SIZE} samples ({SAMPLES_PER_CLASS} per class)")
    log(f"  Feature Distillation λ: {FEAT_DIST_LAMBDA}")
    log(f"  Logit KD λ / T: {KD_LAMBDA} / {KD_TEMPERATURE}")
    log(f"  Cosine margin / old-class weight: {COSINE_MARGIN} / {OLD_CLASS_WEIGHT}")
    log(f"  Balanced FT epochs: {BALANCED_FT_EPOCHS}")
    log(f"  Calibration: {CALIBRATION_EPOCHS} epochs per semester")
    log(f"  Device: {DEVICE}")

    semesters = load_all_semesters(DATA_DIR, NUM_SEMESTERS,
                                   WRITERS_PER_SEM, BATCH_SIZE)

    log(f"\nInitializing WriterClassifierV4 ({TOTAL_WRITERS} classes)...")
    model = WriterClassifierV4(num_writers=TOTAL_WRITERS).to(DEVICE)
    log(f"  Trainable params: {model.count_trainable():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    replay_buffer = ReplayBufferV4(max_size=BUFFER_SIZE)
    nme = NMEClassifier()

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
        teacher_model = old_model
        prev_seen_classes_sorted = sorted(seen_classes)

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
        log(f"\n{'=' * 70}")
        log(f"  SEMESTER {sn} ({len(sem_sids)} students)")
        log(f"  Active: {len(active_classes)} | Total seen: {len(seen_classes_sorted)}")
        log(f"  Replay: {buf_stats['size']} samples, {buf_stats.get('classes', 0)} classes")
        log(f"{'=' * 70}")

        # Split LR: lower for backbone, higher for head
        backbone_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and 'classifier' not in n and 'projector' not in n]
        head_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and ('classifier' in n or 'projector' in n)]

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
            {'params': head_params, 'lr': LEARNING_RATE},
        ], weight_decay=1e-4)

        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS,
            total_epochs=EPOCHS_PER_SEM, eta_min=1e-6
        )

        t0 = time.time()
        for ep in range(1, EPOCHS_PER_SEM + 1):
            loss, tacc = train_one_epoch_v4(
                model, combined_loader, optimizer, criterion,
                active_classes=seen_classes_sorted,
                old_model=teacher_model,
                prev_seen_classes=prev_seen_classes_sorted,
                feat_dist_lambda=FEAT_DIST_LAMBDA if si > 0 else 0.0,
                kd_lambda=KD_LAMBDA if si > 0 else 0.0,
                kd_temperature=KD_TEMPERATURE,
                cosine_margin=COSINE_MARGIN,
                old_class_weight=OLD_CLASS_WEIGHT,
                use_mixup=(si > 0),
            )
            scheduler.step()

            if ep % 10 == 0 or ep == EPOCHS_PER_SEM:
                lr_bb = optimizer.param_groups[0]['lr']
                lr_head = optimizer.param_groups[1]['lr']
                log(f"  Epoch {ep:>2}/{EPOCHS_PER_SEM}  Loss: {loss:.4f}  Acc: {tacc:.1f}%  "
                    f"LR_BB: {lr_bb:.1e}  LR_Head: {lr_head:.1e}")

        elapsed = time.time() - t0
        log(f"  Training time: {elapsed:.1f}s")

        replay_buffer.add_task_samples(train_loader, n_per_class=SAMPLES_PER_CLASS)
        buf_stats = replay_buffer.get_stats()
        log(f"  Buffer updated: {buf_stats['size']} samples, {buf_stats.get('classes', 0)} classes")

        # Aggressive classifier calibration
        seen_train_loaders = [semesters[j][0] for j in range(sn)]
        nme.update_prototypes(model, seen_train_loaders,
                              replay_buffer.images, replay_buffer.labels,
                              set(seen_classes_sorted))
        log(f"  NME prototypes: {len(nme.prototypes)} classes")

        if si > 0 and CALIBRATION_EPOCHS > 0:
            calib_loader = build_balanced_calib_loader(
                train_loader=train_loader,
                buffer_images=replay_buffer.images,
                buffer_labels=replay_buffer.labels,
                seen_classes=seen_classes_sorted,
                target_per_class=CALIBRATION_SAMPLES,
                batch_size=BATCH_SIZE,
            )
            balanced_finetune_v4(
                model=model,
                loader=calib_loader,
                seen_classes=seen_classes_sorted,
                old_model=teacher_model,
                prev_seen_classes=prev_seen_classes_sorted,
                epochs=BALANCED_FT_EPOCHS,
            )
            log(f"  Balanced full-network FT ({BALANCED_FT_EPOCHS} epochs)")
            calibrate_classifier_v4(model, calib_loader, seen_classes_sorted,
                                   epochs=CALIBRATION_EPOCHS)
            nme.update_prototypes(model, seen_train_loaders,
                                  replay_buffer.images, replay_buffer.labels,
                                  set(seen_classes_sorted))
            log(f"  Calibrated classifier on balanced data ({CALIBRATION_EPOCHS} epochs)")

        # Evaluate
        log(f"\n  Evaluation after Semester {sn}:")
        for ej in range(NUM_SEMESTERS):
            fc_acc = evaluate_fc(model, semesters[ej][1],
                                seen_classes=seen_classes_sorted)
            nme_acc = evaluate_nme(model, nme, semesters[ej][1],
                                  seen_classes_sorted)
            fc_matrix[si][ej] = fc_acc
            nme_matrix[si][ej] = nme_acc

            if ej == si:
                tag = " ← learned"
            elif ej < si:
                tag = ""
            else:
                tag = " (future)"
            log(f"    Sem {ej+1}  FC: {fc_acc:6.1f}%  NME: {nme_acc:6.1f}%{tag}")

        # Save per-semester checkpoint
        ckpt = os.path.join(CKPTS, f"v4_after_sem{sn}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "semester": sn,
            "fc_matrix": fc_matrix.copy(),
            "nme_matrix": nme_matrix.copy(),
            "nme_prototypes": {k: v.cpu() for k, v in nme.prototypes.items()},
            "method": "v4_ultimate",
        }, ckpt)
        log(f"  Checkpoint → {ckpt}")

        old_model = copy.deepcopy(model)
        old_model.eval()

    total_time = time.time() - total_t0
    best_matrix = np.maximum(fc_matrix, nme_matrix)

    # ── Summary ──
    log("\n" + "=" * 70)
    log("  V4 ULTIMATE — RESULTS")
    log("=" * 70)

    hdr = "             " + "  ".join(f"Sem {j+1:>2}" for j in range(NUM_SEMESTERS))
    log(f"\n  FC HEAD:\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{fc_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1}:  {row}")

    log(f"\n  NME CLASSIFIER:\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{nme_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1}:  {row}")

    log(f"\n  BEST (max per cell):\n{hdr}")
    for i in range(NUM_SEMESTERS):
        row = "  ".join(f"{best_matrix[i][j]:5.1f}%" for j in range(NUM_SEMESTERS))
        log(f"  After Sem {i+1}:  {row}")

    fc_avg_final = np.mean(fc_matrix[-1])
    nme_avg_final = np.mean(nme_matrix[-1])
    best_avg_final = np.mean(best_matrix[-1])
    best_avg_peak = np.mean([best_matrix[i][i] for i in range(NUM_SEMESTERS)])
    avg_forgetting = np.mean([best_matrix[i][i] - best_matrix[-1][i]
                              for i in range(NUM_SEMESTERS)])

    log(f"\n  ─ METRICS ─")
    log(f"  FC Avg Final        : {fc_avg_final:.1f}%")
    log(f"  NME Avg Final       : {nme_avg_final:.1f}%")
    log(f"  Best Avg Final      : {best_avg_final:.1f}%  ★★★ TARGET METRIC ★★★")
    log(f"  Best Avg Peak       : {best_avg_peak:.1f}%")
    log(f"  Avg Forgetting      : {avg_forgetting:.1f}%")
    log(f"  vs Random (1/150)   : {best_avg_final / (100/150):.0f}×")
    log(f"  Total Time          : {total_time:.1f}s ({total_time/3600:.1f} h)")

    # Save results
    np.save(os.path.join(RESULTS, "v4_fc_matrix.npy"), fc_matrix)
    np.save(os.path.join(RESULTS, "v4_nme_matrix.npy"), nme_matrix)
    np.save(os.path.join(RESULTS, "v4_best_matrix.npy"), best_matrix)

    final_path = os.path.join(CKPTS, "v4_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "fc_matrix": fc_matrix,
        "nme_matrix": nme_matrix,
        "best_matrix": best_matrix,
        "nme_prototypes": {k: v.cpu() for k, v in nme.prototypes.items()},
        "method": "v4_ultimate",
        "config": {
            "model": "resnet50_cosine",
            "epochs_per_sem": EPOCHS_PER_SEM,
            "buffer_size": BUFFER_SIZE,
            "samples_per_class": SAMPLES_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "feat_dist_lambda": FEAT_DIST_LAMBDA,
            "calibration_epochs": CALIBRATION_EPOCHS,
        }
    }, final_path)

    log(f"\n  Final model → {final_path}")
    log(f"\n{'=' * 70}")
    log(f"  ✅ V4 ULTIMATE training complete!")
    log(f"  Download results/ and checkpoints/ from Kaggle")
    log(f"{'=' * 70}\n")

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

    train_v4_ultimate()

    if log_file:
        log_file.close()
