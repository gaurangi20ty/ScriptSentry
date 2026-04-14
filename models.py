"""
ScriptSentry — Models Module
==============================
WriterClassifier: ResNet-18 backbone with a custom classification head
for handwriting writer / student identification.

Usage:
    python models.py           # quick architecture sanity check
"""

import torch
import torch.nn as nn
from torchvision import models
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset


class WriterClassifier(nn.Module):
    """
    Handwriting writer classifier.

    Architecture
    ------------
    • Backbone : ResNet-18 pretrained on ImageNet
      – All layers frozen EXCEPT layer3 + layer4
    • Head     : FC(512 → 256) → ReLU → Dropout → FC(256 → num_writers)

    Input : (B, 3, 128, 128)   3-channel grayscale (repeated)
    Output: (B, num_writers)    raw logits
    """

    def __init__(self, num_writers=100):
        """
        Args:
            num_writers: total number of writer/student classes
        """
        super().__init__()
        self.num_writers = num_writers

        # --- Backbone ---
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last three residual blocks for better adaptation
        for p in self.backbone.layer2.parameters():
            p.requires_grad = True
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        # Replace original fc with identity (we add our own head)
        in_features = self.backbone.fc.in_features        # 512
        self.backbone.fc = nn.Identity()

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_writers),
        )

    # ---------------------------------------------------------
    def forward(self, x):
        """Full forward pass → logits."""
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract 512-d feature embeddings (before head)."""
        with torch.no_grad():
            return self.backbone(x)

    # ---------------------------------------------------------
    def count_trainable(self):
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        """Total (trainable + frozen) parameters."""
        return sum(p.numel() for p in self.parameters())


# ================================================================
# Experience Replay Buffer (Phase 2)
# ================================================================
class _BufferDataset(Dataset):
    """Thin wrapper so replay buffer items return (tensor, int) like WriterDataset."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ReplayBuffer:
    """
    Fixed-size memory buffer that stores (image_tensor, label) pairs
    from previous semesters.  When full, new samples replace old ones
    via reservoir sampling to keep a balanced representation.

    Usage:
        buffer = ReplayBuffer(max_size=500)
        buffer.add_task_samples(train_loader, n_per_class=5)
        combined = buffer.get_combined_loader(new_loader, batch_size=32)
    """

    def __init__(self, max_size=500):
        """
        Args:
            max_size: maximum number of (image, label) pairs to store.
        """
        self.max_size = max_size
        self.images = []     # list of tensors
        self.labels = []     # list of ints
        self._count = 0      # total samples seen (for reservoir sampling)

    def __len__(self):
        """Current buffer size."""
        return len(self.images)

    def add_task_samples(self, data_loader, n_per_class=5):
        """
        Add representative samples from a task/semester to the buffer.

        Collects up to `n_per_class` samples for each unique label in
        the data loader.  If the buffer overflows `max_size`, reservoir
        sampling is used to decide whether to keep new or old samples.

        Args:
            data_loader : DataLoader for the semester's training set
            n_per_class : int  number of samples to store per writer
        """
        # Collect candidate samples grouped by class
        class_samples = {}
        for images, labels in data_loader:
            for img, lab in zip(images, labels):
                lab_int = lab.item()
                if lab_int not in class_samples:
                    class_samples[lab_int] = []
                if len(class_samples[lab_int]) < n_per_class:
                    class_samples[lab_int].append(img.clone())

        # Flatten candidates
        candidates = []
        for lab, imgs in class_samples.items():
            for img in imgs:
                candidates.append((img, lab))

        # Add to buffer using reservoir sampling
        for img, lab in candidates:
            self._count += 1
            if len(self.images) < self.max_size:
                self.images.append(img)
                self.labels.append(lab)
            else:
                # Reservoir sampling: replace a random existing sample
                j = random.randint(0, self._count - 1)
                if j < self.max_size:
                    self.images[j] = img
                    self.labels[j] = lab

    def get_combined_loader(self, new_task_loader, batch_size=32):
        """
        Combine buffer contents with a new task's data into a single
        shuffled DataLoader.

        Args:
            new_task_loader : DataLoader for the new semester
            batch_size      : int  batch size for the combined loader

        Returns:
            DataLoader with interleaved replay + new samples
        """
        if len(self.images) == 0:
            return new_task_loader

        # Wrap buffer as a dataset that returns (tensor, int) to match
        # WriterDataset's output format (avoids collation type mismatch)
        buf_dataset = _BufferDataset(self.images, self.labels)

        # Combine with new task dataset
        combined = ConcatDataset([new_task_loader.dataset, buf_dataset])
        return DataLoader(combined, batch_size=batch_size,
                          shuffle=True, num_workers=0)

    def get_stats(self):
        """Return buffer statistics as a dict."""
        if not self.labels:
            return {"size": 0, "classes": 0}
        unique = set(self.labels)
        return {
            "size": len(self.images),
            "classes": len(unique),
            "samples_per_class": len(self.images) / max(len(unique), 1),
        }


# ================================================================
# Elastic Weight Consolidation (Phase 3)
# ================================================================
class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    After each task/semester, the Fisher Information Matrix is computed
    to estimate which parameters were important for that task.  During
    subsequent training, an EWC penalty discourages large changes to
    those important parameters.

    Penalty = (lambda / 2) * sum_i  F_i * (theta_i - theta*_i)^2

    Usage:
        ewc = EWC(model, lambda_=1000)
        # After training semester 1:
        ewc.consolidate(train_loader, active_classes)
        # During semester 2 training:
        ce_loss = criterion(out, labels)
        total_loss = ce_loss + ewc.penalty(model)
    """

    def __init__(self, model, lambda_=1000):
        """
        Args:
            model   : WriterClassifier — the model being trained
            lambda_ : float — EWC penalty weight (higher = more protection)
        """
        self.lambda_ = lambda_
        self.tasks = []  # list of (fisher_dict, params_dict) per task

    def compute_fisher(self, model, data_loader, active_classes=None,
                       num_samples=None):
        """
        Compute the diagonal Fisher Information Matrix for the current
        task using the empirical Fisher approximation.

        F_i = E[ (d log p(y|x; theta) / d theta_i)^2 ]

        Approximated by the average squared gradient of the log-likelihood
        over training samples.

        Args:
            model          : WriterClassifier
            data_loader    : DataLoader for the current semester
            active_classes : list of int — class indices active this sem
            num_samples    : int or None — max samples to use (None = all)

        Returns:
            fisher : dict  {param_name: tensor of same shape}
        """
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        model.eval()
        criterion = nn.CrossEntropyLoss()
        n_used = 0

        for images, labels in data_loader:
            if num_samples is not None and n_used >= num_samples:
                break

            model.zero_grad()
            out = model(images)

            # Mask to active classes (same as training)
            if active_classes is not None:
                mask = torch.full_like(out, float('-inf'))
                mask[:, active_classes] = out[:, active_classes]
                out = mask

            loss = criterion(out, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * labels.size(0)

            n_used += labels.size(0)

        # Average
        for name in fisher:
            fisher[name] /= max(n_used, 1)

        return fisher

    def consolidate(self, model, data_loader, active_classes=None,
                    num_samples=None):
        """
        Snapshot current model parameters and compute Fisher matrix.
        Call this AFTER finishing training on a semester.

        Args:
            model          : WriterClassifier (just finished training)
            data_loader    : training DataLoader for the semester
            active_classes : list of int — class indices for this semester
            num_samples    : int or None — max samples for Fisher calc
        """
        fisher = self.compute_fisher(model, data_loader, active_classes,
                                     num_samples)

        # Deep-copy current parameters
        params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone()

        self.tasks.append((fisher, params))

    def penalty(self, model):
        """
        Compute the EWC penalty across all consolidated tasks.

        penalty = (lambda / 2) * sum_tasks sum_params F_i * (theta - theta*)^2

        Args:
            model : WriterClassifier (current parameters)

        Returns:
            loss : scalar tensor
        """
        if not self.tasks:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for fisher, old_params in self.tasks:
            for name, param in model.named_parameters():
                if param.requires_grad and name in fisher:
                    loss += (fisher[name] * (param - old_params[name]).pow(2)).sum()

        return (self.lambda_ / 2.0) * loss

    def num_tasks(self):
        """Number of consolidated tasks."""
        return len(self.tasks)


# ================================================================
# Quick self-test
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ScriptSentry — Model Validation")
    print("=" * 60)

    model = WriterClassifier(num_writers=100)
    x = torch.randn(4, 3, 128, 128)

    logits = model(x)
    feats = model.get_features(x)

    print(f"  Input shape   : {x.shape}")
    print(f"  Logits shape  : {logits.shape}")
    print(f"  Feature shape : {feats.shape}")
    print(f"  Total params  : {model.count_total():,}")
    print(f"  Trainable     : {model.count_trainable():,}")
    print(f"  Frozen        : {model.count_total() - model.count_trainable():,}")
    print("\n  ✓ Model OK!\n")
