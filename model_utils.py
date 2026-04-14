import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image
from torchvision import models, transforms

from models import WriterClassifier


TOTAL_WRITERS_DEFAULT = 150


class PadToSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        fill = 255 if img.mode in ("L", "1") else (255, 255, 255)
        out = Image.new(img.mode, (side, side), fill)
        out.paste(img, ((side - w) // 2, (side - h) // 2))
        return out


class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.sigma = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.sigma * F.linear(x_norm, w_norm)


class WriterClassifierV4(nn.Module):
    """Mirror of kaggle_ultimate_v4.py architecture for inference/debug."""

    def __init__(self, num_writers: int = TOTAL_WRITERS_DEFAULT):
        super().__init__()
        self.num_writers = num_writers
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

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(x)
        return self.projector(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_features(x)
        return self.classifier(feat)


@dataclass
class ModelBundle:
    model: nn.Module
    checkpoint: Dict
    checkpoint_path: str
    model_type: str
    device: torch.device
    num_writers: int
    tasks_learned: int
    task_accuracies: List[float]
    overall_avg_accuracy: float
    model_file_size_mb: float


def get_test_transform_v4() -> transforms.Compose:
    """Exact test transform from kaggle_ultimate_v4.py."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def find_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        return None

    v4_final = ckpt_dir / "v4_final.pt"
    if v4_final.exists():
        return str(v4_final)

    v4_sem = sorted(ckpt_dir.glob("v4_after_sem*.pt"))
    if v4_sem:
        return str(v4_sem[-1])

    all_pt = sorted(ckpt_dir.glob("*.pt"))
    return str(all_pt[-1]) if all_pt else None


def _detect_v4_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return (
        "projector.0.weight" in state_dict
        or "classifier.sigma" in state_dict
        or any(k.startswith("backbone.layer1.0.conv3") for k in state_dict)
    )


def _build_model_for_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Tuple[nn.Module, str, int]:
    if _detect_v4_from_state_dict(state_dict):
        num_writers = int(state_dict["classifier.weight"].shape[0])
        return WriterClassifierV4(num_writers=num_writers), "WriterClassifierV4", num_writers

    num_writers = TOTAL_WRITERS_DEFAULT
    if "classifier.3.weight" in state_dict:
        num_writers = int(state_dict["classifier.3.weight"].shape[0])
    return WriterClassifier(num_writers=num_writers), "WriterClassifier", num_writers


def _pick_accuracy_matrix(ckpt: Dict) -> Optional[np.ndarray]:
    for key in ("best_matrix", "nme_matrix", "fc_matrix", "acc_matrix"):
        if key in ckpt and ckpt[key] is not None:
            mat = np.array(ckpt[key], dtype=float)
            if mat.ndim == 2:
                return mat
    return None


def _compute_task_metrics(ckpt: Dict, matrix: Optional[np.ndarray]) -> Tuple[int, List[float], float]:
    semester = int(ckpt.get("semester", 0) or 0)

    if matrix is None:
        tasks = max(semester, 0)
        return tasks, [], 0.0

    nrows, ncols = matrix.shape
    if semester <= 0:
        # Fallback when semester is absent (for final checkpoints).
        semester = int(np.count_nonzero(np.any(matrix > 0, axis=1)))
        if semester <= 0:
            semester = nrows

    semester = min(max(semester, 1), nrows)
    task_count = min(semester, ncols)

    row = matrix[semester - 1]
    task_accuracies = [float(v) for v in row[:task_count]]
    avg = float(np.mean(task_accuracies)) if task_accuracies else 0.0
    return task_count, task_accuracies, avg


def load_model_bundle(checkpoint_path: str, device: str = "cpu") -> ModelBundle:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Model file not found. Place your model file in the same folder as app.py"
        )

    torch_device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    if "model_state_dict" not in ckpt:
        raise RuntimeError("Invalid checkpoint: missing model_state_dict")

    state_dict = ckpt["model_state_dict"]

    try:
        model, model_type, num_writers = _build_model_for_checkpoint(state_dict)
        model = model.to(torch_device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Wrong model architecture. Check kaggle_ultimate_v4.py\n{exc}") from exc

    matrix = _pick_accuracy_matrix(ckpt)
    task_count, task_acc, avg_acc = _compute_task_metrics(ckpt, matrix)
    file_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    return ModelBundle(
        model=model,
        checkpoint=ckpt,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=torch_device,
        num_writers=num_writers,
        tasks_learned=task_count,
        task_accuracies=task_acc,
        overall_avg_accuracy=avg_acc,
        model_file_size_mb=file_mb,
    )


@torch.no_grad()
def run_inference_topk(
    model: nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    top_k: int = 5,
    seen_classes: Optional[List[int]] = None,
) -> Tuple[List[int], List[float]]:
    transform = get_test_transform_v4()
    x = transform(pil_image.convert("L")).unsqueeze(0).to(device)

    logits = model(x)
    if seen_classes is not None:
        mask = torch.full_like(logits, float("-inf"))
        mask[:, seen_classes] = logits[:, seen_classes]
        logits = mask

    probs = F.softmax(logits, dim=1)
    vals, idxs = probs.topk(top_k, dim=1)

    indices = [int(i) for i in idxs[0].tolist()]
    confidences = [float(v) for v in vals[0].tolist()]
    return indices, confidences


@torch.no_grad()
def extract_feature_vector(model: nn.Module, pil_image: Image.Image, device: torch.device) -> np.ndarray:
    transform = get_test_transform_v4()
    x = transform(pil_image.convert("L")).unsqueeze(0).to(device)

    if hasattr(model, "get_features"):
        feat = model.get_features(x)
    else:
        feat = model.backbone(x)

    feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_enrolled_writers(json_path: str) -> Dict:
    if not os.path.exists(json_path):
        return {
            "writers": {},
            "total_enrolled": 0,
            "last_updated": str(date.today()),
        }

    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(np.dot(a, b))


def _feature_centroid(vectors: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack(vectors, axis=0)
    return _l2_normalize(stacked.mean(axis=0))


def predict_topk_enrolled_writers(
    model: nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    enrolled_store: Dict,
    top_k: int = 5,
) -> List[Dict[str, float]]:
    """Predict top-k among enrolled writers using cosine similarity in feature space."""
    writers = enrolled_store.get("writers", {})
    if not writers:
        return []

    feat = extract_feature_vector(model, pil_image, device)
    candidates = []
    for writer_id, meta in writers.items():
        vecs = meta.get("feature_vectors", [])
        if not vecs:
            continue
        centroid = _feature_centroid([np.array(v, dtype=np.float32) for v in vecs])
        sim = _cosine_similarity(feat, centroid)
        candidates.append({
            "writer_id": writer_id,
            "name": meta.get("name", writer_id),
            "score": sim,
        })

    if not candidates:
        return []

    raw = np.array([c["score"] for c in candidates], dtype=np.float32)
    # Softmax over cosine scores for a confidence-like percentage distribution.
    scaled = raw * 10.0
    exps = np.exp(scaled - np.max(scaled))
    probs = exps / (np.sum(exps) + 1e-8)

    for i, p in enumerate(probs.tolist()):
        candidates[i]["confidence"] = float(p)

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:top_k]


def _resolve_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Use the exact last conv path for v4 (backbone.layer4[-1].conv3)."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        last_block = model.backbone.layer4[-1]
        if hasattr(last_block, "conv3"):
            return last_block.conv3
        if hasattr(last_block, "conv2"):
            return last_block.conv2

    raise RuntimeError("Could not resolve GradCAM target layer from model structure")


@torch.no_grad()
def _prepare_input_tensor(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = get_test_transform_v4()
    return transform(pil_image.convert("L")).unsqueeze(0).to(device)


def generate_gradcam_overlay(
    model: nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    class_idx: Optional[int] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Generate GradCAM overlay as RGB numpy array."""
    model.eval()
    target_layer = _resolve_gradcam_target_layer(model)

    activations = {}
    gradients = {}

    def _fwd_hook(_module, _inp, out):
        activations["value"] = out.detach()

    def _bwd_hook(_module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_fwd = target_layer.register_forward_hook(_fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        x = _prepare_input_tensor(pil_image, device)
        logits = model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        acts = activations["value"]
        grads = gradients["value"]

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        orig = pil_image.convert("RGB").resize((224, 224))
        orig_np = np.array(orig).astype(np.float32) / 255.0

        heatmap = cm.get_cmap("RdBu_r")(cam)[..., :3]
        overlay = (1.0 - alpha) * orig_np + alpha * heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        return (overlay * 255.0).astype(np.uint8)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
