import json
import re
from datetime import date
from typing import Dict, List

import numpy as np
from PIL import Image

from model_utils import extract_feature_vector, load_enrolled_writers


def _safe_writer_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower() or "writer_unknown"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(np.dot(a, b))


def _centroid(vectors: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack(vectors, axis=0)
    return _l2_normalize(stacked.mean(axis=0))


def _save_enrolled_writers(json_path: str, payload: Dict) -> None:
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _parse_image(uploaded_file) -> Image.Image:
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext not in {"jpg", "jpeg", "png"}:
        raise ValueError("Please upload JPG or PNG only")
    return Image.open(uploaded_file).convert("L")


def enroll_writer(
    writer_name: str,
    uploaded_files,
    model,
    device,
    json_path: str,
    task_id: int,
) -> Dict:
    if not writer_name.strip():
        raise ValueError("Please enter a writer name or ID")

    if uploaded_files is None or len(uploaded_files) < 3 or len(uploaded_files) > 5:
        raise ValueError("Upload 3 to 5 handwriting images")

    images = [_parse_image(f) for f in uploaded_files]
    features = [extract_feature_vector(model, img, device) for img in images]

    store = load_enrolled_writers(json_path)
    writers = store.setdefault("writers", {})

    writer_id = _safe_writer_id(writer_name)
    new_centroid = _centroid(features)

    existing_sims = []
    for wid, meta in writers.items():
        vecs = meta.get("feature_vectors", [])
        if not vecs:
            continue
        existing_centroid = _centroid([np.array(v, dtype=np.float32) for v in vecs])
        sim = _cosine(new_centroid, existing_centroid)
        existing_sims.append((wid, meta.get("name", wid), sim))

    existing_sims = sorted(existing_sims, key=lambda x: x[2], reverse=True)
    nearest_existing = existing_sims[0][2] if existing_sims else 0.0

    intra = float(np.mean([_cosine(new_centroid, f) for f in features]))
    intra_01 = (intra + 1.0) / 2.0
    sep_01 = (1.0 - nearest_existing) / 2.0
    confidence = max(0.0, min(100.0, (0.7 * intra_01 + 0.3 * sep_01) * 100.0))

    writers[writer_id] = {
        "name": writer_name.strip(),
        "feature_vectors": [f.tolist() for f in features],
        "enrolled_date": str(date.today()),
        "task_id": int(task_id),
        "sample_count": len(features),
    }

    store["total_enrolled"] = len(writers)
    store["last_updated"] = str(date.today())
    _save_enrolled_writers(json_path, store)

    similar_writers = [
        {"writer_id": wid, "name": nm, "similarity": float(sim)}
        for wid, nm, sim in existing_sims[:5]
    ]

    # Immediate enrollment test in feature space.
    test_feature = features[0]
    best_id = None
    best_name = None
    best_sim = -1.0
    for wid, meta in writers.items():
        vecs = meta.get("feature_vectors", [])
        if not vecs:
            continue
        cent = _centroid([np.array(v, dtype=np.float32) for v in vecs])
        sim = _cosine(test_feature, cent)
        if sim > best_sim:
            best_sim = sim
            best_id = wid
            best_name = meta.get("name", wid)

    enrollment_working = best_id == writer_id

    return {
        "writer_id": writer_id,
        "writer_name": writer_name.strip(),
        "sample_count": len(features),
        "confidence": confidence,
        "similar_writers": similar_writers,
        "predicted_writer_id": best_id,
        "predicted_writer_name": best_name,
        "predicted_similarity": best_sim,
        "enrollment_working": enrollment_working,
    }
