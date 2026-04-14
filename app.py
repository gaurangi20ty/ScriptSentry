import io
import importlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from model_utils import (
    find_latest_checkpoint,
    load_enrolled_writers,
    load_model_bundle,
    predict_topk_enrolled_writers,
)
from enrollment import enroll_writer

try:
    pdfium = importlib.import_module("pypdfium2")
    PDFIUM_AVAILABLE = True
except Exception:
    PDFIUM_AVAILABLE = False


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
ENROLLED_JSON = PROJECT_DIR / "enrolled_writers.json"
CASES_JSON = PROJECT_DIR / "results" / "phase1_cases.json"
AUDIT_JSON = PROJECT_DIR / "results" / "phase1_audit_log.json"
POLICIES_JSON = PROJECT_DIR / "results" / "phase2_policies.json"
DEPARTMENT_DATA_DIR = PROJECT_DIR / "results" / "department_data"

DEPARTMENTS = [
    "Education",
    "Banking",
    "Property and Land Records",
    "General Use",
]

DEFAULT_POLICIES = {
    "Education": {"match_threshold": 78.0, "review_threshold": 62.0},
    "Banking": {"match_threshold": 84.0, "review_threshold": 68.0},
    "Property and Land Records": {"match_threshold": 82.0, "review_threshold": 66.0},
    "General Use": {"match_threshold": 75.0, "review_threshold": 60.0},
}

DOMAIN_BEHAVIORS = {
    "Education": {
        "mismatch_penalty": 20.0,
        "review_penalty": 8.0,
        "rescan_penalty": 18.0,
        "actions": {
            "MATCH": "Auto-pass and archive with exam metadata",
            "NEEDS_REVIEW": "Route to academic integrity reviewer",
            "MISMATCH": "Raise proxy-writing alert for manual verification",
            "NEEDS_RESCAN": "Request clearer answer-sheet scan",
        },
    },
    "Banking": {
        "mismatch_penalty": 28.0,
        "review_penalty": 12.0,
        "rescan_penalty": 22.0,
        "actions": {
            "MATCH": "Proceed with controlled verification approval",
            "NEEDS_REVIEW": "Escalate to compliance desk for second check",
            "MISMATCH": "Hold transaction and trigger fraud workflow",
            "NEEDS_RESCAN": "Request high-quality document resubmission",
        },
    },
    "Property and Land Records": {
        "mismatch_penalty": 24.0,
        "review_penalty": 10.0,
        "rescan_penalty": 20.0,
        "actions": {
            "MATCH": "Proceed with deed processing",
            "NEEDS_REVIEW": "Queue for registry officer review",
            "MISMATCH": "Flag potential deed/signatory anomaly",
            "NEEDS_RESCAN": "Request clearer deed or registry scan",
        },
    },
    "General Use": {
        "mismatch_penalty": 20.0,
        "review_penalty": 8.0,
        "rescan_penalty": 18.0,
        "actions": {
            "MATCH": "Auto-pass",
            "NEEDS_REVIEW": "Send to manual review queue",
            "MISMATCH": "High-risk alert and manual verification",
            "NEEDS_RESCAN": "Upload a clearer scan",
        },
    },
}


def _dept_slug(department: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", department.lower()).strip("_")
    return slug or "general_use"


def _active_department() -> str:
    return st.session_state.get("selected_department") or "General Use"


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_iso_utc(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _department_dir(department: str | None = None) -> Path:
    dept = department or _active_department()
    return DEPARTMENT_DATA_DIR / _dept_slug(dept)


def _enrolled_json_path(department: str | None = None) -> Path:
    return _department_dir(department) / "enrolled_writers.json"


def _cases_json_path(department: str | None = None) -> Path:
    return _department_dir(department) / "cases.json"


def _audit_json_path(department: str | None = None) -> Path:
    return _department_dir(department) / "audit_log.json"


def _inject_modern_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Keyframes ─────────────────────────────────── */
        @keyframes ss-gradient-shift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes ss-pulse-green {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.45); }
            50%      { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        }
        @keyframes ss-fade-in {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes ss-shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes ss-float {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-3px); }
        }

        /* ── Base ───────────────────────────────────────── */
        .stApp {
            background: linear-gradient(135deg, #f0f4ff 0%, #f8fafc 30%, #f0fdf4 60%, #faf5ff 100%);
            color: #0f172a;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            padding-left: 1.8rem;
            padding-right: 1.8rem;
            max-width: 1280px;
        }

        /* ── Typography ─────────────────────────────────── */
        h1 {
            font-family: 'Poppins', sans-serif !important;
            color: #0f172a !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
            font-size: 2rem !important;
        }
        h2 {
            font-family: 'Poppins', sans-serif !important;
            color: #1e293b !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        h3 {
            font-family: 'Poppins', sans-serif !important;
            color: #1e293b !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em;
        }
        h4 {
            font-family: 'Inter', sans-serif !important;
            color: #334155 !important;
            font-weight: 600 !important;
        }

        p, label, span, div {
            color: #334155;
        }

        /* ── Scrollbar ──────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

        /* ── Sidebar ────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(175deg, #0c1426 0%, #162752 35%, #1a3a6e 60%, #0e5e56 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at 30% 10%, rgba(99,102,241,0.12) 0%, transparent 60%),
                        radial-gradient(ellipse at 70% 90%, rgba(16,185,129,0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
            color: #ffffff !important;
        }

        /* ── Animated Header Card ───────────────────────── */
        .ss-header-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 25%, #0891b2 50%, #0f766e 75%, #1e3a8a 100%);
            background-size: 300% 300%;
            animation: ss-gradient-shift 8s ease infinite;
            border-radius: 20px;
            padding: 28px 32px;
            color: white;
            box-shadow: 0 10px 40px rgba(30, 58, 138, 0.22), 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }
        .ss-header-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .ss-header-card::after {
            content: '';
            position: absolute;
            bottom: -30%;
            left: -10%;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
            pointer-events: none;
        }
        .ss-header-card * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            position: relative;
            z-index: 1;
        }

        /* ── Glassmorphism Cards ─────────────────────────── */
        .ss-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-left: 4px solid;
            border-image: linear-gradient(180deg, #3b82f6, #0f766e) 1;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
            margin-bottom: 16px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            animation: ss-fade-in 0.4s ease-out;
        }
        .ss-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 28px rgba(59, 130, 246, 0.1), 0 4px 12px rgba(0, 0, 0, 0.06);
            border-color: rgba(199, 210, 254, 0.9);
        }

        /* ── Status / Decision Badges ────────────────────── */
        .ss-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .ss-badge-match { background: linear-gradient(135deg, #dcfce7, #bbf7d0); color: #166534 !important; border: 1px solid #86efac; }
        .ss-badge-mismatch { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b !important; border: 1px solid #fca5a5; }
        .ss-badge-review { background: linear-gradient(135deg, #fef9c3, #fde68a); color: #92400e !important; border: 1px solid #fcd34d; }
        .ss-badge-rescan { background: linear-gradient(135deg, #dbeafe, #bfdbfe); color: #1e40af !important; border: 1px solid #93c5fd; }
        .ss-badge-online { background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46 !important; border: 1px solid #6ee7b7; }
        .ss-badge-offline { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b !important; border: 1px solid #fca5a5; }

        .ss-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
            border: 1px solid #c7d2fe;
            color: #3730a3 !important;
            border-radius: 999px;
            padding: 6px 16px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        /* ── Priority Badges ──────────────────────────── */
        .ss-priority-critical { background: #fee2e2; color: #991b1b !important; border: 1px solid #fca5a5; border-radius: 999px; padding: 3px 10px; font-weight: 700; font-size: 0.75rem; }
        .ss-priority-high { background: #ffedd5; color: #9a3412 !important; border: 1px solid #fdba74; border-radius: 999px; padding: 3px 10px; font-weight: 700; font-size: 0.75rem; }
        .ss-priority-medium { background: #dbeafe; color: #1e40af !important; border: 1px solid #93c5fd; border-radius: 999px; padding: 3px 10px; font-weight: 700; font-size: 0.75rem; }
        .ss-priority-low { background: #f1f5f9; color: #475569 !important; border: 1px solid #cbd5e1; border-radius: 999px; padding: 3px 10px; font-weight: 700; font-size: 0.75rem; }

        /* ── Empty State ─────────────────────────────────── */
        .ss-empty {
            text-align: center;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            border: 2px dashed #c7d2fe;
            border-radius: 20px;
            padding: 40px 24px;
            color: #475569;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
            animation: ss-fade-in 0.5s ease-out;
        }
        .ss-empty h4 {
            color: #1e293b !important;
            margin-bottom: 8px;
        }

        /* ── Dividers ────────────────────────────────────── */
        hr {
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, #cbd5e1, transparent);
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }

        /* ── Metrics ─────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(226, 232, 240, 0.7);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
            transition: all 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.08);
            transform: translateY(-1px);
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 800 !important;
            font-family: 'Poppins', sans-serif !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* ── Alerts ──────────────────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid rgba(226, 232, 240, 0.6);
            backdrop-filter: blur(8px);
        }
        [data-testid="stAlert"] p {
            color: #1e293b !important;
        }

        /* ── File Uploader ───────────────────────────────── */
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, rgba(248,250,252,0.9) 0%, rgba(241,245,249,0.9) 100%) !important;
            border: 2px dashed #94a3b8 !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3b82f6 !important;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1) !important;
        }
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] label,
        [data-testid="stFileUploaderDropzone"] div {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
        }

        /* ── Inputs ──────────────────────────────────────── */
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.2s ease !important;
        }
        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12), 0 2px 6px rgba(0,0,0,0.04) !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            caret-color: #3b82f6 !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-baseweb="select"] * {
            color: #0f172a !important;
        }

        /* ── Select Popover ──────────────────────────────── */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] [role="listbox"],
        div[role="listbox"] {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
        }
        [data-baseweb="popover"] [role="option"],
        div[role="option"] {
            background: #ffffff !important;
            color: #111827 !important;
            transition: background 0.15s ease !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        div[role="option"]:hover {
            background: #f0f4ff !important;
            color: #111827 !important;
        }
        [data-baseweb="popover"] [role="option"][aria-selected="true"],
        div[role="option"][aria-selected="true"] {
            background: #eef2ff !important;
            color: #111827 !important;
            font-weight: 600 !important;
        }
        [data-baseweb="popover"] [role="option"] *,
        div[role="option"] * {
            color: #111827 !important;
        }

        /* ── Sidebar Inputs ──────────────────────────────── */
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="textarea"] > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            caret-color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="popover"] [role="option"],
        [data-testid="stSidebar"] [data-baseweb="popover"] [role="option"] * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        /* ── Buttons ─────────────────────────────────────── */
        [data-testid="stButton"] > button {
            border-radius: 12px;
            border: 1.5px solid #e2e8f0;
            background: rgba(255, 255, 255, 0.9);
            color: #1e293b;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            padding: 0.55rem 1rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.01em;
        }
        [data-testid="stButton"] > button:hover {
            border-color: #93c5fd;
            background: #f0f4ff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12);
        }

        /* Primary buttons */
        [data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
            font-weight: 700;
        }
        [data-testid="stButton"] > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
            transform: translateY(-2px);
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #10b981, #0f766e) !important;
            color: #ffffff !important;
            border: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0f766e, #065f46) !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"],
        [data-testid="stSidebar"] [data-testid="stButton"] > button:not([kind="primary"]) {
            background: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        /* ── Placeholders ────────────────────────────────── */
        input::placeholder, textarea::placeholder,
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="textarea"] textarea::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder,
        [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder {
            color: rgba(255, 255, 255, 0.5) !important;
            opacity: 1 !important;
        }

        /* ── Checkboxes ──────────────────────────────────── */
        [data-testid="stCheckbox"] [role="checkbox"],
        [data-testid="stCheckbox"] [role="checkbox"] svg,
        [data-testid="stCheckbox"] [data-baseweb="checkbox"],
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] * {
            color: #3b82f6 !important;
            fill: #3b82f6 !important;
            stroke: #3b82f6 !important;
            border-color: #cbd5e1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] [role="checkbox"],
        [data-testid="stSidebar"] [data-testid="stCheckbox"] [role="checkbox"] svg,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] [data-baseweb="checkbox"],
        [data-testid="stSidebar"] [data-testid="stCheckbox"] [data-baseweb="checkbox"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
            border-color: #ffffff !important;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label * {
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        /* ── Data Tables ─────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        }

        /* ── Tabs ────────────────────────────────────────── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 2px;
            background: rgba(241, 245, 249, 0.8);
            border-radius: 12px;
            padding: 4px;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 8px 16px;
            transition: all 0.2s ease;
            color: #64748b !important;
            background: transparent;
        }
        [data-testid="stTabs"] [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.7);
            color: #1e293b !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff !important;
            color: #2563eb !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: transparent !important;
        }

        /* ── Custom Components ───────────────────────────── */
        .ss-domain-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin: 20px 0;
        }
        .ss-domain-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border: 1.5px solid #e2e8f0;
            border-radius: 18px;
            padding: 24px 22px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .ss-domain-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            border-radius: 18px 18px 0 0;
        }
        .ss-domain-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(59, 130, 246, 0.12), 0 4px 12px rgba(0, 0, 0, 0.04);
            border-color: #93c5fd;
        }
        .ss-domain-card .ss-icon {
            font-size: 2.2rem;
            margin-bottom: 10px;
            display: block;
        }
        .ss-domain-card .ss-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: #1e293b;
            margin-bottom: 6px;
        }
        .ss-domain-card .ss-desc {
            font-size: 0.82rem;
            color: #64748b;
            line-height: 1.4;
        }
        .ss-domain-card-edu::before { background: linear-gradient(90deg, #3b82f6, #6366f1); }
        .ss-domain-card-bank::before { background: linear-gradient(90deg, #10b981, #0891b2); }
        .ss-domain-card-prop::before { background: linear-gradient(90deg, #f59e0b, #ef4444); }
        .ss-domain-card-gen::before { background: linear-gradient(90deg, #8b5cf6, #ec4899); }

        /* ── Feature Pills ───────────────────────────────── */
        .ss-features-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 16px 0;
            justify-content: center;
        }
        .ss-feature-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.78rem;
            font-weight: 500;
            color: #475569;
            backdrop-filter: blur(4px);
        }

        /* ── Animated Confidence Bar ─────────────────────── */
        .ss-conf-bar-wrap {
            background: #f1f5f9;
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
            margin: 4px 0 12px 0;
        }
        .ss-conf-bar {
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #3b82f6, #0891b2);
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ── Sidebar Brand ───────────────────────────────── */
        .ss-sidebar-brand {
            text-align: center;
            padding: 8px 0 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 12px;
        }
        .ss-sidebar-brand .ss-logo-text {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 1.4rem;
            background: linear-gradient(135deg, #ffffff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent !important;
            letter-spacing: -0.02em;
        }
        .ss-sidebar-brand .ss-version {
            font-size: 0.7rem;
            opacity: 0.6;
            letter-spacing: 0.05em;
        }

        /* ── Status Dot ──────────────────────────────────── */
        .ss-status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .ss-status-dot-online {
            background: #10b981;
            animation: ss-pulse-green 2s infinite;
        }
        .ss-status-dot-offline {
            background: #ef4444;
        }

        /* ── Styled Expander ─────────────────────────────── */
        [data-testid="stExpander"] {
            border: 1px solid rgba(226, 232, 240, 0.6);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(6px);
        }

        /* ── Progress bars ───────────────────────────────── */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #3b82f6, #0891b2) !important;
            border-radius: 8px;
        }

        /* ── Hero Section ────────────────────────────────── */
        .ss-hero {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 25%, #0891b2 50%, #0f766e 75%, #1e3a8a 100%);
            background-size: 300% 300%;
            animation: ss-gradient-shift 8s ease infinite;
            border-radius: 24px;
            padding: 40px 36px;
            color: white;
            text-align: center;
            box-shadow: 0 12px 48px rgba(30, 58, 138, 0.2), 0 4px 12px rgba(0,0,0,0.06);
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .ss-hero::before {
            content: '';
            position: absolute;
            top: -40%;
            right: -15%;
            width: 350px;
            height: 350px;
            background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
            pointer-events: none;
        }
        .ss-hero * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            position: relative;
            z-index: 1;
        }
        .ss-hero .ss-hero-subtitle {
            opacity: 0.85;
            font-size: 0.95rem;
            margin-top: 8px;
        }
        .ss-hero .ss-hero-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 2.4rem;
            letter-spacing: -0.03em;
            margin: 6px 0;
        }

        /* ── Prediction Card ─────────────────────────────── */
        .ss-pred-card {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 10px 16px;
            margin-bottom: 8px;
            transition: all 0.2s ease;
        }
        .ss-pred-card:hover {
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        .ss-pred-rank {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 1.1rem;
            color: #3b82f6;
            min-width: 24px;
        }
        .ss-pred-name {
            flex: 1;
            font-weight: 600;
            color: #1e293b;
        }
        .ss-pred-conf {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #0f766e;
            font-size: 0.9rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def _reset_uploaded_artifact() -> None:
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_filename = ""
    st.session_state.quality_result = None
    st.session_state.upload_key_counter += 1


def _reset_single_enrollment_state() -> None:
    st.session_state.enroll_writer_name = ""
    st.session_state.enroll_writer_name__seed = ""
    st.session_state.enroll_writer_uploader_counter += 1


def _reset_bulk_enrollment_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("bulk_name_"):
            st.session_state.pop(key, None)
    st.session_state.bulk_groups = []
    st.session_state.bulk_uploader_counter += 1


def _pil_to_named_bytesio(img: Image.Image, name: str):
    buff = io.BytesIO()
    img.convert("L").save(buff, format="PNG")
    buff.seek(0)
    buff.name = name
    return buff


def _images_from_pdf_bytes(pdf_bytes: bytes, source_name: str) -> List[Image.Image]:
    if not PDFIUM_AVAILABLE:
        raise ValueError("PDF support requires pypdfium2. Install it from requirements and restart.")
    out: List[Image.Image] = []
    pdf = pdfium.PdfDocument(pdf_bytes)
    max_pages = min(len(pdf), 5)
    for i in range(max_pages):
        page = pdf.get_page(i)
        bitmap = page.render(scale=2.0)
        pil_img = bitmap.to_pil()
        out.append(pil_img.convert("L"))
        page.close()
    pdf.close()
    if not out:
        raise ValueError(f"No renderable pages found in PDF: {source_name}")
    return out


def _extract_images_from_uploaded_file(uploaded_file) -> List[Image.Image]:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith((".jpg", ".jpeg", ".png")):
        return [Image.open(io.BytesIO(raw)).convert("L")]

    if name.endswith(".pdf"):
        return _images_from_pdf_bytes(raw, uploaded_file.name)

    if name.endswith(".zip"):
        images: List[Image.Image] = []
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            for entry in zf.namelist():
                lower = entry.lower()
                if lower.endswith((".jpg", ".jpeg", ".png")):
                    with zf.open(entry) as fh:
                        images.append(Image.open(io.BytesIO(fh.read())).convert("L"))
                elif lower.endswith(".pdf"):
                    with zf.open(entry) as fh:
                        images.extend(_images_from_pdf_bytes(fh.read(), entry))
                if len(images) >= 10:
                    break
        if not images:
            raise ValueError("ZIP has no supported files. Add JPG/PNG/PDF files.")
        return images

    raise ValueError("Unsupported format. Use JPG, PNG, PDF, or ZIP.")


def _init_session_state() -> None:
    if "bundle" not in st.session_state:
        st.session_state.bundle = None
    if "load_error" not in st.session_state:
        st.session_state.load_error = ""
    if "checkpoint_path" not in st.session_state:
        auto_ckpt = find_latest_checkpoint(str(CHECKPOINTS_DIR))
        st.session_state.checkpoint_path = auto_ckpt or ""
    if "selected_department" not in st.session_state:
        st.session_state.selected_department = ""
    if "app_page" not in st.session_state:
        st.session_state.app_page = "domain"
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "new_case"
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "upload_key_counter" not in st.session_state:
        st.session_state.upload_key_counter = 0
    if "quality_result" not in st.session_state:
        st.session_state.quality_result = None
    if "case_result" not in st.session_state:
        st.session_state.case_result = None
    if "department_policies" not in st.session_state:
        st.session_state.department_policies = DEFAULT_POLICIES.copy()
    if "enroll_writer_uploader_counter" not in st.session_state:
        st.session_state.enroll_writer_uploader_counter = 0
    if "bulk_uploader_counter" not in st.session_state:
        st.session_state.bulk_uploader_counter = 0


def _ensure_persistence_files() -> None:
    DEPARTMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dept in DEPARTMENTS:
        ddir = _department_dir(dept)
        ddir.mkdir(parents=True, exist_ok=True)

        enrolled_path = _enrolled_json_path(dept)
        if not enrolled_path.exists():
            enrolled_path.write_text(
                json.dumps(
                    {
                        "writers": {},
                        "total_enrolled": 0,
                        "last_updated": datetime.utcnow().date().isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        cases_path = _cases_json_path(dept)
        if not cases_path.exists():
            cases_path.write_text("[]", encoding="utf-8")

        audit_path = _audit_json_path(dept)
        if not audit_path.exists():
            audit_path.write_text("[]", encoding="utf-8")

    if not POLICIES_JSON.exists():
        POLICIES_JSON.write_text(json.dumps(DEFAULT_POLICIES, indent=2), encoding="utf-8")


def _read_json_array(path: Path) -> list:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _write_json_array(path: Path, payload: list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_policies() -> dict:
    try:
        payload = json.loads(POLICIES_JSON.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return DEFAULT_POLICIES.copy()


def _save_policies(policies: dict) -> None:
    POLICIES_JSON.write_text(json.dumps(policies, indent=2), encoding="utf-8")


def _valid_writer_name(name: str) -> bool:
    # Allow alphanumeric names with spaces, apostrophes, hyphens, and underscores.
    # Examples: "Gaurangi", "roll 1", "a05_013", "writer-2"
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s'_\-]{0,49}", name.strip()))


def _infer_writer_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"([_-]?\d+)$", "", stem).strip(" _-")
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return "Writer"
    return stem.title()


def _seed_text_input(key: str, value: str) -> None:
    current = st.session_state.get(key, "")
    seed_key = f"{key}__seed"
    previous_seed = st.session_state.get(seed_key, "")
    if not current or current == previous_seed:
        st.session_state[key] = value
        st.session_state[seed_key] = value


def _extract_group_key_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]

    # Remove up to two trailing numeric chunks (e.g., 00 07) so one writer is not split.
    removed = 0
    while tokens and removed < 2 and re.fullmatch(r"\d{1,3}", tokens[-1]):
        tokens.pop()
        removed += 1

    # IAM-style normalization: a05-013-xx-yy -> a05 013
    if len(tokens) >= 2 and re.fullmatch(r"[A-Za-z]\d{2}", tokens[0]) and re.fullmatch(r"\d{3}", tokens[1]):
        tokens = tokens[:2]

    normalized = " ".join(tokens).strip()
    return normalized or stem


def _normalize_writer_folder_name(folder_name: str) -> str:
    base = Path(folder_name).name.strip()
    # Preserve numeric suffixes in folder names (e.g., roll_1, roll_2).
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or folder_name


def _group_zip_samples_by_writer(zip_bytes: bytes) -> list:
    groups: dict[str, dict] = {}
    generic_folder_names = {"images", "image", "files", "docs", "documents", "data", "dataset", "samples"}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            lower = info.filename.lower()
            if not lower.endswith((".jpg", ".jpeg", ".png", ".pdf")):
                continue

            raw = zf.read(info.filename)
            entry_path = Path(info.filename)
            parent_name = entry_path.parent.name.strip() if entry_path.parent else ""

            # Prefer grouping by folder; fallback to filename when folder looks generic.
            if parent_name and parent_name.lower() not in generic_folder_names:
                inferred_name = _normalize_writer_folder_name(parent_name)
            else:
                inferred_name = _extract_group_key_from_filename(entry_path.name)

            group_key = inferred_name.lower()
            group = groups.setdefault(
                group_key,
                {
                    "default_name": inferred_name,
                    "files": [],
                },
            )

            if lower.endswith(".pdf"):
                images = _images_from_pdf_bytes(raw, info.filename)
                for page_index, page_img in enumerate(images, start=1):
                    buff = io.BytesIO()
                    page_img.convert("L").save(buff, format="PNG")
                    group["files"].append(
                        {
                            "filename": f"{Path(info.filename).stem}_page{page_index}.png",
                            "bytes": buff.getvalue(),
                        }
                    )
            else:
                group["files"].append({"filename": info.filename, "bytes": raw})

    grouped_list = []
    for idx, (group_key, group) in enumerate(sorted(groups.items(), key=lambda item: item[1]["default_name"].lower())):
        grouped_list.append(
            {
                "id": idx,
                "group_key": group_key,
                "default_name": group["default_name"],
                "files": group["files"],
            }
        )
    return grouped_list


def _delete_writer_by_id(writer_id: str) -> bool:
    enrolled_path = _enrolled_json_path()
    try:
        payload = json.loads(enrolled_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    writers = payload.get("writers", {})
    if writer_id not in writers:
        return False

    del writers[writer_id]
    payload["writers"] = writers
    payload["total_enrolled"] = len(writers)
    payload["last_updated"] = datetime.utcnow().date().isoformat()
    enrolled_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _append_audit(event_type: str, details: dict) -> None:
    logs = _read_json_array(_audit_json_path())
    logs.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event_type": event_type,
            "department": st.session_state.get("selected_department") or "General Use",
            "details": details,
        }
    )
    _write_json_array(_audit_json_path(), logs)


def _read_active_department_cases() -> list:
    selected = st.session_state.get("selected_department")
    cases = _read_json_array(_cases_json_path())
    if not selected:
        return cases
    return [c for c in cases if c.get("department") == selected]


def _next_case_id(existing_cases: list) -> str:
    return f"CASE-{len(existing_cases) + 1:05d}"


def _writer_display_maps(writers: dict) -> tuple[list, dict]:
    writer_items = sorted(
        [(wid, meta.get("name", wid)) for wid, meta in writers.items()],
        key=lambda x: str(x[1]).lower(),
    )
    labels: list[str] = []
    label_to_id: dict[str, str] = {}
    for wid, name in writer_items:
        display_name = str(name).strip() if str(name).strip() else wid
        if display_name in label_to_id:
            display_name = f"{display_name} [{wid}]"
        labels.append(display_name)
        label_to_id[display_name] = wid
    return labels, label_to_id


def _compute_quality_metrics(pil_image: Image.Image) -> dict:
    gray = np.asarray(pil_image.convert("L"), dtype=np.float32)
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    focus = float(np.mean(np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)))

    quality_score = min(100.0, max(0.0, contrast * 1.2 + focus * 0.8))
    if quality_score >= 55.0:
        quality_status = "PASS"
    elif quality_score >= 35.0:
        quality_status = "REVIEW"
    else:
        quality_status = "FAIL"

    return {
        "contrast": contrast,
        "brightness": brightness,
        "focus": focus,
        "quality_score": quality_score,
        "quality_status": quality_status,
    }


def _decision_from_prediction(
    claimed_writer_id: str,
    top_preds: list,
    quality_status: str,
    policy: dict,
    department: str,
    metadata: dict,
) -> dict:
    top1 = top_preds[0]
    top1_id = top1["writer_id"]
    top1_conf = float(top1["confidence"] * 100.0)
    match_threshold = float(policy.get("match_threshold", 75.0))
    review_threshold = float(policy.get("review_threshold", 60.0))

    if quality_status == "FAIL":
        decision = "NEEDS_RESCAN"
        action = "Upload a clearer scan before taking any fraud decision"
    elif top1_id == claimed_writer_id and top1_conf >= match_threshold:
        decision = "MATCH"
        action = "Auto-pass"
    elif top1_conf < review_threshold:
        decision = "NEEDS_REVIEW"
        action = "Send to manual review queue"
    else:
        decision = "MISMATCH"
        action = "High-risk alert and manual verification"

    behavior = DOMAIN_BEHAVIORS.get(department, DOMAIN_BEHAVIORS["General Use"])
    action = behavior["actions"].get(decision, action)

    risk_score = max(0.0, min(100.0, 100.0 - top1_conf))
    if decision == "MISMATCH":
        risk_score = min(100.0, risk_score + float(behavior.get("mismatch_penalty", 20.0)))
    elif decision == "NEEDS_REVIEW":
        risk_score = min(100.0, risk_score + float(behavior.get("review_penalty", 8.0)))
    elif decision == "NEEDS_RESCAN":
        risk_score = min(100.0, risk_score + float(behavior.get("rescan_penalty", 18.0)))

    # Banking-only risk amplifier for high-value transactions.
    if department == "Banking":
        amount = _to_float(metadata.get("transaction_amount", 0), default=0.0)
        if amount >= 500000:
            risk_score = min(100.0, risk_score + 10.0)

    return {
        "decision": decision,
        "recommended_action": action,
        "top1_confidence": top1_conf,
        "risk_score": risk_score,
        "top_match_name": top1.get("name", top1_id),
    }


def _department_metadata_form(department: str) -> dict:
    st.markdown("#### Department Case Metadata")
    metadata: dict = {}

    if department == "Education":
        c1, c2 = st.columns(2)
        metadata["exam_id"] = c1.text_input("Exam ID", key="meta_exam_id")
        metadata["hall_ticket"] = c2.text_input("Hall Ticket", key="meta_hall_ticket")
        metadata["subject"] = c1.text_input("Subject", key="meta_subject")
        metadata["semester"] = c2.text_input("Semester", key="meta_semester")
    elif department == "Banking":
        c1, c2 = st.columns(2)
        metadata["account_or_application_id"] = c1.text_input("Account/Application ID", key="meta_acc_id")
        metadata["branch_code"] = c2.text_input("Branch Code", key="meta_branch_code")
        metadata["transaction_type"] = c1.text_input("Transaction Type", key="meta_txn_type")
        metadata["transaction_amount"] = float(c2.number_input("Transaction Amount", min_value=0.0, value=0.0, step=1000.0, key="meta_txn_amount"))
    elif department == "Property and Land Records":
        c1, c2 = st.columns(2)
        metadata["deed_number"] = c1.text_input("Deed Number", key="meta_deed_no")
        metadata["registry_office"] = c2.text_input("Registry Office", key="meta_registry")
        metadata["parcel_id"] = c1.text_input("Parcel ID", key="meta_parcel")
    elif department == "Government and Public Services":
        c1, c2 = st.columns(2)
        metadata["file_number"] = c1.text_input("File Number", key="meta_file_no")
        metadata["service_type"] = c2.text_input("Service Type", key="meta_service_type")
        metadata["citizen_record_id"] = c1.text_input("Citizen Record ID", key="meta_citizen_id")
    else:
        metadata["reference_id"] = st.text_input("Reference ID", key="meta_ref_id")
        metadata["notes"] = st.text_input("Case Notes", key="meta_notes")

    return {k: v for k, v in metadata.items() if str(v).strip()}


def _aggregate_top_predictions(per_image_results: list, top_k: int = 5) -> list:
    score_by_writer = {}
    name_by_writer = {}
    count = max(1, len(per_image_results))
    for item in per_image_results:
        for pred in item.get("top_predictions", []):
            wid = pred.get("writer_id", "unknown")
            name_by_writer[wid] = pred.get("name", wid)
            score_by_writer[wid] = score_by_writer.get(wid, 0.0) + float(pred.get("confidence", 0.0))

    ranked = sorted(score_by_writer.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "writer_id": wid,
            "name": name_by_writer.get(wid, wid),
            "confidence": total / count,
        }
        for wid, total in ranked
    ]


def _aggregate_multi_image_decision(per_image_results: list) -> dict:
    if not per_image_results:
        raise ValueError("No image results to aggregate")

    decisions = [r["decision"]["decision"] for r in per_image_results]
    counts = {k: decisions.count(k) for k in ["MATCH", "MISMATCH", "NEEDS_REVIEW", "NEEDS_RESCAN"]}
    n = len(decisions)

    if counts["NEEDS_RESCAN"] == n:
        final_decision = "NEEDS_RESCAN"
        action = "All pages/images failed quality checks. Re-upload clearer files."
    elif counts["MATCH"] == n:
        final_decision = "MATCH"
        action = "Consistent match across all processed images."
    elif counts["MISMATCH"] >= max(1, (n + 1) // 2):
        final_decision = "MISMATCH"
        action = "Majority mismatch across processed images."
    else:
        final_decision = "NEEDS_REVIEW"
        action = "Mixed evidence across images. Manual review recommended."

    avg_conf = float(np.mean([r["decision"]["top1_confidence"] for r in per_image_results]))
    avg_risk = float(np.mean([r["decision"]["risk_score"] for r in per_image_results]))
    avg_quality = float(np.mean([r["quality"]["quality_score"] for r in per_image_results]))

    top_name_votes = {}
    for r in per_image_results:
        nm = r["decision"]["top_match_name"]
        top_name_votes[nm] = top_name_votes.get(nm, 0) + 1
    top_match_name = sorted(top_name_votes.items(), key=lambda x: x[1], reverse=True)[0][0]

    if all(r["quality"]["quality_status"] == "PASS" for r in per_image_results):
        quality_status = "PASS"
    elif all(r["quality"]["quality_status"] == "FAIL" for r in per_image_results):
        quality_status = "FAIL"
    else:
        quality_status = "REVIEW"

    return {
        "decision": final_decision,
        "recommended_action": action,
        "top1_confidence": avg_conf,
        "risk_score": avg_risk,
        "top_match_name": top_match_name,
        "quality_status": quality_status,
        "quality_score": avg_quality,
    }


def _build_evidence_pdf_bytes(case_record: dict) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.95, "ScriptSentry - Case Evidence Report", ha="center", fontsize=20, fontweight="bold")
        fig.text(0.07, 0.88, f"Case ID: {case_record['case_id']}", fontsize=12)
        fig.text(0.07, 0.85, f"Department: {case_record['department']}", fontsize=12)
        fig.text(0.07, 0.82, f"Timestamp (UTC): {case_record['created_at_utc']}", fontsize=11)
        fig.text(0.07, 0.79, f"Claimed Writer: {case_record['claimed_writer_label']}", fontsize=11)
        fig.text(0.07, 0.76, f"Decision: {case_record['decision']}", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.73, f"Recommended Action: {case_record['recommended_action']}", fontsize=11)
        fig.text(0.07, 0.70, f"Confidence: {case_record['top1_confidence']:.1f}%", fontsize=11)
        fig.text(0.07, 0.67, f"Risk Score: {case_record['risk_score']:.1f}/100", fontsize=11)
        fig.text(0.07, 0.64, f"Image Quality: {case_record['quality_status']} ({case_record['quality_score']:.1f})", fontsize=11)
        fig.text(0.07, 0.61, f"Verification Mode: {case_record.get('verification_mode', 'Single Image')}", fontsize=11)
        fig.text(0.07, 0.58, f"Images Processed: {case_record.get('images_processed', 1)}", fontsize=11)

        fig.text(0.07, 0.53, "Top-5 Predictions:", fontsize=12, fontweight="bold")
        y = 0.50
        for pred in case_record.get("top_predictions", []):
            conf = float(pred.get("confidence", 0.0) * 100.0)
            fig.text(0.09, y, f"- {pred.get('name', 'Unknown')}: {conf:.1f}%", fontsize=10)
            y -= 0.03

        fig.text(0.07, 0.10, "Audit-ready artifact generated by ScriptSentry Product Dashboard.", fontsize=9, color="#64748b")
        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def _is_valid_uploaded_image(file_obj) -> bool:
    ext = file_obj.name.lower().split(".")[-1]
    return ext in {"jpg", "jpeg", "png"}


def _load_model_from_session_path() -> None:
    ckpt_path = st.session_state.checkpoint_path.strip()
    if not ckpt_path:
        st.session_state.bundle = None
        st.session_state.load_error = "Checkpoint path is empty."
        return

    try:
        with st.spinner("Loading model..."):
            st.session_state.bundle = load_model_bundle(ckpt_path, device="cpu")
        st.session_state.load_error = ""
    except Exception as exc:
        st.session_state.bundle = None
        st.session_state.load_error = str(exc)


def _render_header() -> None:
    dept = st.session_state.get("selected_department") or "Not selected"
    bundle = st.session_state.bundle
    is_online = bundle is not None
    status_text = "Online" if is_online else "Offline"
    status_class = "ss-badge-online" if is_online else "ss-badge-offline"
    dot_class = "ss-status-dot-online" if is_online else "ss-status-dot-offline"
    writers = load_enrolled_writers(str(_enrolled_json_path())).get("writers", {})
    st.markdown(
        f"""
        <div class='ss-header-card'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px'>
                <div>
                    <div style='font-size:12px;font-weight:600;opacity:0.8;text-transform:uppercase;letter-spacing:0.08em'>Handwriting Verification Platform</div>
                    <div style='font-size:32px;font-weight:800;line-height:1.15;margin-top:4px;font-family:Poppins,sans-serif;letter-spacing:-0.02em'>ScriptSentry</div>
                    <div style='margin-top:6px;font-size:14px;opacity:0.9'>Department: <b>{dept}</b></div>
                </div>
                <div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap'>
                    <span class='ss-badge {status_class}'><span class='{dot_class}' style='width:7px;height:7px;border-radius:50%;display:inline-block'></span> Model {status_text}</span>
                    <span class='ss-badge' style='background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.3);color:#fff !important'>&#x1F465; {len(writers)} Writers</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_domain_selector() -> None:
    st.markdown("### Select Domain")
    st.selectbox(
        "Department",
        options=["Select..."] + DEPARTMENTS,
        key="department_selector",
    )
    if st.button("Launch Workspace", type="primary", use_container_width=True):
        value = st.session_state.get("department_selector", "Select...")
        if value == "Select...":
            st.warning("Please choose a department first.")
        else:
            st.session_state.selected_department = value
            st.success(f"Workspace ready for {value}")


def _render_sidebar_domain_selector() -> None:
    st.sidebar.markdown("### Domain")
    current = st.session_state.get("selected_department")
    options = ["Select..."] + DEPARTMENTS
    idx = options.index(current) if current in DEPARTMENTS else 0
    selected = st.sidebar.selectbox(
        "Active Department",
        options=options,
        index=idx,
        key="sidebar_department_selector",
    )
    if st.sidebar.button("Apply Domain", use_container_width=True):
        if selected == "Select...":
            st.sidebar.warning("Please choose a valid department.")
        else:
            st.session_state.selected_department = selected
            st.sidebar.success("Domain updated")


def _render_sidebar() -> None:
    st.sidebar.markdown(
        "<div class='ss-sidebar-brand'>"
        "<div class='ss-logo-text'>&#x1F6E1;&#xFE0F; ScriptSentry</div>"
        "<div class='ss-version'>v2.0 — Continual Learning Engine</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    bundle = st.session_state.bundle
    is_online = bundle is not None
    dot_class = "ss-status-dot-online" if is_online else "ss-status-dot-offline"
    status_label = "Online" if is_online else "Offline"
    st.sidebar.markdown(
        f"<div style='margin:4px 0 12px 0'>"
        f"<span class='ss-status-dot {dot_class}'></span>"
        f"<span style='font-size:0.85rem;font-weight:600'>Model: {status_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    writers = load_enrolled_writers(str(_enrolled_json_path())).get("writers", {})
    st.sidebar.metric("Enrolled Writers", len(writers))

    if is_online:
        model_type = getattr(bundle, "model_type", "Unknown")
        tasks_learned = getattr(bundle, "tasks_learned", 0)
        file_size = getattr(bundle, "model_file_size_mb", 0.0)
        st.sidebar.caption(f"Model: {model_type} · {tasks_learned} tasks · {file_size:.0f} MB")

    with st.sidebar.expander("Model Settings", expanded=False):
        st.text_input("Checkpoint path", key="checkpoint_path")
        if st.button("Reload Model", use_container_width=True):
            _load_model_from_session_path()
        if st.session_state.load_error:
            st.error(st.session_state.load_error)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("Change Domain", use_container_width=True):
        st.session_state.app_page = "domain"
        st.session_state.selected_department = ""
        st.rerun()


def _render_quality_result(quality: dict) -> None:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Quality Score", f"{quality['quality_score']:.1f}/100")
    q2.metric("Status", quality["quality_status"])
    q3.metric("Contrast", f"{quality['contrast']:.1f}")
    q4.metric("Focus", f"{quality['focus']:.1f}")

    if quality["quality_status"] == "PASS":
        st.success("Scan quality is good for automated decision support.")
    elif quality["quality_status"] == "REVIEW":
        st.warning("Scan quality is moderate. Manual review is recommended for critical cases.")
    else:
        st.error("Scan quality is poor. Please re-upload a clearer image.")


def _domain_workspace_definition(department: str) -> dict:
    if department == "Education":
        return {
            "new_case": "Submision Verification",
            "ops": "Proxy Alerts",
            "queue": "Integrity Queue",
            "audit": "Academic Audit",
            "analytics": "Academic Analytics",
            "writers": "Student Registry",
            "ops_help": "Shows potential proxy-writing and unresolved integrity alerts.",
        }
    if department == "Banking":
        return {
            "new_case": "Doc Verification",
            "ops": "Escalations",
            "queue": "Compliance Queue",
            "audit": "Compliance Audit",
            "analytics": "Risk Analytics",
            "writers": "Customer Registry",
            "ops_help": "Shows high-risk and high-value verification escalations.",
        }
    if department == "Property and Land Records":
        return {
            "new_case": "Deed Verification",
            "ops": "Legal Alerts",
            "queue": "Registry Queue",
            "audit": "Land Audit",
            "analytics": "Registry Analytics",
            "writers": "Signatory Registry",
            "ops_help": "Shows deed/signatory anomalies requiring legal review.",
        }
    return {
        "new_case": "New Case",
        "ops": "Risk Alerts",
        "queue": "Review Queue",
        "audit": "Audit Log",
        "analytics": "Analytics",
        "writers": "Writer Management",
        "ops_help": "Shows high-risk and unresolved verification alerts.",
    }


def _page_domain_operational_view(department: str) -> None:
    cfg = _domain_workspace_definition(department)
    st.markdown(f"### {cfg['ops']}")
    st.caption(cfg["ops_help"])

    cases = _read_active_department_cases()
    if not cases:
        st.markdown("<div class='ss-empty'><h4>No Alerts</h4><p>No cases available yet for alert generation.</p></div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(cases)
    now = datetime.utcnow()

    if department == "Education":
        alerts = df[df["decision"].isin(["MISMATCH", "NEEDS_REVIEW", "NEEDS_RESCAN"])].copy()
    elif department == "Banking":
        tx = df["department_metadata"].apply(lambda x: _to_float((x or {}).get("transaction_amount", 0), 0.0))
        alerts = df[(df["risk_score"].astype(float) >= 80) | (df["decision"] == "MISMATCH") | (tx >= 500000)].copy()
    elif department == "Property and Land Records":
        alerts = df[(df["decision"] == "MISMATCH") | (df["priority"].isin(["High", "Critical"]))].copy()
    else:
        alerts = df[(df["risk_score"].astype(float) >= 75) | (df["decision"] == "MISMATCH")].copy()

    if alerts.empty:
        st.success("No operational alerts in current department.")
        return

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Alerts", len(alerts))
    a2.metric("Open Alerts", int((alerts["review_status"] == "OPEN").sum()))
    a3.metric("Avg Alert Risk", f"{alerts['risk_score'].astype(float).mean():.1f}")

    show_cols = [
        c for c in [
            "case_id",
            "decision",
            "risk_score",
            "priority",
            "assigned_to",
            "review_status",
            "created_at_utc",
        ] if c in alerts.columns
    ]
    alert_view = alerts[show_cols].rename(columns={
        "case_id": "Case ID",
        "decision": "Decision",
        "risk_score": "Risk",
        "priority": "Priority",
        "assigned_to": "Assigned To",
        "review_status": "Status",
        "created_at_utc": "Created",
    })
    st.dataframe(alert_view.sort_values(by="Risk", ascending=False), use_container_width=True)


def _page_new_case(title: str = "New Case") -> None:
    st.markdown(f"### {title}")

    if not st.session_state.selected_department:
        st.info("Select and launch a department workspace first.")
        return

    if st.session_state.bundle is None:
        st.error("Model is offline. Set a valid checkpoint path in sidebar and reload model.")
        return

    enrolled = load_enrolled_writers(str(_enrolled_json_path()))
    writers = enrolled.get("writers", {})
    if not writers:
        st.markdown("<div class='ss-empty'><h4>No Enrolled Writers</h4><p>Add writers to enrolled_writers.json before creating new cases.</p></div>", unsafe_allow_html=True)
        return

    labels, label_to_id = _writer_display_maps(writers)

    upload_key = f"verification_artifact_{st.session_state.upload_key_counter}"
    artifact = st.file_uploader(
        "Upload verification artifact (JPG, PNG, PDF, ZIP)",
        type=["jpg", "jpeg", "png", "pdf", "zip"],
        key=upload_key,
    )

    h1, h2 = st.columns([1, 1])
    with h1:
        if st.button("Use Uploaded Artifact", use_container_width=True):
            if artifact is None:
                st.warning("Please upload a file first.")
            else:
                st.session_state.uploaded_image_bytes = artifact.getvalue()
                st.session_state.uploaded_filename = artifact.name
                st.success(f"Selected: {artifact.name}")
    with h2:
        if st.button("Remove Selected File", use_container_width=True):
            _reset_uploaded_artifact()
            st.success("Selected file removed.")

    if not st.session_state.uploaded_image_bytes:
        st.markdown("<div class='ss-empty'><h4>No Verification File Selected</h4><p>Upload an artifact and click Use Uploaded Artifact.</p></div>", unsafe_allow_html=True)
        return

    c1, c2 = st.columns([1.1, 1])
    with c1:
        try:
            pseudo_file = io.BytesIO(st.session_state.uploaded_image_bytes)
            pseudo_file.name = st.session_state.uploaded_filename
            extracted = _extract_images_from_uploaded_file(pseudo_file)
            img = extracted[0]
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            return

        st.image(img, caption=st.session_state.uploaded_filename or "Verification artifact", use_container_width=True)
        if len(extracted) > 1:
            st.caption(f"{len(extracted)} images detected in artifact.")
    with c2:
        claimed_label = st.selectbox("Claimed writer name", labels, key="new_case_claimed_name")
        verification_mode = st.radio(
            "Verification Mode",
            ["Single Image", "Multi Image (aggregate)"],
            key="verification_mode",
        )

        preview_quality = _compute_quality_metrics(img)
        _render_quality_result(preview_quality)

    dept = st.session_state.selected_department
    metadata = {}
    assigned_to = st.text_input("Assign Reviewer", key="case_assigned_to", placeholder="Reviewer name")
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], key="case_priority")

    policy = st.session_state.department_policies.get(dept, DEFAULT_POLICIES.get(dept, {"match_threshold": 75.0, "review_threshold": 60.0}))
    st.caption(
        f"Policy for {dept}: match >= {policy.get('match_threshold', 75):.0f}% | review below {policy.get('review_threshold', 60):.0f}%"
    )

    if st.button("Run Verification", type="primary", use_container_width=True):
        try:
            claimed_writer_id = label_to_id[claimed_label]
            with st.spinner("Running writer verification..."):
                if verification_mode == "Single Image":
                    top_preds = predict_topk_enrolled_writers(
                        model=st.session_state.bundle.model,
                        pil_image=img,
                        device=st.session_state.bundle.device,
                        enrolled_store=enrolled,
                        top_k=5,
                    )
                    if not top_preds:
                        raise ValueError("No predictions available.")
                    quality = _compute_quality_metrics(img)
                    decision = _decision_from_prediction(
                        claimed_writer_id,
                        top_preds,
                        quality["quality_status"],
                        policy,
                        dept,
                        metadata,
                    )
                    per_image_summary = []
                    images_processed = 1
                else:
                    per_image_results = []
                    for idx, image_item in enumerate(extracted, start=1):
                        image_top_preds = predict_topk_enrolled_writers(
                            model=st.session_state.bundle.model,
                            pil_image=image_item,
                            device=st.session_state.bundle.device,
                            enrolled_store=enrolled,
                            top_k=5,
                        )
                        if not image_top_preds:
                            continue
                        image_quality = _compute_quality_metrics(image_item)
                        image_decision = _decision_from_prediction(
                            claimed_writer_id,
                            image_top_preds,
                            image_quality["quality_status"],
                            policy,
                            dept,
                            metadata,
                        )
                        per_image_results.append(
                            {
                                "image_index": idx,
                                "quality": image_quality,
                                "decision": image_decision,
                                "top_predictions": image_top_preds,
                            }
                        )

                    if not per_image_results:
                        raise ValueError("No valid pages/images could be verified.")

                    decision = _aggregate_multi_image_decision(per_image_results)
                    top_preds = _aggregate_top_predictions(per_image_results, top_k=5)
                    quality = {
                        "quality_status": decision["quality_status"],
                        "quality_score": decision["quality_score"],
                    }
                    per_image_summary = [
                        {
                            "image_index": r["image_index"],
                            "decision": r["decision"]["decision"],
                            "confidence": round(float(r["decision"]["top1_confidence"]), 1),
                            "risk_score": round(float(r["decision"]["risk_score"]), 1),
                            "quality_status": r["quality"]["quality_status"],
                        }
                        for r in per_image_results
                    ]
                    images_processed = len(per_image_results)

            cases = _read_active_department_cases()
            case_record = {
                "case_id": _next_case_id(cases),
                "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "department": st.session_state.selected_department,
                "claimed_writer_id": claimed_writer_id,
                "claimed_writer_label": claimed_label,
                "decision": decision["decision"],
                "recommended_action": decision["recommended_action"],
                "top1_confidence": decision["top1_confidence"],
                "risk_score": decision["risk_score"],
                "top_match_name": decision["top_match_name"],
                "quality_status": quality["quality_status"],
                "quality_score": quality["quality_score"],
                "top_predictions": top_preds,
                "department_metadata": metadata,
                "assigned_to": assigned_to.strip(),
                "priority": priority,
                "verification_mode": verification_mode,
                "images_processed": images_processed,
                "per_image_summary": per_image_summary,
                "review_status": "OPEN" if decision["decision"] in {"MISMATCH", "NEEDS_REVIEW", "NEEDS_RESCAN"} else "AUTO_CLOSED",
                "review_notes": "",
            }

            cases.append(case_record)
            _write_json_array(_cases_json_path(), cases)
            _append_audit(
                "case_created",
                {
                    "case_id": case_record["case_id"],
                    "decision": case_record["decision"],
                    "risk_score": case_record["risk_score"],
                    "assigned_to": case_record["assigned_to"],
                    "priority": case_record["priority"],
                },
            )

            st.session_state.case_result = case_record
            _reset_uploaded_artifact()
            st.success(f"Case {case_record['case_id']} created successfully.")
        except Exception as exc:
            st.error(f"Verification failed: {exc}")

    case = st.session_state.get("case_result")
    if not case:
        return

    st.markdown("---")
    st.markdown("### Decision Panel")

    # Decision badge
    decision_val = case["decision"]
    badge_map = {
        "MATCH": "ss-badge-match",
        "MISMATCH": "ss-badge-mismatch",
        "NEEDS_REVIEW": "ss-badge-review",
        "NEEDS_RESCAN": "ss-badge-rescan",
    }
    badge_cls = badge_map.get(decision_val, "ss-badge-review")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Case ID", case["case_id"])
    with m2:
        st.markdown(f"<div style='padding:4px 0'><small style='color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;font-size:0.82rem'>Decision</small><br><span class='ss-badge {badge_cls}' style='margin-top:6px'>{decision_val}</span></div>", unsafe_allow_html=True)
    m3.metric("Confidence", f"{case['top1_confidence']:.1f}%")
    m4.metric("Risk Score", f"{case['risk_score']:.1f}")
    st.caption(f"Mode: {case.get('verification_mode', 'Single Image')} \u00b7 Images: {case.get('images_processed', 1)}")

    st.markdown(
        f"<div class='ss-card'>"
        f"<div style='display:flex;gap:20px;flex-wrap:wrap'>"
        f"<div><span style='color:#64748b;font-size:0.8rem;font-weight:600'>TOP MATCH</span><br>"
        f"<span style='font-size:1.1rem;font-weight:700;color:#0f172a'>{case['top_match_name']}</span></div>"
        f"<div style='flex:1'><span style='color:#64748b;font-size:0.8rem;font-weight:600'>RECOMMENDED ACTION</span><br>"
        f"<span style='font-weight:600;color:#1e293b'>{case['recommended_action']}</span></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Top-5 Predictions")
    for rank, pred in enumerate(case.get("top_predictions", []), start=1):
        conf_pct = float(pred["confidence"] * 100.0)
        bar_width = int(max(0, min(100, conf_pct)))
        name = pred.get("name", "Unknown")
        st.markdown(
            f"<div class='ss-pred-card'>"
            f"<span class='ss-pred-rank'>#{rank}</span>"
            f"<span class='ss-pred-name'>{name}</span>"
            f"<span class='ss-pred-conf'>{conf_pct:.1f}%</span>"
            f"</div>"
            f"<div class='ss-conf-bar-wrap'><div class='ss-conf-bar' style='width:{bar_width}%'></div></div>",
            unsafe_allow_html=True,
        )

    if case.get("per_image_summary"):
        st.markdown("#### Per-image Verification Summary")
        st.dataframe(pd.DataFrame(case["per_image_summary"]), use_container_width=True)

    report_bytes = _build_evidence_pdf_bytes(case)
    st.download_button(
        "Download Evidence PDF",
        data=report_bytes,
        file_name=f"{case['case_id']}_evidence.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _page_review_queue(title: str = "Review Queue") -> None:
    st.markdown(f"### {title}")
    cases = _read_active_department_cases()
    open_cases = [c for c in cases if c.get("review_status") == "OPEN"]

    if not open_cases:
        st.markdown("<div class='ss-empty'><h4>No Open Cases</h4><p>All pending cases are resolved.</p></div>", unsafe_allow_html=True)
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        dept_filter = st.selectbox("Department Filter", ["All"] + DEPARTMENTS, key="queue_dept_filter")
    with f2:
        min_risk = st.slider("Minimum Risk", 0, 100, 0, key="queue_min_risk")
    with f3:
        sort_mode = st.selectbox("Sort By", ["Risk Desc", "Latest First"], key="queue_sort_mode")

    filtered = []
    for c in open_cases:
        if dept_filter != "All" and c.get("department") != dept_filter:
            continue
        if float(c.get("risk_score", 0.0)) < float(min_risk):
            continue
        filtered.append(c)

    if sort_mode == "Risk Desc":
        filtered = sorted(filtered, key=lambda x: float(x.get("risk_score", 0.0)), reverse=True)
    else:
        filtered = sorted(filtered, key=lambda x: x.get("created_at_utc", ""), reverse=True)

    if not filtered:
        st.info("No cases match current filters.")
        return

    df = pd.DataFrame(
        [
            {
                "Case ID": c["case_id"],
                "Department": c.get("department", "-"),
                "Claimed Writer": c.get("claimed_writer_label", "-"),
                "Decision": c.get("decision", "-"),
                "Risk": round(float(c.get("risk_score", 0.0)), 1),
                "Priority": c.get("priority", "-"),
                "Assigned": c.get("assigned_to", ""),
                "Created": c.get("created_at_utc", "-"),
            }
            for c in filtered
        ]
    )
    st.dataframe(df, use_container_width=True)

    case_id = st.selectbox("Select case", [c["case_id"] for c in filtered], key="queue_case_id")
    selected_case = next((c for c in filtered if c["case_id"] == case_id), None)
    if selected_case is None:
        return

    st.markdown(
        f"<div class='ss-card'><b>Claimed Writer:</b> {selected_case['claimed_writer_label']}<br>"
        f"<b>Current Decision:</b> {selected_case['decision']}<br>"
        f"<b>Risk Score:</b> {selected_case['risk_score']:.1f}<br>"
        f"<b>Priority:</b> {selected_case.get('priority', '-')}<br>"
        f"<b>Assigned To:</b> {selected_case.get('assigned_to', '')}</div>",
        unsafe_allow_html=True,
    )
    notes = st.text_area("Reviewer notes", value=selected_case.get("review_notes", ""), key="queue_notes")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Mark as Cleared", use_container_width=True):
            for item in cases:
                if item["case_id"] == case_id:
                    item["review_status"] = "CLEARED"
                    item["review_notes"] = notes
            _write_json_array(_cases_json_path(), cases)
            _append_audit("case_cleared", {"case_id": case_id})
            st.success(f"{case_id} marked as CLEARED")
    with c2:
        if st.button("Mark as Confirmed Fraud", use_container_width=True):
            for item in cases:
                if item["case_id"] == case_id:
                    item["review_status"] = "CONFIRMED_FRAUD"
                    item["review_notes"] = notes
            _write_json_array(_cases_json_path(), cases)
            _append_audit("case_confirmed_fraud", {"case_id": case_id})
            st.error(f"{case_id} marked as CONFIRMED_FRAUD")


def _page_audit_log(title: str = "Audit Log") -> None:
    st.markdown(f"### {title}")
    logs = _read_json_array(_audit_json_path())

    if not logs:
        st.markdown("<div class='ss-empty'><h4>No Audit Records</h4><p>Events will appear here once users start processing cases.</p></div>", unsafe_allow_html=True)
        return

    logs_df = pd.DataFrame(logs)
    s1, s2 = st.columns(2)
    with s1:
        event_filter = st.selectbox("Event Type", ["All"] + sorted(logs_df["event_type"].dropna().unique().tolist()), key="audit_event_filter")
    with s2:
        search_term = st.text_input("Search", key="audit_search", placeholder="case id, reviewer, decision")

    if "department" in logs_df.columns and st.session_state.selected_department:
        logs_df = logs_df[logs_df["department"] == st.session_state.selected_department]
    if event_filter != "All":
        logs_df = logs_df[logs_df["event_type"] == event_filter]
    if search_term.strip():
        logs_df = logs_df[logs_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)]

    if logs_df.empty:
        st.info("No logs for the active department yet.")
        return

    logs_df = logs_df.sort_values(by="timestamp", ascending=False)
    st.dataframe(logs_df.head(300), use_container_width=True)

    csv_bytes = logs_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Logs CSV",
        data=csv_bytes,
        file_name="audit_log_export.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_department_specific_analytics(df: pd.DataFrame, department: str) -> None:
    st.markdown("#### Department Insights")

    if department == "Education":
        proxy_alert_rate = float((df["decision"].isin(["MISMATCH", "NEEDS_REVIEW"]).sum() / max(1, len(df))) * 100.0)
        exam_ids = df["department_metadata"].apply(lambda x: (x or {}).get("exam_id", "")).replace("", np.nan).dropna().nunique()
        semester_cov = df["department_metadata"].apply(lambda x: (x or {}).get("semester", "")).replace("", np.nan).dropna().nunique()

        k1, k2, k3 = st.columns(3)
        k1.metric("Proxy Alert Rate", f"{proxy_alert_rate:.1f}%")
        k2.metric("Unique Exam IDs", int(exam_ids))
        k3.metric("Semester Coverage", int(semester_cov))

    elif department == "Banking":
        tx_amount = df["department_metadata"].apply(lambda x: _to_float((x or {}).get("transaction_amount", 0), 0.0))
        high_value = int((tx_amount >= 500000).sum())
        escalation = int((df["risk_score"].astype(float) >= 80).sum())
        mismatch_rate = float((df["decision"].eq("MISMATCH").sum() / max(1, len(df))) * 100.0)

        k1, k2, k3 = st.columns(3)
        k1.metric("High-Value Cases", high_value)
        k2.metric("Escalation Candidates", escalation)
        k3.metric("Mismatch Rate", f"{mismatch_rate:.1f}%")

    elif department == "Property and Land Records":
        deeds = df["department_metadata"].apply(lambda x: (x or {}).get("deed_number", "")).replace("", np.nan).dropna().nunique()
        offices = df["department_metadata"].apply(lambda x: (x or {}).get("registry_office", "")).replace("", np.nan).dropna().nunique()
        legal_priority = int(((df["decision"] == "MISMATCH") | (df["review_status"] == "CONFIRMED_FRAUD")).sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Unique Deeds Tracked", int(deeds))
        k2.metric("Registry Offices", int(offices))
        k3.metric("Legal Priority Cases", legal_priority)

    else:
        high_risk = int((df["risk_score"].astype(float) >= 75).sum())
        needs_review = int((df["decision"] == "NEEDS_REVIEW").sum())
        avg_conf = float(df["top1_confidence"].astype(float).mean()) if "top1_confidence" in df else 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric("High-Risk Cases", high_risk)
        k2.metric("Needs Review", needs_review)
        k3.metric("Average Confidence", f"{avg_conf:.1f}%")


def _page_analytics(title: str = "Analytics") -> None:
    st.markdown(f"### {title}")
    cases = _read_active_department_cases()
    if not cases:
        st.markdown("<div class='ss-empty'><h4>No Case Data</h4><p>Create cases to view analytics.</p></div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(cases)
    if st.session_state.selected_department:
        df = df[df["department"] == st.session_state.selected_department]
    active_department = st.session_state.selected_department or "General Use"
    if df.empty:
        st.info("No analytics for the active department yet.")
        return

    total_cases = len(df)
    open_cases = int((df["review_status"] == "OPEN").sum())
    confirmed = int((df["review_status"] == "CONFIRMED_FRAUD").sum())
    avg_risk = float(df["risk_score"].astype(float).mean()) if "risk_score" in df else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", total_cases)
    c2.metric("Open Cases", open_cases)
    c3.metric("Confirmed Fraud", confirmed)
    c4.metric("Average Risk", f"{avg_risk:.1f}")

    st.markdown("#### Department Decision Mix")
    decision_counts = df["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
    st.bar_chart(decision_counts, x="Decision", y="Count")

    st.markdown("#### Priority Distribution")
    if "priority" in df:
        priority_counts = df["priority"].fillna("Unspecified").value_counts().rename_axis("Priority").reset_index(name="Count")
        st.bar_chart(priority_counts, x="Priority", y="Count")

    _render_department_specific_analytics(df, active_department)


def _page_writer_management(title: str = "Writer Management") -> None:
    st.markdown(f"### {title}")
    if st.session_state.bundle is None:
        st.error("Model is offline. Reload model from sidebar before writer enrollment.")
        return

    enrolled = load_enrolled_writers(str(_enrolled_json_path()))
    writers = enrolled.get("writers", {})
    st.metric("Total Writers", len(writers))

    st.markdown("#### Enroll New Writer")
    st.caption("Upload images, a PDF, or a ZIP. The writer name will be suggested from the filename and can be edited.")
    uploaded_files = st.file_uploader(
        "Upload handwriting artifacts (JPG, PNG, PDF, ZIP)",
        type=["jpg", "jpeg", "png", "pdf", "zip"],
        accept_multiple_files=True,
        key=f"enroll_writer_files_single_{st.session_state.enroll_writer_uploader_counter}",
    )

    if uploaded_files:
        suggested_name = _infer_writer_name_from_filename(uploaded_files[0].name)
        _seed_text_input("enroll_writer_name", suggested_name)

    writer_name = st.text_input("Writer Name", key="enroll_writer_name", placeholder="Example: Aarav Singh")

    if st.button("Enroll Writer", type="primary", use_container_width=True):
        if not _valid_writer_name(writer_name):
            st.error("Invalid writer name. Use letters/numbers (with optional spaces, apostrophes, underscores, and hyphens).")
        elif uploaded_files is None or len(uploaded_files) == 0:
            st.error("Please upload at least one supported file.")
        else:
            try:
                extracted_images: List[Image.Image] = []
                for uf in uploaded_files:
                    extracted_images.extend(_extract_images_from_uploaded_file(uf))

                if len(extracted_images) < 3:
                    st.error("Could not extract at least 3 handwriting samples from the uploaded files.")
                    return

                if len(extracted_images) > 5:
                    st.warning("More than 5 samples found. Using first 5 samples for enrollment.")
                    extracted_images = extracted_images[:5]

                enroll_payload = [
                    _pil_to_named_bytesio(img, f"enroll_{idx+1}.png")
                    for idx, img in enumerate(extracted_images)
                ]

                with st.spinner("Enrolling writer..."):
                    result = enroll_writer(
                        writer_name=writer_name.strip(),
                        uploaded_files=enroll_payload,
                        model=st.session_state.bundle.model,
                        device=st.session_state.bundle.device,
                        json_path=str(_enrolled_json_path()),
                        task_id=max(1, int(getattr(st.session_state.bundle, "tasks_learned", 1))),
                    )
                _append_audit("writer_enrolled", {"writer_id": result["writer_id"], "writer_name": result["writer_name"]})
                st.success(f"Writer enrolled: {result['writer_name']}")
                st.write(f"Enrollment confidence: {result['confidence']:.1f}%")
                _reset_single_enrollment_state()
                st.rerun()
            except Exception as exc:
                st.error(f"Enrollment failed: {exc}")

    st.markdown("---")
    st.markdown("#### Bulk Registration from ZIP")
    st.caption("Upload a ZIP containing handwriting files. File names are grouped into writer names, which you can edit before approval.")
    bulk_zip = st.file_uploader(
        "Upload ZIP for bulk registration",
        type=["zip"],
        accept_multiple_files=False,
        key=f"bulk_zip_upload_{st.session_state.bulk_uploader_counter}",
    )

    if bulk_zip:
        try:
            st.session_state.bulk_groups = _group_zip_samples_by_writer(bulk_zip.getvalue())
        except Exception as exc:
            st.session_state.bulk_groups = []
            st.error(f"Could not parse ZIP: {exc}")

    bulk_groups = st.session_state.get("bulk_groups", [])
    if bulk_groups:
        st.markdown("##### Extracted Writers")
        bulk_preview_rows = []
        for group in bulk_groups:
            bulk_preview_rows.append({
                "Default Name": group["default_name"],
                "Samples": len(group["files"]),
            })
        st.dataframe(pd.DataFrame(bulk_preview_rows), use_container_width=True)

        st.markdown("##### Edit Writer Names")
        edited_groups = []
        for group in bulk_groups:
            row1, row2 = st.columns([2, 1])
            name_key = f"bulk_name_{group['id']}"
            _seed_text_input(name_key, group["default_name"])
            with row1:
                edited_name = st.text_input(
                    f"Writer name for {group['default_name']}",
                    key=name_key,
                    placeholder="Edit name if needed",
                )
            with row2:
                st.write(f"{len(group['files'])} sample(s)")
            edited_groups.append({
                "name": edited_name,
                "files": group["files"],
            })

        approve_key = f"bulk_approve_{st.session_state.bulk_uploader_counter}"
        approve_bulk = st.checkbox("I approve the extracted names and sample counts", key=approve_key)
        if st.button("Register All Writers", type="primary", use_container_width=True):
            if not approve_bulk:
                st.warning("Please approve the extracted names before registering.")
            else:
                success_rows = []
                error_rows = []
                for item in edited_groups:
                    try:
                        writer_name_value = item["name"].strip()
                        if not _valid_writer_name(writer_name_value):
                            raise ValueError("Invalid writer name")
                        if len(item["files"]) < 3:
                            raise ValueError("Need at least 3 samples")

                        sample_images = []
                        for sample in item["files"][:5]:
                            sample_images.append(Image.open(io.BytesIO(sample["bytes"])).convert("L"))

                        enroll_payload = [_pil_to_named_bytesio(img, f"{writer_name_value}_{idx+1}.png") for idx, img in enumerate(sample_images)]

                        result = enroll_writer(
                            writer_name=writer_name_value,
                            uploaded_files=enroll_payload,
                            model=st.session_state.bundle.model,
                            device=st.session_state.bundle.device,
                            json_path=str(_enrolled_json_path()),
                            task_id=max(1, int(getattr(st.session_state.bundle, "tasks_learned", 1))),
                        )
                        success_rows.append({"Writer": result["writer_name"], "Confidence": f"{result['confidence']:.1f}%"})
                        _append_audit("writer_enrolled_bulk", {"writer_id": result["writer_id"], "writer_name": result["writer_name"]})
                    except Exception as exc:
                        error_rows.append({"Writer": item["name"], "Error": str(exc)})

                if success_rows:
                    st.success(f"Registered {len(success_rows)} writer(s) successfully.")
                    st.dataframe(pd.DataFrame(success_rows), use_container_width=True)
                if success_rows and not error_rows:
                    _reset_bulk_enrollment_state()
                    st.rerun()
                if error_rows:
                    st.warning("Some writers could not be registered.")
                    st.dataframe(pd.DataFrame(error_rows), use_container_width=True)

    st.markdown("#### Delete Writer")
    enrolled = load_enrolled_writers(str(_enrolled_json_path()))
    writers = enrolled.get("writers", {})
    if not writers:
        st.info("No writers available to delete.")
        return

    labels, label_to_id = _writer_display_maps(writers)
    selected_label = st.selectbox("Select writer to delete", labels, key="delete_writer_label")
    confirm_delete = st.checkbox("I understand this action permanently removes the writer", key="delete_confirm")
    if st.button("Delete Writer", use_container_width=True):
        if not confirm_delete:
            st.warning("Please confirm deletion before continuing.")
        else:
            writer_id = label_to_id[selected_label]
            if _delete_writer_by_id(writer_id):
                _append_audit("writer_deleted", {"writer_id": writer_id, "writer_name": selected_label})
                st.success(f"Writer deleted: {selected_label}")
            else:
                st.error("Unable to delete writer. Please try again.")


def _render_main_panel() -> None:
    _render_header()

    if not st.session_state.selected_department:
        _render_domain_selector()
        st.markdown("<div class='ss-empty'><h4>Choose a Domain to Begin</h4><p>After selecting a department, continue to dashboard and use feature tabs in the main area.</p></div>", unsafe_allow_html=True)
        return

    dept = st.session_state.selected_department
    cfg = _domain_workspace_definition(dept)

    st.markdown("<div class='ss-pill'>Department Workspace</div>", unsafe_allow_html=True)
    tab_new, tab_ops, tab_queue, tab_audit, tab_analytics, tab_writers = st.tabs(
        [
            cfg["new_case"],
            cfg["ops"],
            cfg["queue"],
            cfg["audit"],
            cfg["analytics"],
            cfg["writers"],
        ]
    )
    with tab_new:
        _page_new_case(cfg["new_case"])
    with tab_ops:
        _page_domain_operational_view(dept)
    with tab_queue:
        _page_review_queue(cfg["queue"])
    with tab_audit:
        _page_audit_log(cfg["audit"])
    with tab_analytics:
        _page_analytics(cfg["analytics"])
    with tab_writers:
        _page_writer_management(cfg["writers"])


def _render_domain_selection_page() -> None:
    st.markdown(
        """
        <div class='ss-hero'>
            <div style='font-size:3rem;margin-bottom:8px'>&#x1F6E1;&#xFE0F;</div>
            <div class='ss-hero-title'>ScriptSentry</div>
            <div class='ss-hero-subtitle'>AI-Powered Handwriting Verification Platform</div>
            <div style='margin-top:14px'>
                <div class='ss-features-row'>
                    <span class='ss-feature-pill' style='background:rgba(255,255,255,0.18);border-color:rgba(255,255,255,0.3);color:#fff !important'>&#x270D;&#xFE0F; Writer Identification</span>
                    <span class='ss-feature-pill' style='background:rgba(255,255,255,0.18);border-color:rgba(255,255,255,0.3);color:#fff !important'>&#x1F9E0; Continual Learning</span>
                    <span class='ss-feature-pill' style='background:rgba(255,255,255,0.18);border-color:rgba(255,255,255,0.3);color:#fff !important'>&#x1F50D; GradCAM Analysis</span>
                    <span class='ss-feature-pill' style='background:rgba(255,255,255,0.18);border-color:rgba(255,255,255,0.3);color:#fff !important'>&#x1F4CA; Multi-Image Verification</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Select Your Department")
    st.caption("Choose a domain workspace to access tailored verification features.")

    st.markdown(
        """
        <div class='ss-domain-grid'>
            <div class='ss-domain-card ss-domain-card-edu'>
                <span class='ss-icon'>&#x1F393;</span>
                <div class='ss-title'>Education</div>
                <div class='ss-desc'>Exam paper verification, proxy-writing detection, academic integrity</div>
            </div>
            <div class='ss-domain-card ss-domain-card-bank'>
                <span class='ss-icon'>&#x1F3E6;</span>
                <div class='ss-title'>Banking</div>
                <div class='ss-desc'>Signature verification, transaction fraud detection, compliance</div>
            </div>
            <div class='ss-domain-card ss-domain-card-prop'>
                <span class='ss-icon'>&#x1F3DB;&#xFE0F;</span>
                <div class='ss-title'>Property & Land Records</div>
                <div class='ss-desc'>Deed verification, signatory validation, registry processing</div>
            </div>
            <div class='ss-domain-card ss-domain-card-gen'>
                <span class='ss-icon'>&#x1F527;</span>
                <div class='ss-title'>General Use</div>
                <div class='ss-desc'>Flexible writer verification for any handwriting analysis scenario</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Department",
        options=["Select..."] + DEPARTMENTS,
        key="department_selector",
        label_visibility="collapsed",
    )
    if st.button("Launch Workspace", type="primary", use_container_width=True):
        value = st.session_state.get("department_selector", "Select...")
        if value == "Select...":
            st.warning("Please choose a department first.")
        else:
            st.session_state.selected_department = value
            st.session_state.app_page = "dashboard"
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="ScriptSentry Dashboard", layout="wide")
    _inject_modern_theme()
    _init_session_state()
    _ensure_persistence_files()
    st.session_state.department_policies = _load_policies()

    if st.session_state.bundle is None and not st.session_state.load_error:
        _load_model_from_session_path()

    if st.session_state.app_page == "domain":
        _render_domain_selection_page()
    else:
        _render_sidebar()
        _render_main_panel()


if __name__ == "__main__":
    main()
