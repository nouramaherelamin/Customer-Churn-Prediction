# ============================================================
# ChurnAI — Customer Churn Prediction
# Final Portfolio Edition
# Owned & Developed by Noura Maher Elamin
# ============================================================

import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Consistent dark chart theme for the portfolio UI.
plt.rcParams.update({
    "figure.facecolor": "#0f151d",
    "axes.facecolor": "#0f151d",
    "axes.edgecolor": "#2a3544",
    "axes.labelcolor": "#aeb9c8",
    "axes.titlecolor": "#eef2f7",
    "xtick.color": "#8d9aad",
    "ytick.color": "#8d9aad",
    "text.color": "#dce5ef",
    "grid.color": "#2a3544",
    "legend.facecolor": "#111923",
    "legend.edgecolor": "#2a3544",
})


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "churn_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.pkl")
COMPARISON_PATH = os.path.join(BASE_DIR, "model_comparison_results.csv")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold_results.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")
FAVICON_PATH = os.path.join(BASE_DIR, "nm_favicon.png")


def _build_nm_favicon():
    """
    Generate a small, clean "NM" monogram favicon (dark background,
    blue text) the first time the app runs, instead of using any
    robot icon or logo image. Falls back to a plain emoji icon if
    Pillow is unavailable.
    """
    if os.path.exists(FAVICON_PATH):
        return FAVICON_PATH

    try:
        from PIL import Image, ImageDraw, ImageFont

        size = 64
        image = Image.new("RGB", (size, size), color="#0d141d")
        draw = ImageDraw.Draw(image)

        font = None
        for font_name in (
            "DejaVuSans-Bold.ttf",
            "Arial Bold.ttf",
            "Arial.ttf",
        ):
            try:
                font = ImageFont.truetype(font_name, 30)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        text = "NM"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.text(
            (
                (size - text_w) / 2 - bbox[0],
                (size - text_h) / 2 - bbox[1],
            ),
            text,
            fill="#2f8cff",
            font=font,
        )

        image.save(FAVICON_PATH, format="PNG")
        return FAVICON_PATH

    except Exception:
        return "📊"


PAGE_ICON = _build_nm_favicon()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# OWNER
# ============================================================

OWNER = "Noura Maher Elamin"
LINKEDIN_URL = "https://www.linkedin.com/in/nouramaherelamin/"
GITHUB_URL = "https://github.com/nouramaherelamin"



# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #080b10;
    --panel: #11161f;
    --panel-2: #151b25;
    --panel-3: #0d1219;
    --border: #202a38;
    --border-soft: rgba(99, 115, 137, .20);
    --text: #f2f5f9;
    --muted: #7f8da1;
    --blue: #2f8cff;
    --blue-2: #1677ff;
    --green: #28e28b;
    --red: #ff4d5e;
    --yellow: #f6c453;
}

html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
}

p, span, div, label, input, textarea, button {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.hero-title,
.section-title,
.panel-title,
.result-title,
.metric-value,
.top-brand,
.top-value,
.footer-brand,
.sidebar-wordmark,
.developer-name,
.timeline-name {
    font-family: "Space Grotesk", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 75% 8%, rgba(25, 105, 190, .10), transparent 28%),
        radial-gradient(circle at 12% 42%, rgba(24, 80, 135, .06), transparent 25%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1540px;
    padding: 0 30px 30px;
}

#MainMenu,
footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #0a0f16 0%, #090d14 100%);
    border-right: 1px solid #18212c;
}

section[data-testid="stSidebar"] > div {
    padding: 18px 14px;
}

.sidebar-brand {
    padding: 12px 10px 20px;
    border-bottom: 1px solid #18212c;
    margin-bottom: 20px;
}

.sidebar-wordmark {
    font-family: "Space Grotesk", sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -.5px;
}

.sidebar-sub {
    color: #68778b;
    font-size: 9px;
    margin-top: 3px;
}

.sidebar-label {
    color: #68778b;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 18px 10px 8px;
}

div[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

div[data-testid="stSidebar"] .stRadio label {
    border-radius: 9px;
    padding: 8px 10px;
    transition: .2s ease;
    color: #8e9bad !important;
}

div[data-testid="stSidebar"] .stRadio label:hover {
    background: #111a27;
    color: #fff !important;
}

div[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
}

div[data-testid="stSidebar"] .stRadio label {
    border: 1px solid transparent;
    margin-bottom: 2px;
}

div[data-testid="stSidebar"] .stRadio label p {
    font-size: 12px !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

div[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(28, 128, 255, .18), rgba(28, 128, 255, .06));
    border: 1px solid rgba(47, 140, 255, .20);
    color: #fff !important;
}

.sidebar-owner {
    margin-top: 28px;
    padding: 18px 12px;
    border: 1px solid #1c2735;
    background: #0c121a;
    border-radius: 11px;
    text-align: center;
}

.sidebar-owner-name {
    color: #fff;
    font-weight: 700;
    font-size: 11px;
}

.sidebar-owner-rights {
    color: #5f6c7d;
    font-size: 8px;
    margin-top: 7px;
}

/* ============================================================
   TOP BAR
   ============================================================ */

.topbar {
    height: 72px;
    display: grid;
    grid-template-columns: 1.25fr repeat(4, minmax(100px, 1fr)) 1.15fr 78px;
    align-items: center;
    border-bottom: 1px solid #18212c;
    margin: 0 -30px 0;
    padding: 0 30px;
    background: rgba(8, 11, 16, .90);
}

.top-brand {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
    font-size: 19px;
    color: #fff;
}

.top-sub {
    color: #68778b;
    font-size: 8px;
    margin-top: 2px;
}

.top-metric {
    padding-left: 18px;
    border-left: 1px solid #1d2733;
}

.top-label {
    color: #738196;
    font-size: 8px;
}

.top-value {
    color: #fff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 12px;
    font-weight: 700;
    margin-top: 3px;
}

.top-value.green {
    color: var(--green);
}

.top-value.blue {
    color: #66aaff;
}

.top-owner {
    text-align: right;
    padding-right: 12px;
    color: #d9dee7;
    font-size: 9px;
    font-weight: 600;
    white-space: nowrap;
}

.top-links {
    display: flex;
    justify-content: flex-end;
    gap: 7px;
}

.icon-link {
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-decoration: none !important;
    color: #dfe6f0 !important;
    background: #111822;
    border: 1px solid #273343;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 800;
    transition: .2s ease;
}

.icon-link:hover {
    background: #172232;
    border-color: #398fff;
    color: #fff !important;
    transform: translateY(-1px);
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    min-height: 360px;
    margin: 30px 0 25px;
    padding: 48px 0 35px;
    overflow: hidden;
    border-bottom: 1px solid #17212d;
}

.hero-glow {
    position: absolute;
    width: 540px;
    height: 330px;
    right: 2%;
    top: -30px;
    background:
        radial-gradient(ellipse at center, rgba(26, 126, 255, .19), transparent 58%);
    filter: blur(8px);
    pointer-events: none;
}

.hero-grid {
    position: absolute;
    right: 0;
    top: 0;
    width: 48%;
    height: 100%;
    opacity: .17;
    background-image:
        linear-gradient(rgba(64, 122, 190, .18) 1px, transparent 1px),
        linear-gradient(90deg, rgba(64, 122, 190, .18) 1px, transparent 1px);
    background-size: 35px 35px;
    mask-image: linear-gradient(90deg, transparent, #000 35%);
}

.hero-copy {
    position: relative;
    z-index: 2;
    max-width: 690px;
}

.eyebrow {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(35, 142, 255, .28);
    background: rgba(22, 119, 255, .07);
    color: #5ca9ff;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: 1.1px;
}

.hero-title {
    margin: 18px 0 15px;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(43px, 5.5vw, 70px);
    line-height: .99;
    letter-spacing: -2.7px;
    color: #f5f7fb;
    font-weight: 700;
}

.hero-title span {
    color: #3997ff;
}

.hero-description {
    color: #8e9bad;
    font-size: 13px;
    line-height: 1.75;
    max-width: 650px;
}

.hero-actions {
    display: flex;
    gap: 10px;
    margin-top: 24px;
}

.hero-mini-grid {
    display: grid;
    grid-template-columns: repeat(4, 110px);
    gap: 22px;
    margin-top: 36px;
}

.hero-mini-value {
    color: #fff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 18px;
    font-weight: 700;
}

.hero-mini-label {
    color: #647287;
    font-size: 8px;
    margin-top: 4px;
}

/* ============================================================
   CARDS
   ============================================================ */

.section-title {
    font-family: "Space Grotesk", sans-serif;
    color: #eef2f7;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -.4px;
}

.section-kicker {
    color: #4b9cff;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: 1.2px;
    margin-bottom: 7px;
    text-transform: uppercase;
}

.section-desc {
    color: #6e7d91;
    font-size: 10px;
    line-height: 1.65;
    margin-top: 4px;
}

.metric-card {
    min-height: 102px;
    padding: 17px;
    border-radius: 9px;
    border: 1px solid #202b39;
    background: linear-gradient(145deg, #111720, #0d131b);
    box-shadow: 0 12px 28px rgba(0, 0, 0, .17);
    transition: .2s ease;
}

.metric-card:hover {
    border-color: #2a3b50;
    transform: translateY(-2px);
}

.metric-icon {
    color: #4b9cff;
    font-size: 16px;
    margin-bottom: 10px;
}

.metric-label {
    color: #718095;
    font-size: 8px;
}

.metric-value {
    color: #f4f6f9;
    font-family: "Space Grotesk", sans-serif;
    font-size: 23px;
    font-weight: 700;
    margin-top: 3px;
}

.panel {
    padding: 20px;
    border-radius: 9px;
    border: 1px solid #202b39;
    background: linear-gradient(145deg, #111720, #0d131b);
    box-shadow: 0 12px 30px rgba(0, 0, 0, .15);
    margin-bottom: 16px;
}

.panel-title {
    color: #eef2f7;
    font-family: "Space Grotesk", sans-serif;
    font-size: 16px;
    font-weight: 700;
}

.panel-desc {
    color: #718095;
    font-size: 9px;
    line-height: 1.65;
    margin-top: 4px;
}

.divider {
    height: 1px;
    background: #1b2531;
    margin: 16px 0;
}

/* ============================================================
   PREDICTION RESULT
   ============================================================ */

.result-panel {
    min-height: 100%;
    padding: 22px;
    border-radius: 9px;
    border: 1px solid #202b39;
    background:
        radial-gradient(circle at 90% 5%, rgba(27, 132, 255, .10), transparent 30%),
        linear-gradient(145deg, #121923, #0d131b);
}

.result-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #61d8a0;
    background: rgba(40, 226, 139, .06);
    border: 1px solid rgba(40, 226, 139, .15);
    padding: 5px 8px;
    border-radius: 999px;
    font-size: 8px;
    font-weight: 700;
}

.result-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin: 16px 0 5px;
}

.result-title.low {
    color: #38e88f;
}

.result-title.medium {
    color: #f3c34e;
}

.result-title.high {
    color: #ff5967;
}

.probability {
    font-family: "Space Grotesk", sans-serif;
    font-size: 39px;
    font-weight: 700;
    color: #fff;
}

.probability.low {
    color: #35e890;
}

.probability.medium {
    color: #f3c34e;
}

.probability.high {
    color: #ff5967;
}

.result-copy {
    color: #8491a4;
    font-size: 10px;
    line-height: 1.7;
}

.action-box {
    margin-top: 17px;
    padding: 14px;
    border-radius: 8px;
    border: 1px solid #223248;
    background: #0e1722;
}

.action-title {
    color: #66aaff;
    font-size: 9px;
    font-weight: 800;
}

.action-copy {
    color: #9ba7b8;
    font-size: 9px;
    line-height: 1.65;
    margin-top: 7px;
}

.risk-bar {
    margin-top: 14px;
    height: 6px;
    border-radius: 99px;
    background: #1c2734;
    overflow: hidden;
}

.risk-fill {
    height: 100%;
    border-radius: 99px;
}

.gauge-card {
    display: flex;
    justify-content: center;
    padding: 12px 0 2px;
}

.gauge {
    width: 158px;
    height: 158px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: conic-gradient(
        #31e690 var(--progress),
        #1b2837 0
    );
    position: relative;
}

.gauge:before {
    content: "";
    position: absolute;
    width: 124px;
    height: 124px;
    border-radius: 50%;
    background: #101720;
    border: 1px solid #243141;
}

.gauge-inner {
    position: relative;
    z-index: 2;
    text-align: center;
}

.gauge-number {
    color: #fff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 28px;
    font-weight: 700;
}

.gauge-label {
    color: #68778a;
    font-size: 8px;
}

/* ============================================================
   SIGNALS / EXPLAINABILITY
   ============================================================ */

.signal-row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1b2530;
}

.signal-row:last-child {
    border-bottom: 0;
}

.signal-name {
    color: #aab4c3;
    font-size: 9px;
}

.signal-value {
    color: #e8edf4;
    font-size: 9px;
    font-weight: 700;
}

.feature-row {
    display: grid;
    grid-template-columns: 190px 50px 1fr;
    gap: 10px;
    align-items: center;
    padding: 7px 0;
}

.feature-name {
    color: #aeb7c5;
    font-size: 9px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.feature-score {
    color: #6f7d90;
    font-size: 8px;
    text-align: right;
}

.feature-track {
    height: 5px;
    background: #1a2532;
    border-radius: 99px;
    overflow: hidden;
}

.feature-fill {
    height: 100%;
    background: #39df98;
    border-radius: 99px;
}

.check-row {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 7px 0;
    color: #9ca8b9;
    font-size: 9px;
    line-height: 1.55;
}

.check {
    color: #34e28e;
}

/* ============================================================
   TABLE / CHART
   ============================================================ */

.chart-panel {
    padding: 16px;
    border: 1px solid #202b39;
    border-radius: 9px;
    background: #10161e;
}

.data-note {
    padding: 12px 14px;
    border-left: 2px solid #348fff;
    background: rgba(31, 130, 255, .045);
    color: #8390a2;
    font-size: 9px;
    line-height: 1.65;
    margin: 12px 0;
}

/* ============================================================
   PROJECT / DEVELOPER
   ============================================================ */

.timeline {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 7px;
}

.timeline-step {
    padding: 15px 8px;
    border: 1px solid #202b39;
    border-radius: 8px;
    background: #0f151d;
    text-align: center;
}

.timeline-num {
    color: #fff;
    font-size: 9px;
    font-weight: 800;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 7px;
    border-radius: 50%;
    background: #176fe8;
}

.timeline-name {
    color: #dce3ec;
    font-size: 8px;
    font-weight: 700;
}

.timeline-desc {
    color: #667589;
    font-size: 7px;
    margin-top: 3px;
}

.developer-card {
    padding: 26px;
    border-radius: 10px;
    border: 1px solid #253244;
    background:
        radial-gradient(circle at 90% 10%, rgba(35, 139, 255, .10), transparent 30%),
        #101720;
}

.developer-kicker {
    color: #5aa5ff;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: 1px;
}

.developer-name {
    color: #fff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 28px;
    font-weight: 700;
    margin: 8px 0 16px;
}

.developer-links {
    display: flex;
    gap: 8px;
}

.developer-link {
    display: inline-flex;
    text-decoration: none !important;
    color: #dce5ef !important;
    background: #141d28;
    border: 1px solid #29384a;
    border-radius: 7px;
    padding: 8px 13px;
    font-size: 9px;
    font-weight: 700;
}

.developer-link:hover {
    color: #fff !important;
    border-color: #398fff;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin: 55px -30px -30px;
    padding: 30px;
    border-top: 1px solid #1a2430;
    background: #0c1118;
}

.footer-grid {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 1fr;
    gap: 35px;
}

.footer-brand {
    color: #fff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 18px;
    font-weight: 700;
}

.footer-desc {
    color: #647286;
    font-size: 9px;
    line-height: 1.7;
    max-width: 320px;
    margin-top: 8px;
}

.footer-heading {
    color: #dce3ec;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .7px;
    margin-bottom: 10px;
}

.footer-item {
    color: #718095;
    font-size: 9px;
    margin: 7px 0;
}

.footer-bottom {
    border-top: 1px solid #1b2530;
    margin-top: 24px;
    padding-top: 17px;
    display: flex;
    justify-content: space-between;
    gap: 20px;
    color: #59677a;
    font-size: 8px;
}

.st-key-footer_wrap {
    margin: 55px -30px -30px;
    padding: 30px 30px 6px;
    border-top: 1px solid #1a2430;
    background: #0c1118;
}

.st-key-footer_wrap .stButton {
    margin: 0;
}

.st-key-footer_wrap .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #718095 !important;
    font-size: 9px !important;
    font-weight: 500 !important;
    padding: 3px 0 !important;
    min-height: unset !important;
    height: auto !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
}

.st-key-footer_wrap .stButton > button:hover {
    color: #2f8cff !important;
    transform: none !important;
    background: transparent !important;
}

/* ============================================================
   STREAMLIT CONTROLS
   ============================================================ */

.stButton > button {
    border-radius: 7px;
    min-height: 40px;
    border: 1px solid rgba(55, 145, 255, .45);
    background: #1979f3;
    color: white;
    font-size: 10px;
    font-weight: 700;
    transition: .2s ease;
}

.stButton > button:hover {
    background: #2785fb;
    border-color: #5aa5ff;
    transform: translateY(-1px);
}

.stDownloadButton > button {
    border-radius: 7px;
    min-height: 40px;
    border: 1px solid #29405b;
    background: #111b27;
    color: #dfe7f0;
    font-size: 10px;
    font-weight: 700;
}

div[data-baseweb="input"],
div[data-baseweb="select"] {
    background: #0d141d !important;
    border-color: #243244 !important;
}

input, textarea {
    color: #edf2f8 !important;
}

label {
    color: #8996a8 !important;
    font-size: 9px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0d131b;
    border: 1px solid #1e2a37;
    padding: 5px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    color: #738196;
    font-size: 9px;
    border-radius: 6px;
    padding: 9px 12px;
}

.stTabs [aria-selected="true"] {
    color: #fff !important;
    background: #162235;
}

[data-testid="stDataFrame"] {
    border: 1px solid #202b39;
    border-radius: 8px;
    overflow: hidden;
}

hr {
    border-color: #1a2430 !important;
}

@media(max-width: 1050px) {
    .topbar {
        grid-template-columns: 1.2fr repeat(3, 1fr) 1fr;
    }
    .topbar .hide-small {
        display: none;
    }
    .timeline {
        grid-template-columns: repeat(4, 1fr);
    }
}

@media(max-width: 800px) {
    .block-container {
        padding: 0 15px 20px;
    }
    .topbar {
        margin: 0 -15px;
        padding: 0 15px;
        grid-template-columns: 1fr 1fr 1fr;
    }
    .top-owner, .top-links {
        display: none;
    }
    .hero-title {
        font-size: 45px;
    }
    .hero-mini-grid {
        grid-template-columns: repeat(2, 110px);
    }
    .footer {
        margin-left: -15px;
        margin-right: -15px;
    }
    .footer-grid {
        grid-template-columns: 1fr 1fr;
    }
}

</style>
""")

# ============================================================
# FINAL UI OVERRIDES
# ============================================================
# These rules intentionally come last so Streamlit's generated component
# styles cannot shrink the portfolio typography or restore radio controls.
st.html("""
<style>

/* --- Typography hierarchy --- */
.stApp, .stApp * {
    -webkit-font-smoothing: antialiased;
    text-rendering: geometricPrecision;
}

body, .stApp {
    font-size: 15px !important;
}

.section-kicker, .developer-kicker, .sidebar-label, .footer-heading {
    font-size: 10px !important;
    letter-spacing: 1.25px !important;
}

.section-title {
    font-size: 24px !important;
    line-height: 1.25 !important;
    letter-spacing: -.5px !important;
}

.section-desc {
    font-size: 12px !important;
    line-height: 1.65 !important;
}

.panel-title {
    font-size: 18px !important;
    line-height: 1.35 !important;
}

.panel-desc {
    font-size: 11px !important;
    line-height: 1.65 !important;
}

.metric-label, .top-label, .hero-mini-label {
    font-size: 10px !important;
}

.metric-value {
    font-size: 28px !important;
}

.top-brand {
    font-size: 22px !important;
}

.top-sub {
    font-size: 9px !important;
}

.top-value {
    font-size: 14px !important;
}

.top-owner {
    font-size: 10px !important;
}

.sidebar-wordmark {
    font-size: 25px !important;
}

.sidebar-sub {
    font-size: 10px !important;
}

.sidebar-owner-name {
    font-size: 12px !important;
}

.sidebar-owner-rights {
    font-size: 9px !important;
}

.footer-brand {
    font-size: 21px !important;
}

.footer-desc, .footer-item {
    font-size: 10px !important;
}

.footer-bottom {
    font-size: 9px !important;
}

/* --- Clean custom sidebar navigation --- */
section[data-testid="stSidebar"] .stButton {
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 40px !important;
    height: 40px !important;
    padding: 0 12px !important;
    margin: 0 0 3px 0 !important;
    border-radius: 7px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #aeb9c8 !important;
    font-family: "Inter", sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #111a26 !important;
    border-color: #1d2b3d !important;
    color: #ffffff !important;
    transform: none !important;
}

/* Streamlit primary button = active nav item */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(47,140,255,.16), rgba(47,140,255,.05)) !important;
    border-color: rgba(47,140,255,.24) !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    font-size: 13px !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* Remove the old radio widget completely */
section[data-testid="stSidebar"] div[data-testid="stRadio"] {
    display: none !important;
}

/* --- Inputs: readable, polished, not tiny --- */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stMultiSelect label, .stSlider label {
    font-size: 11px !important;
    font-weight: 600 !important;
}

.stTextInput input, .stNumberInput input,
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    font-size: 13px !important;
}

.stButton > button, .stDownloadButton > button {
    font-size: 12px !important;
    font-weight: 700 !important;
}

/* --- Make cards breathe like the reference dashboard --- */
.metric-card {
    padding: 20px !important;
    min-height: 125px !important;
}

.panel {
    padding: 22px !important;
}

.signal-name, .signal-value, .feature-name {
    font-size: 11px !important;
}

.check-row, .action-copy, .result-copy {
    font-size: 11px !important;
}

.timeline-name {
    font-size: 10px !important;
}

.timeline-desc {
    font-size: 8px !important;
}

.developer-name {
    font-size: 31px !important;
}

.developer-link {
    font-size: 11px !important;
    padding: 9px 14px !important;
}

/* --- Hide Streamlit chrome / deploy area --- */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeaderActionElements"],
[data-testid="stAppDeployButton"] {
    display: none !important;
    visibility: hidden !important;
}

/* --- Keep the dashboard centered and closer to the reference --- */
.block-container {
    max-width: 1580px !important;
    padding-top: 0 !important;
}

.topbar {
    height: 76px !important;
}

/* --- Dark charts: no more huge white rectangles --- */
.stPlotlyChart, .element-container:has(canvas) {
    border-radius: 9px !important;
}

/* Streamlit dataframe typography */
[data-testid="stDataFrame"] * {
    font-size: 11px !important;
}

</style>
""")


# ============================================================
# HELPERS
# ============================================================

def html(content):
    st.html(content)


def find_column(dataframe, candidates):
    lookup = {str(c).strip().lower(): c for c in dataframe.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for col in dataframe.columns:
        col_low = str(col).lower()
        if any(candidate.lower() in col_low for candidate in candidates):
            return col

    return None


def load_optional_csv(path):
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_model_features(model_object):
    return list(getattr(model_object, "feature_names_in_", []))


def encode_single_customer(raw):
    """
    Rebuild the exact one-hot feature layout expected by the saved model.
    This avoids the common single-row get_dummies bug.
    """
    model_features = get_model_features(model)

    if not model_features:
        raise ValueError(
            "The saved model does not expose feature_names_in_."
        )

    encoded = pd.DataFrame(
        0.0,
        index=[0],
        columns=model_features,
    )

    for col in [
        "Account length",
        "Area code",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total day charge",
        "Total eve minutes",
        "Total eve calls",
        "Total eve charge",
        "Total night minutes",
        "Total night calls",
        "Total night charge",
        "Total intl minutes",
        "Total intl calls",
        "Total intl charge",
        "Customer service calls",
    ]:
        if col in encoded.columns and col in raw:
            encoded.loc[0, col] = float(raw[col])

    categorical = {
        f"State_{raw['State']}": 1.0,
        f"International plan_{raw['International plan']}": 1.0,
        f"Voice mail plan_{raw['Voice mail plan']}": 1.0,
    }

    for col, value in categorical.items():
        if col in encoded.columns:
            encoded.loc[0, col] = value

    return encoded.astype(float)


def risk_bucket(probability):
    pct = probability * 100

    if pct < 30:
        return (
            "Low Risk",
            "low",
            "Continue normal engagement and monitor customer behavior.",
        )

    if pct < 60:
        return (
            "Medium Risk",
            "medium",
            "Review service experience and monitor the customer more closely.",
        )

    return (
        "High Risk",
        "high",
        "Consider proactive retention outreach and review service issues.",
    )


def risk_signals(raw):
    signals = []

    if raw["Customer service calls"] >= 4:
        signals.append(
            ("Customer service calls", "Elevated support activity")
        )
    elif raw["Customer service calls"] >= 2:
        signals.append(
            ("Customer service calls", "Moderate support activity")
        )
    else:
        signals.append(
            ("Customer service calls", "Low support activity")
        )

    if raw["International plan"] == "Yes":
        signals.append(
            ("International plan", "Active")
        )
    else:
        signals.append(
            ("International plan", "Not active")
        )

    if raw["Total day minutes"] >= 250:
        signals.append(
            ("Day usage", "High usage pattern")
        )
    else:
        signals.append(
            ("Day usage", "Within normal observed range")
        )

    if raw["Total intl minutes"] >= 20:
        signals.append(
            ("International usage", "High international usage")
        )
    else:
        signals.append(
            ("International usage", "Within normal observed range")
        )

    if raw["Account length"] >= float(df["Account length"].median()):
        signals.append(
            ("Account length", "At or above dataset median")
        )
    else:
        signals.append(
            ("Account length", "Below dataset median")
        )

    return signals


def similar_customer_count(raw):
    """
    Descriptive profile matching, not a model prediction.
    Counts customers with the same state and plans and reasonably close
    key numeric usage values.
    """
    try:
        mask = (
            (df["State"].astype(str) == str(raw["State"])) &
            (df["International plan"].astype(str) == str(raw["International plan"])) &
            (df["Voice mail plan"].astype(str) == str(raw["Voice mail plan"]))
        )

        subset = df.loc[mask].copy()

        if subset.empty:
            return 0

        ranges = [
            ("Account length", 0.20),
            ("Total day minutes", 0.20),
            ("Total eve minutes", 0.20),
            ("Total night minutes", 0.20),
            ("Total intl minutes", 0.25),
        ]

        for col, tolerance in ranges:
            center = float(raw[col])
            if center == 0:
                subset = subset[subset[col].abs() <= 1]
            else:
                low = center * (1 - tolerance)
                high = center * (1 + tolerance)
                subset = subset[
                    subset[col].between(low, high)
                ]

        return int(len(subset))

    except Exception:
        return 0


def build_pdf_report(record):
    """
    Creates a compact PDF report if ReportLab is installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return None, (
            "PDF export requires ReportLab. "
            "Install it with: pip install reportlab"
        )

    filename = os.path.join(
        BASE_DIR,
        "customer_churn_report.pdf",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ChurnTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#111827"),
        spaceAfter=10,
    )

    body_style = ParagraphStyle(
        "ChurnBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#374151"),
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    story = [
        Paragraph("ChurnAI — Customer Churn Report", title_style),
        Paragraph(
            "Customer Churn Prediction | Owned & Developed by "
            "Noura Maher Elamin",
            body_style,
        ),
        Spacer(1, 8),
    ]

    summary_data = [
        ["Prediction", record["Prediction"]],
        ["Churn Probability", f'{record["Churn Probability"]:.2f}%'],
        ["Risk Level", record["Risk Level"]],
        ["Timestamp", record["Timestamp"]],
    ]

    table = Table(summary_data, colWidths=[45 * mm, 120 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4fb")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story += [
        table,
        Spacer(1, 13),
        Paragraph("Customer Profile", styles["Heading2"]),
        Spacer(1, 5),
    ]

    profile_rows = [
        ["State", str(record["State"])],
        ["International Plan", str(record["International Plan"])],
        ["Voice Mail Plan", str(record["Voice Mail Plan"])],
        ["Customer Service Calls", str(record["Customer Service Calls"])],
        ["Total Day Minutes", str(record["Total Day Minutes"])],
        ["Total Intl Minutes", str(record["Total Intl Minutes"])],
        ["Account Length", str(record["Account Length"])],
    ]

    profile_table = Table(
        profile_rows,
        colWidths=[55 * mm, 110 * mm],
    )

    profile_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story += [
        profile_table,
        Spacer(1, 13),
        Paragraph("Suggested Business Action", styles["Heading2"]),
        Spacer(1, 5),
        Paragraph(str(record["Suggested Action"]), body_style),
        Spacer(1, 13),
        Paragraph(
            "Important: the risk signals displayed by the application are "
            "descriptive indicators for the entered profile. They should "
            "not be interpreted as causal explanations.",
            body_style,
        ),
        Spacer(1, 18),
        Paragraph(
            "© 2026 Noura Maher Elamin — All Rights Reserved",
            body_style,
        ),
    ]

    doc.build(story)

    return filename, None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = [str(c).strip() for c in data.columns]
    return data


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # The project artifact may have been saved with joblib even though
    # the file extension is .pkl. Try pickle first, then joblib.
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except Exception as pickle_error:

        try:
            import joblib
            return joblib.load(MODEL_PATH)

        except Exception as joblib_error:
            raise RuntimeError(
                "The saved model could not be loaded.\n\n"
                f"Pickle error: {pickle_error}\n\n"
                f"Joblib error: {joblib_error}\n\n"
                "If the artifact is corrupted, recreate it from the notebook "
                "and save it again with joblib.dump(model, 'customer_churn_model.pkl')."
            )


try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error("Project files could not be loaded.")
    st.code(str(error))
    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================

TARGET_COL = "Churn"

TOTAL_CUSTOMERS = len(df)
TOTAL_COLUMNS = len(df.columns)
TOTAL_FEATURES = TOTAL_COLUMNS - 1

CHURNED = int(df[TARGET_COL].sum())
RETAINED = int(TOTAL_CUSTOMERS - CHURNED)
CHURN_RATE = (
    CHURNED / TOTAL_CUSTOMERS * 100
    if TOTAL_CUSTOMERS
    else 0
)

MISSING_VALUES = int(df.isnull().sum().sum())
DUPLICATES = int(df.duplicated().sum())

DATA_QUALITY = (
    "Clean"
    if MISSING_VALUES == 0 and DUPLICATES == 0
    else "Needs Review"
)

STATE_OPTIONS = (
    sorted(df["State"].astype(str).unique().tolist())
    if "State" in df.columns
    else ["CA"]
)

AREA_CODE_OPTIONS = (
    sorted(df["Area code"].dropna().astype(int).unique().tolist())
    if "Area code" in df.columns
    else [415]
)


# ============================================================
# PERSISTENT HISTORY
# ============================================================
#
# Reliable, explicit-error history implementation.
# Every function below always uses the same HISTORY_PATH
# (BASE_DIR/prediction_history.csv), never the process's current
# working directory, so it behaves the same however the app is
# launched.


def load_history_from_disk():
    """
    Read prediction_history.csv from disk.

    Returns (records, error):
      - records: list[dict] — [] if the file does not exist yet
        (this is the normal "no predictions saved" state, not an error).
      - error: the actual Exception if the file exists but could not
        be read (corrupt CSV, permission issue, etc.), otherwise None.
        Callers must surface this to the user — it must never be
        swallowed into a silent empty list.
    """
    if not os.path.exists(HISTORY_PATH):
        return [], None

    try:
        history_df = pd.read_csv(HISTORY_PATH)
        return history_df.to_dict("records"), None
    except Exception as error:
        return [], error


def save_history_to_disk(records):
    """
    Overwrite prediction_history.csv on disk with the complete list
    of records. Raises on failure — callers must handle/report it.
    """
    history_df = pd.DataFrame(records)
    history_df.to_csv(HISTORY_PATH, index=False)


def add_prediction_to_history(record):
    """
    Append exactly one prediction record to prediction_history.csv
    and keep st.session_state.history in sync with disk.

    Flow (matches the required save sequence):
      1. Read existing history from disk.
      2. Append exactly one record.
      3. Write the complete history back to disk.
      4. Reload the history from disk.
      5. Update st.session_state.history.

    Returns (success: bool, error: Exception | None). Never silently
    swallows a real failure — the caller decides how to display it.
    """
    existing_records, read_error = load_history_from_disk()

    if read_error is not None:
        return False, read_error

    updated_records = existing_records + [record]

    try:
        save_history_to_disk(updated_records)
    except Exception as error:
        return False, error

    reloaded_records, reload_error = load_history_from_disk()

    if reload_error is not None:
        return False, reload_error

    st.session_state.history = reloaded_records
    return True, None


if "history" not in st.session_state:
    _initial_history, _initial_history_error = load_history_from_disk()
    st.session_state.history = _initial_history
    st.session_state.history_load_error = (
        str(_initial_history_error)
        if _initial_history_error is not None
        else None
    )



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html("""
    <div class="sidebar-brand">
        <div class="sidebar-wordmark">ChurnAI</div>
        <div class="sidebar-sub">Customer Churn Prediction</div>
    </div>
    """)

    st.markdown(
        '<div class="sidebar-label">MAIN</div>',
        unsafe_allow_html=True,
    )

    # Custom navigation buttons — no Streamlit radio circles/icons.
    nav_items = [
        ("Home", "HOME"),
        ("Predict", "PREDICT"),
        ("Explore Data", "DATA"),
        ("Business Insights", "INSIGHTS"),
        ("Model Lab", "MODEL"),
        ("Prediction History", "HISTORY"),
        ("About Project", "ABOUT"),
    ]

    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home"

    for nav_name, nav_code in nav_items:
        active = st.session_state.nav_page == nav_name
        label = nav_name
        if st.button(
            label,
            key=f"nav_{nav_code}",
            width="stretch",
            type="primary" if active else "secondary",
        ):
            st.session_state.nav_page = nav_name
            st.rerun()

    page = st.session_state.nav_page

    st.markdown(
        '<div class="sidebar-label">QUICK LINKS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <a class="developer-link"
           href="{GITHUB_URL}"
           target="_blank"
           style="display:block;text-align:center;margin-bottom:7px;">
           GitHub
        </a>

        <a class="developer-link"
           href="{LINKEDIN_URL}"
           target="_blank"
           style="display:block;text-align:center;">
           LinkedIn
        </a>
        """,
        unsafe_allow_html=True,
    )

    html(f"""
    <div class="sidebar-owner">
        <div class="sidebar-owner-name">{OWNER}</div>
        <div class="sidebar-owner-rights">
            © 2026 — All Rights Reserved
        </div>
    </div>
    """)


# ============================================================
# TOP BAR
# ============================================================

html(f"""
<div class="topbar">

    <div>
        <div class="top-brand">ChurnAI</div>
        <div class="top-sub">Customer Churn Prediction</div>
    </div>

    <div class="top-metric">
        <div class="top-label">Customers</div>
        <div class="top-value">{TOTAL_CUSTOMERS:,}</div>
    </div>

    <div class="top-metric">
        <div class="top-label">Churn Rate</div>
        <div class="top-value green">{CHURN_RATE:.1f}%</div>
    </div>

    <div class="top-metric">
        <div class="top-label">Features</div>
        <div class="top-value blue">{TOTAL_FEATURES}</div>
    </div>

    <div class="top-metric hide-small">
        <div class="top-label">Model</div>
        <div class="top-value">Random Forest</div>
    </div>

    <div class="top-owner">{OWNER}</div>

    <div class="top-links">
        <a class="icon-link" href="{LINKEDIN_URL}" target="_blank" aria-label="LinkedIn">in</a>
        <a class="icon-link" href="{GITHUB_URL}" target="_blank" aria-label="GitHub">GH</a>
    </div>

</div>
""")


# ============================================================
# HOME
# ============================================================

if page == "Home":

    html(f"""
    <div class="hero">

        <div class="hero-glow"></div>
        <div class="hero-grid"></div>

        <div class="hero-copy">

            <div class="eyebrow">
                DATA-DRIVEN CUSTOMER INTELLIGENCE
            </div>

            <div class="hero-title">
                Turn Customer Data<br>
                into <span>Smarter Decisions.</span>
            </div>

            <div class="hero-description">
                ChurnAI is an end-to-end Customer Churn Prediction
                application that combines data exploration, machine
                learning, model evaluation, explainability, and
                business-oriented insights in one dashboard.
            </div>

            <div class="hero-mini-grid">

                <div>
                    <div class="hero-mini-value">{TOTAL_CUSTOMERS:,}</div>
                    <div class="hero-mini-label">Customers analyzed</div>
                </div>

                <div>
                    <div class="hero-mini-value">{TOTAL_FEATURES}</div>
                    <div class="hero-mini-label">Input features</div>
                </div>

                <div>
                    <div class="hero-mini-value">{CHURNED:,}</div>
                    <div class="hero-mini-label">Churned customers</div>
                </div>

                <div>
                    <div class="hero-mini-value">{CHURN_RATE:.1f}%</div>
                    <div class="hero-mini-label">Observed churn rate</div>
                </div>

            </div>

        </div>

    </div>
    """)

    hero_col1, hero_col2, hero_col3 = st.columns([1, 1, 3])

    with hero_col1:
        if st.button(
            "Predict Customer Churn",
            key="hero_cta_predict",
            width="stretch",
            type="primary",
        ):
            st.session_state.nav_page = "Predict"
            st.rerun()

    with hero_col2:
        if st.button(
            "Explore Data",
            key="hero_cta_explore",
            width="stretch",
            type="secondary",
        ):
            st.session_state.nav_page = "Explore Data"
            st.rerun()

    html("""
    <div style="margin:22px 0 15px;">
        <div class="section-kicker">PLATFORM</div>
        <div class="section-title">
            Everything the churn dashboard needs, nothing it doesn't.
        </div>
        <div class="section-desc">
            Built around the complete machine learning workflow, from raw
            customer data to an interactive prediction experience.
        </div>
    </div>
    """)

    feature_cols = st.columns(4)

    home_features = [
        (
            "↗",
            "Churn Prediction",
            "Estimate customer churn probability using the deployed Random Forest model.",
            "Predict",
        ),
        (
            "◫",
            "Explore Data",
            "Inspect dataset quality, distributions, categories, and customer behavior.",
            "Explore Data",
        ),
        (
            "⌁",
            "Model Lab",
            "Compare models, inspect metrics, ROC curves, thresholds, and feature importance.",
            "Model Lab",
        ),
        (
            "◇",
            "Business Insights",
            "Turn descriptive customer patterns into practical retention-oriented observations.",
            "Business Insights",
        ),
    ]

    for col, (icon, title, desc, target_page) in zip(feature_cols, home_features):
        with col:
            html(f"""
            <div class="metric-card" style="min-height:155px;">
                <div class="metric-icon">{icon}</div>
                <div style="color:#e9eef5;font-size:11px;font-weight:700;">
                    {title}
                </div>
                <div style="color:#718095;font-size:8px;line-height:1.7;margin-top:8px;">
                    {desc}
                </div>
            </div>
            """)

            if st.button(
                "Open →",
                key=f"home_card_{target_page.replace(' ', '_')}",
                width="stretch",
            ):
                st.session_state.nav_page = target_page
                st.rerun()

    st.write("")

    html("""
    <div class="panel">
        <div class="section-kicker">PROJECT FLOW</div>
        <div class="section-title">From dataset to deployment.</div>
        <div class="section-desc">
            The application exposes the important stages of the project instead
            of hiding the work behind one prediction button.
        </div>

        <div class="timeline" style="margin-top:18px;">

            <div class="timeline-step">
                <div class="timeline-num">1</div>
                <div class="timeline-name">EDA</div>
                <div class="timeline-desc">Explore</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">2</div>
                <div class="timeline-name">Quality</div>
                <div class="timeline-desc">Validate</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">3</div>
                <div class="timeline-name">Encoding</div>
                <div class="timeline-desc">Preprocess</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">4</div>
                <div class="timeline-name">Models</div>
                <div class="timeline-desc">Compare</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">5</div>
                <div class="timeline-name">Tuning</div>
                <div class="timeline-desc">Optimize</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">6</div>
                <div class="timeline-name">Evaluation</div>
                <div class="timeline-desc">Measure</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">7</div>
                <div class="timeline-name">Deployment</div>
                <div class="timeline-desc">Streamlit</div>
            </div>

        </div>
    </div>
    """)


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    html("""
    <div style="margin:27px 0 18px;">
        <div class="section-kicker">PREDICTION ENGINE</div>
        <div class="section-title">Predict Customer Churn</div>
        <div class="section-desc">
            Enter the customer's complete profile. The application rebuilds
            the exact feature layout expected by the saved model.
        </div>
    </div>
    """)

    form_col, result_col = st.columns([1.25, .85], gap="large")

    with form_col:

        html("""
        <div class="panel">
            <div class="panel-title">Customer Information</div>
            <div class="panel-desc">
                Customer profile, plans, usage, and charge information.
            </div>
            <div class="divider"></div>
        </div>
        """)

        html("""
        <div style="color:#dfe6ef;font-size:10px;font-weight:700;margin-bottom:8px;">
            Customer Profile
        </div>
        """)

        c1, c2, c3 = st.columns(3)

        with c1:
            state = st.selectbox(
                "State",
                STATE_OPTIONS,
            )

        with c2:
            account_length = st.number_input(
                "Account Length",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
            )

        with c3:
            area_code = st.selectbox(
                "Area Code",
                AREA_CODE_OPTIONS,
            )

        html("""
        <div class="divider"></div>

        <div style="color:#dfe6ef;font-size:10px;font-weight:700;margin-bottom:8px;">
            Customer Plans
        </div>
        """)

        p1, p2 = st.columns(2)

        with p1:
            international_plan = st.selectbox(
                "International Plan",
                ["No", "Yes"],
            )

        with p2:
            voice_mail_plan = st.selectbox(
                "Voice Mail Plan",
                ["No", "Yes"],
            )

        number_vmail_messages = st.number_input(
            "Number of Voice Mail Messages",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
        )

        html("""
        <div class="divider"></div>

        <div style="color:#dfe6ef;font-size:10px;font-weight:700;margin-bottom:8px;">
            Usage Information
        </div>
        """)

        u1, u2, u3 = st.columns(3)

        with u1:
            total_day_minutes = st.number_input(
                "Total Day Minutes",
                min_value=0.0,
                max_value=500.0,
                value=180.0,
                step=1.0,
            )

        with u2:
            total_day_calls = st.number_input(
                "Total Day Calls",
                min_value=0,
                max_value=200,
                value=100,
                step=1,
            )

        with u3:
            total_day_charge = st.number_input(
                "Total Day Charge",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=.1,
            )

        u4, u5, u6 = st.columns(3)

        with u4:
            total_eve_minutes = st.number_input(
                "Total Eve Minutes",
                min_value=0.0,
                max_value=500.0,
                value=200.0,
                step=1.0,
            )

        with u5:
            total_eve_calls = st.number_input(
                "Total Eve Calls",
                min_value=0,
                max_value=200,
                value=100,
                step=1,
            )

        with u6:
            total_eve_charge = st.number_input(
                "Total Eve Charge",
                min_value=0.0,
                max_value=100.0,
                value=17.0,
                step=.1,
            )

        u7, u8, u9 = st.columns(3)

        with u7:
            total_night_minutes = st.number_input(
                "Total Night Minutes",
                min_value=0.0,
                max_value=500.0,
                value=200.0,
                step=1.0,
            )

        with u8:
            total_night_calls = st.number_input(
                "Total Night Calls",
                min_value=0,
                max_value=200,
                value=100,
                step=1,
            )

        with u9:
            total_night_charge = st.number_input(
                "Total Night Charge",
                min_value=0.0,
                max_value=100.0,
                value=9.0,
                step=.1,
            )

        html("""
        <div class="divider"></div>

        <div style="color:#dfe6ef;font-size:10px;font-weight:700;margin-bottom:8px;">
            International & Support
        </div>
        """)

        i1, i2, i3 = st.columns(3)

        with i1:
            total_intl_minutes = st.number_input(
                "Total Intl Minutes",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=1.0,
            )

        with i2:
            total_intl_calls = st.number_input(
                "Total Intl Calls",
                min_value=0,
                max_value=100,
                value=4,
                step=1,
            )

        with i3:
            total_intl_charge = st.number_input(
                "Total Intl Charge",
                min_value=0.0,
                max_value=100.0,
                value=2.7,
                step=.1,
            )

        customer_service_calls = st.number_input(
            "Customer Service Calls",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

        st.write("")

        predict_clicked = st.button(
            "Predict Customer Churn",
            width="stretch",
        )

    with result_col:

        if predict_clicked:

            raw_customer = {
                "State": state,
                "Account length": account_length,
                "Area code": area_code,
                "International plan": international_plan,
                "Voice mail plan": voice_mail_plan,
                "Number vmail messages": number_vmail_messages,
                "Total day minutes": total_day_minutes,
                "Total day calls": total_day_calls,
                "Total day charge": total_day_charge,
                "Total eve minutes": total_eve_minutes,
                "Total eve calls": total_eve_calls,
                "Total eve charge": total_eve_charge,
                "Total night minutes": total_night_minutes,
                "Total night calls": total_night_calls,
                "Total night charge": total_night_charge,
                "Total intl minutes": total_intl_minutes,
                "Total intl calls": total_intl_calls,
                "Total intl charge": total_intl_charge,
                "Customer service calls": customer_service_calls,
            }

            with st.spinner("Analyzing customer profile..."):
                time.sleep(.25)

                try:
                    encoded_customer = encode_single_customer(
                        raw_customer
                    )

                    prediction = int(
                        model.predict(encoded_customer)[0]
                    )

                    probability = float(
                        model.predict_proba(
                            encoded_customer
                        )[0][1]
                    )

                except Exception as error:
                    st.error("Prediction failed.")
                    st.code(str(error))
                    st.stop()

            # Freeze this prediction so it survives reruns triggered by
            # anything else on this page (Save, downloads, etc.) — a
            # plain st.button() return value is only True for the one
            # rerun in which it was clicked, so without this snapshot
            # the entire result block (Save button included) would
            # disappear the instant any other button on the page fired.
            st.session_state.pred_snapshot = {
                "raw_customer": raw_customer,
                "prediction": prediction,
                "probability": probability,
            }

        has_prediction = (
            st.session_state.get("pred_snapshot") is not None
        )

        if has_prediction:

            snapshot = st.session_state.pred_snapshot
            raw_customer = snapshot["raw_customer"]
            prediction = snapshot["prediction"]
            probability = snapshot["probability"]

            risk_label, risk_class, action = risk_bucket(
                probability
            )

            probability_pct = probability * 100
            confidence_proxy = max(
                probability,
                1 - probability,
            ) * 100

            similar_count = similar_customer_count(
                raw_customer
            )

            title = (
                "High Churn Risk"
                if risk_class == "high"
                else
                "Medium Churn Risk"
                if risk_class == "medium"
                else
                "Low Churn Risk"
            )

            if prediction == 1:
                prediction_text = "Predicted to Churn"
            else:
                prediction_text = "Predicted to Stay"

            signals = risk_signals(
                raw_customer
            )

            html(f"""
            <div class="result-panel">

                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="color:#e9eef5;font-size:15px;font-weight:700;">
                        Prediction Result
                    </div>

                    <div class="result-status">
                        <span style="color:#34e28e;">●</span>
                        Live Prediction
                    </div>
                </div>

                <div class="result-title {risk_class}">
                    {title}
                </div>

                <div class="result-copy">
                    {prediction_text}. The probability below is the model's
                    estimated probability for the positive class (Churn).
                </div>

                <div style="margin-top:12px;">
                    <div style="color:#7e8ca0;font-size:9px;">
                        Churn Probability
                    </div>

                    <div class="probability {risk_class}">
                        {probability_pct:.1f}%
                    </div>
                </div>

                <div class="risk-bar">
                    <div
                        class="risk-fill"
                        style="
                            width:{probability_pct:.2f}%;
                            background:
                                {'#ff5967' if risk_class == 'high'
                                else '#f3c34e' if risk_class == 'medium'
                                else '#35e890'};
                        "
                    ></div>
                </div>

                <div class="gauge-card">
                    <div
                        class="gauge"
                        style="
                            --progress:{probability_pct * 3.6}deg;
                            background:conic-gradient(
                                {'#ff5967' if risk_class == 'high'
                                else '#f3c34e' if risk_class == 'medium'
                                else '#35e890'}
                                var(--progress),
                                #1b2837 0
                            );
                        "
                    >
                        <div class="gauge-inner">
                            <div class="gauge-number">
                                {probability_pct:.0f}%
                            </div>
                            <div class="gauge-label">
                                CHURN RISK
                            </div>
                        </div>
                    </div>
                </div>

                <div class="action-box">
                    <div class="action-title">
                        Suggested Business Action
                    </div>

                    <div class="action-copy">
                        {action}
                    </div>
                </div>

            </div>
            """)

            # ------------------------------------------------
            # RISK PROFILE
            # ------------------------------------------------

            st.write("")

            a, b = st.columns(2)

            with a:
                html("""
                <div class="panel">
                    <div class="panel-title">Customer Risk Profile</div>
                    <div class="panel-desc">
                        Key information supplied to the model.
                    </div>
                """)

                profile = [
                    ("Risk Level", risk_label),
                    ("Churn Probability", f"{probability_pct:.1f}%"),
                    ("International Plan", raw_customer["International plan"]),
                    ("Voice Mail Plan", raw_customer["Voice mail plan"]),
                    ("Support Calls", raw_customer["Customer service calls"]),
                    ("Day Usage", f"{raw_customer['Total day minutes']:.1f} min"),
                    ("Account Length", f"{raw_customer['Account length']} days"),
                    ("Intl Usage", f"{raw_customer['Total intl minutes']:.1f} min"),
                ]

                for label, value in profile:
                    html(f"""
                    <div class="signal-row">
                        <span class="signal-name">{label}</span>
                        <span class="signal-value">{value}</span>
                    </div>
                    """)

                html("</div>")

            with b:
                html("""
                <div class="panel">
                    <div class="panel-title">Risk Signals</div>
                    <div class="panel-desc">
                        Descriptive indicators from the entered profile.
                    </div>
                """)

                for label, value in signals:
                    html(f"""
                    <div class="check-row">
                        <span class="check">●</span>
                        <span>
                            <strong style="color:#cfd7e3;">
                                {label}
                            </strong>
                            — {value}
                        </span>
                    </div>
                    """)

                html("</div>")

            # ------------------------------------------------
            # CONFIDENCE / SIMILAR CUSTOMERS
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            with c1:
                html(f"""
                <div class="panel">

                    <div class="panel-title">
                        Prediction Confidence Proxy
                    </div>

                    <div class="panel-desc">
                        Distance of the predicted probability from 50%.
                        This is not a calibrated confidence interval.
                    </div>

                    <div style="
                        height:7px;
                        background:#1a2634;
                        border-radius:99px;
                        margin-top:16px;
                        overflow:hidden;
                    ">
                        <div style="
                            width:{confidence_proxy:.1f}%;
                            height:100%;
                            background:#2f8cff;
                            border-radius:99px;
                        "></div>
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-top:8px;
                    ">
                        <span style="color:#718095;font-size:8px;">
                            Prediction confidence proxy
                        </span>

                        <strong style="color:#dce5ef;font-size:9px;">
                            {confidence_proxy:.0f}%
                        </strong>
                    </div>

                </div>
                """)

            with c2:
                html(f"""
                <div class="panel">

                    <div class="panel-title">
                        Similar Customers
                    </div>

                    <div class="panel-desc">
                        Descriptive profile matches in the historical dataset.
                    </div>

                    <div style="
                        display:flex;
                        align-items:end;
                        gap:10px;
                        margin-top:14px;
                    ">
                        <div style="
                            color:#36e596;
                            font-family:'Space Grotesk',sans-serif;
                            font-size:29px;
                            font-weight:700;
                        ">
                            {similar_count:,}
                        </div>

                        <div style="
                            color:#718095;
                            font-size:8px;
                            padding-bottom:5px;
                        ">
                            similar profiles found
                        </div>
                    </div>

                </div>
                """)

            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            importance_df = load_optional_csv(
                IMPORTANCE_PATH
            )

            if importance_df is not None:

                feature_col = find_column(
                    importance_df,
                    ["feature", "features", "feature name"],
                )

                score_col = find_column(
                    importance_df,
                    [
                        "importance",
                        "feature importance",
                        "importance score",
                    ],
                )

                if feature_col and score_col:

                    fi = importance_df[
                        [feature_col, score_col]
                    ].copy()

                    fi.columns = [
                        "Feature",
                        "Importance",
                    ]

                else:
                    fi = pd.DataFrame()

            elif hasattr(model, "feature_importances_"):

                fi = pd.DataFrame({
                    "Feature": get_model_features(model),
                    "Importance": model.feature_importances_,
                })

            else:
                fi = pd.DataFrame()

            if not fi.empty:

                fi["Importance"] = pd.to_numeric(
                    fi["Importance"],
                    errors="coerce",
                )

                fi = fi.dropna().sort_values(
                    "Importance",
                    ascending=False,
                )

                top = fi.head(5)
                max_score = float(
                    top["Importance"].max()
                )

                html("""
                <div class="panel">
                    <div class="panel-title">
                        Why does the model pay attention to these features?
                    </div>
                    <div class="panel-desc">
                        Global feature importance from the trained Random Forest.
                        Importance is a model statistic, not a causal explanation
                        of an individual customer's outcome.
                    </div>
                """)

                for _, row in top.iterrows():

                    score = float(
                        row["Importance"]
                    )

                    width = (
                        score / max_score * 100
                        if max_score
                        else 0
                    )

                    html(f"""
                    <div class="feature-row">

                        <div class="feature-name">
                            {row["Feature"]}
                        </div>

                        <div class="feature-score">
                            {score:.3f}
                        </div>

                        <div class="feature-track">
                            <div
                                class="feature-fill"
                                style="width:{width:.1f}%"
                            ></div>
                        </div>

                    </div>
                    """)

                html("</div>")

            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            record = {
                "Timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "State": raw_customer["State"],
                "Account Length": raw_customer["Account length"],
                "Area Code": raw_customer["Area code"],
                "International Plan": raw_customer["International plan"],
                "Voice Mail Plan": raw_customer["Voice mail plan"],
                "Number Vmail Messages": raw_customer["Number vmail messages"],
                "Total Day Minutes": raw_customer["Total day minutes"],
                "Total Day Calls": raw_customer["Total day calls"],
                "Total Day Charge": raw_customer["Total day charge"],
                "Total Eve Minutes": raw_customer["Total eve minutes"],
                "Total Eve Calls": raw_customer["Total eve calls"],
                "Total Eve Charge": raw_customer["Total eve charge"],
                "Total Night Minutes": raw_customer["Total night minutes"],
                "Total Night Calls": raw_customer["Total night calls"],
                "Total Night Charge": raw_customer["Total night charge"],
                "Total Intl Minutes": raw_customer["Total intl minutes"],
                "Total Intl Calls": raw_customer["Total intl calls"],
                "Total Intl Charge": raw_customer["Total intl charge"],
                "Customer Service Calls": raw_customer["Customer service calls"],
                "Churn Probability": round(
                    probability_pct,
                    2,
                ),
                "Risk Level": risk_label,
                "Prediction": (
                    "Churn"
                    if prediction == 1
                    else "No Churn"
                ),
                "Suggested Action": action,
            }

            st.session_state.last_prediction = record

            if st.button(
                "Save Prediction to History",
                key="save_prediction_btn",
                width="stretch",
            ):

                save_success, save_error = add_prediction_to_history(
                    record
                )

                if save_success:
                    # st.rerun() interrupts the script immediately, so a
                    # message shown right before it is never actually
                    # seen — stash it and display it after the rerun
                    # instead, right below.
                    st.session_state.history_save_message = (
                        "Prediction saved successfully."
                    )
                    st.rerun()
                else:
                    st.error(
                        f"Prediction could not be saved: {save_error}"
                    )

            if st.session_state.get("history_save_message"):
                st.success(st.session_state.history_save_message)
                st.session_state.history_save_message = None

            download_col1, download_col2 = st.columns(2)

            with download_col1:

                prediction_csv = pd.DataFrame(
                    [record]
                ).to_csv(
                    index=False
                )

                st.download_button(
                    "Download CSV Report",
                    prediction_csv,
                    file_name="customer_churn_prediction.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with download_col2:

                pdf_path, pdf_error = build_pdf_report(
                    record
                )

                if pdf_path:

                    with open(
                        pdf_path,
                        "rb",
                    ) as pdf_file:

                        st.download_button(
                            "Download PDF Report",
                            pdf_file.read(),
                            file_name="customer_churn_report.pdf",
                            mime="application/pdf",
                            width="stretch",
                        )

                else:
                    st.info(pdf_error)

        else:

            html("""
            <div class="result-panel">

                <div style="color:#e9eef5;font-size:15px;font-weight:700;">
                    Prediction Result
                </div>

                <div style="
                    color:#657387;
                    font-size:10px;
                    line-height:1.7;
                    margin-top:12px;
                ">
                    Enter the customer information on the left,
                    then run the model to see churn probability,
                    risk level, signals, feature importance,
                    similar profiles, and report options.
                </div>

                <div class="gauge-card" style="margin-top:35px;">
                    <div
                        class="gauge"
                        style="
                            --progress:0deg;
                            background:conic-gradient(
                                #243243 0,
                                #1b2837 0
                            );
                        "
                    >
                        <div class="gauge-inner">
                            <div class="gauge-number">
                                —
                            </div>
                            <div class="gauge-label">
                                WAITING
                            </div>
                        </div>
                    </div>
                </div>

            </div>
            """)


# ============================================================
# EXPLORE DATA
# ============================================================

elif page == "Explore Data":

    html("""
    <div style="margin:27px 0 18px;">
        <div class="section-kicker">DATA EXPLORATION</div>
        <div class="section-title">Explore the Dataset</div>
        <div class="section-desc">
            Inspect quality, target distribution, categorical patterns,
            and numeric behavior.
        </div>
    </div>
    """)

    q1, q2, q3, q4 = st.columns(4)

    quality_cards = [
        ("Rows", f"{df.shape[0]:,}", "Dataset records"),
        ("Columns", f"{df.shape[1]}", "Original columns"),
        ("Missing", f"{MISSING_VALUES}", "Missing values"),
        ("Duplicates", f"{DUPLICATES}", "Duplicate rows"),
    ]

    for col, (label, value, desc) in zip(
        [q1, q2, q3, q4],
        quality_cards,
    ):
        with col:
            html(f"""
            <div class="metric-card">
                <div class="metric-icon">•</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div style="color:#5f6d80;font-size:7px;margin-top:3px;">
                    {desc}
                </div>
            </div>
            """)

    st.write("")

    left, right = st.columns(2)

    with left:

        html("""
        <div class="panel">
            <div class="panel-title">Churn Distribution</div>
            <div class="panel-desc">
                Observed distribution of churned and retained customers.
            </div>
        </div>
        """)

        values = [
            RETAINED,
            CHURNED,
        ]

        labels = [
            "No Churn",
            "Churn",
        ]

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            labels,
            values,
        )

        ax.set_ylabel(
            "Customers"
        )

        ax.set_title(
            "Customer Churn Distribution"
        )

        ax.grid(
            axis="y",
            alpha=.15,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)

    with right:

        html("""
        <div class="panel">
            <div class="panel-title">Churn Rate by Category</div>
            <div class="panel-desc">
                Select a categorical feature to inspect observed churn rates.
            </div>
        </div>
        """)

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if TARGET_COL in categorical_columns:
            categorical_columns.remove(
                TARGET_COL
            )

        if categorical_columns:

            selected_category = st.selectbox(
                "Categorical Feature",
                categorical_columns,
            )

            category_rates = (
                df.groupby(
                    selected_category
                )[TARGET_COL]
                .mean()
                .sort_values(
                    ascending=False
                )
                .head(15)
                * 100
            )

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.bar(
                category_rates.index.astype(str),
                category_rates.values,
            )

            ax.set_ylabel(
                "Churn Rate (%)"
            )

            ax.set_title(
                f"Churn Rate by {selected_category}"
            )

            ax.grid(
                axis="y",
                alpha=.15,
            )

            plt.xticks(
                rotation=35,
                ha="right",
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    html("""
    <div style="margin:26px 0 13px;">
        <div class="section-kicker">NUMERIC ANALYSIS</div>
        <div class="section-title">Customer Usage Patterns</div>
        <div class="section-desc">
            Compare the distribution of a numeric feature for churned and retained customers.
        </div>
    </div>
    """)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if TARGET_COL in numeric_columns:
        numeric_columns.remove(
            TARGET_COL
        )

    selected_numeric = st.selectbox(
        "Numeric Feature",
        numeric_columns,
    )

    fig, ax = plt.subplots(
        figsize=(11, 4.5)
    )

    for churn_value, label in [
        (0, "No Churn"),
        (1, "Churn"),
    ]:

        values = df.loc[
            df[TARGET_COL].astype(int) == churn_value,
            selected_numeric,
        ].dropna()

        ax.hist(
            values,
            bins=25,
            alpha=.55,
            label=label,
        )

    ax.set_title(
        selected_numeric
    )

    ax.set_xlabel(
        selected_numeric
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=.15,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">DATA QUALITY</div>
        <div class="section-title">Quality Check</div>
    </div>
    """)

    quality_df = pd.DataFrame({
        "Check": [
            "Missing Values",
            "Duplicate Rows",
            "Rows",
            "Columns",
        ],
        "Result": [
            MISSING_VALUES,
            DUPLICATES,
            df.shape[0],
            df.shape[1],
        ],
        "Status": [
            "PASS" if MISSING_VALUES == 0 else "REVIEW",
            "PASS" if DUPLICATES == 0 else "REVIEW",
            "INFO",
            "INFO",
        ],
    })

    st.dataframe(
        quality_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">RAW DATA</div>
        <div class="section-title">Dataset Preview</div>
    </div>
    """)

    st.dataframe(
        df.head(100),
        width="stretch",
        height=430,
    )

    st.download_button(
        "Download Dataset CSV",
        df.to_csv(index=False),
        file_name="churn_dataset.csv",
        mime="text/csv",
        width="stretch",
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    html("""
    <div style="margin:27px 0 18px;">
        <div class="section-kicker">BUSINESS INTELLIGENCE</div>
        <div class="section-title">Where is Churn Showing Up?</div>
        <div class="section-desc">
            Descriptive observations from the customer dataset.
            These patterns do not by themselves establish causation.
        </div>
    </div>
    """)

    b1, b2, b3 = st.columns(3)

    business_metrics = [
        (
            "Churned Customers",
            f"{CHURNED:,}",
            f"{CHURN_RATE:.1f}% of dataset",
        ),
        (
            "Retained Customers",
            f"{RETAINED:,}",
            f"{100 - CHURN_RATE:.1f}% of dataset",
        ),
        (
            "Support Calls Median",
            f"{df['Customer service calls'].median():.1f}",
            "calls",
        ),
    ]

    for col, (label, value, sub) in zip(
        [b1, b2, b3],
        business_metrics,
    ):
        with col:
            html(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div style="color:#607086;font-size:7px;margin-top:4px;">
                    {sub}
                </div>
            </div>
            """)

    st.write("")

    dimension_options = [
        "International plan",
        "Voice mail plan",
        "Customer service calls",
        "State",
    ]

    selected_dimension = st.selectbox(
        "Business Dimension",
        [
            x
            for x in dimension_options
            if x in df.columns
        ],
    )

    grouped = (
        df.groupby(
            selected_dimension
        )[TARGET_COL]
        .agg(
            Customers="count",
            Churn_Rate="mean",
        )
    )

    grouped["Churn_Rate"] *= 100

    grouped = grouped.sort_values(
        "Churn_Rate",
        ascending=False,
    )

    st.dataframe(
        grouped.style.format(
            {
                "Churn_Rate": "{:.1f}%"
            }
        ),
        width="stretch",
    )

    fig, ax = plt.subplots(
        figsize=(11, 4.5)
    )

    plot_data = grouped.head(15)

    ax.bar(
        plot_data.index.astype(str),
        plot_data["Churn_Rate"],
    )

    ax.set_ylabel(
        "Observed Churn Rate (%)"
    )

    ax.set_title(
        f"Observed Churn Rate by {selected_dimension}"
    )

    ax.grid(
        axis="y",
        alpha=.15,
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)

    if len(grouped) >= 2:

        highest_group = str(
            grouped.index[0]
        )

        highest_rate = float(
            grouped.iloc[0]["Churn_Rate"]
        )

        lowest_group = str(
            grouped.index[-1]
        )

        lowest_rate = float(
            grouped.iloc[-1]["Churn_Rate"]
        )

        html(f"""
        <div class="data-note">
            <strong style="color:#dbe5ef;">
                Dataset observation:
            </strong>
            The selected dimension shows an observed churn rate of
            <strong style="color:#fff;">
                {highest_rate:.1f}%
            </strong>
            for <strong style="color:#fff;">
                {highest_group}
            </strong>
            and <strong style="color:#fff;">
                {lowest_rate:.1f}%
            </strong>
            for <strong style="color:#fff;">
                {lowest_group}
            </strong>.
            This is a descriptive comparison within this dataset, not a causal conclusion.
        </div>
        """)

    # --------------------------------------------------------
    # INTERNATIONAL / SUPPORT
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        html("""
        <div class="panel">
            <div class="panel-title">
                International Plan
            </div>
            <div class="panel-desc">
                Observed churn rate by international plan status.
            </div>
        </div>
        """)

        if "International plan" in df.columns:

            plan_stats = (
                df.groupby(
                    "International plan"
                )[TARGET_COL]
                .mean()
                * 100
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            ax.bar(
                plan_stats.index.astype(str),
                plan_stats.values,
            )

            ax.set_ylabel(
                "Churn Rate (%)"
            )

            ax.grid(
                axis="y",
                alpha=.15,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

    with right:

        html("""
        <div class="panel">
            <div class="panel-title">
                Customer Service Calls
            </div>
            <div class="panel-desc">
                Observed churn rate across support-call counts.
            </div>
        </div>
        """)

        if "Customer service calls" in df.columns:

            support_stats = (
                df.groupby(
                    "Customer service calls"
                )[TARGET_COL]
                .mean()
                .head(12)
                * 100
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            ax.plot(
                support_stats.index,
                support_stats.values,
                marker="o",
            )

            ax.set_xlabel(
                "Customer Service Calls"
            )

            ax.set_ylabel(
                "Churn Rate (%)"
            )

            ax.grid(
                alpha=.15
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)


# ============================================================
# MODEL LAB
# ============================================================

elif page == "Model Lab":

    html("""
    <div style="margin:27px 0 18px;">
        <div class="section-kicker">MODEL LAB</div>
        <div class="section-title">Understand the Machine Learning Pipeline</div>
        <div class="section-desc">
            Model comparison, live evaluation, ROC analysis, confusion matrix,
            feature importance, threshold analysis, and deployment details.
        </div>
    </div>
    """)

    html("""
    <div class="panel">
        <div class="panel-title">Model Development Pipeline</div>
        <div class="panel-desc">
            The project workflow exposed as a portfolio artifact.
        </div>

        <div class="timeline" style="margin-top:17px;">

            <div class="timeline-step">
                <div class="timeline-num">1</div>
                <div class="timeline-name">EDA</div>
                <div class="timeline-desc">Explore</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">2</div>
                <div class="timeline-name">Preprocessing</div>
                <div class="timeline-desc">Encode</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">3</div>
                <div class="timeline-name">Split</div>
                <div class="timeline-desc">Stratified</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">4</div>
                <div class="timeline-name">Models</div>
                <div class="timeline-desc">Compare</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">5</div>
                <div class="timeline-name">Tuning</div>
                <div class="timeline-desc">Grid Search</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">6</div>
                <div class="timeline-name">Threshold</div>
                <div class="timeline-desc">Analyze</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">7</div>
                <div class="timeline-name">Deploy</div>
                <div class="timeline-desc">Streamlit</div>
            </div>

        </div>
    </div>
    """)

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    html("""
    <div style="margin:27px 0 13px;">
        <div class="section-kicker">MODEL COMPARISON</div>
        <div class="section-title">Training Results</div>
        <div class="section-desc">
            Results exported from the model training workflow.
        </div>
    </div>
    """)

    comparison_df = load_optional_csv(
        COMPARISON_PATH
    )

    if comparison_df is not None:

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
        )

        metric_options = [
            c
            for c in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ]
            if c in comparison_df.columns
        ]

        if metric_options:

            selected_metric = st.selectbox(
                "Comparison Metric",
                metric_options,
            )

            fig, ax = plt.subplots(
                figsize=(11, 4.5)
            )

            values = pd.to_numeric(
                comparison_df[selected_metric],
                errors="coerce",
            )

            ax.bar(
                comparison_df.iloc[:, 0].astype(str),
                values,
            )

            ax.set_ylabel(
                selected_metric
            )

            ax.set_title(
                f"Model Comparison — {selected_metric}"
            )

            ax.grid(
                axis="y",
                alpha=.15,
            )

            plt.xticks(
                rotation=25,
                ha="right",
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

    else:
        st.warning(
            "model_comparison_results.csv was not found."
        )

    # --------------------------------------------------------
    # LIVE EVALUATION
    # --------------------------------------------------------

    html("""
    <div style="margin:27px 0 13px;">
        <div class="section-kicker">LIVE EVALUATION</div>
        <div class="section-title">Saved Model Performance</div>
        <div class="section-desc">
            Metrics calculated from the saved model using the same 80/20
            stratified split configuration used by the project.
        </div>
    </div>
    """)

    try:

        X = df.drop(
            columns=[TARGET_COL]
        )

        y = df[TARGET_COL].astype(int)

        X_encoded = pd.get_dummies(
            X,
            drop_first=True,
        )

        model_features = get_model_features(
            model
        )

        X_live = pd.DataFrame(
            0.0,
            index=X_encoded.index,
            columns=model_features,
        )

        for col in X_encoded.columns:

            if col in X_live.columns:
                X_live[col] = X_encoded[col]

        X_train, X_test, y_train, y_test = train_test_split(
            X_live,
            y,
            test_size=.2,
            random_state=42,
            stratify=y,
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        metrics = [
            (
                "Accuracy",
                accuracy_score(
                    y_test,
                    y_pred,
                ),
            ),
            (
                "Precision",
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
            ),
            (
                "Recall",
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
            ),
            (
                "F1 Score",
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
            ),
            (
                "ROC-AUC",
                roc_auc_score(
                    y_test,
                    y_prob,
                ),
            ),
        ]

        metric_cols = st.columns(5)

        for col, (label, value) in zip(
            metric_cols,
            metrics,
        ):
            with col:
                html(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">
                        {value * 100:.1f}%
                    </div>
                </div>
                """)

    except Exception as error:

        st.warning(
            "Live evaluation could not be calculated."
        )

        st.code(
            str(error)
        )

    # --------------------------------------------------------
    # ROC + CONFUSION
    # --------------------------------------------------------

    roc_col, cm_col = st.columns(2)

    with roc_col:

        html("""
        <div class="panel">
            <div class="panel-title">ROC Curve</div>
            <div class="panel-desc">
                True-positive rate versus false-positive rate.
            </div>
        </div>
        """)

        try:

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob,
            )

            auc_value = roc_auc_score(
                y_test,
                y_prob,
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"AUC = {auc_value:.3f}",
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                linewidth=1,
            )

            ax.set_xlabel(
                "False Positive Rate"
            )

            ax.set_ylabel(
                "True Positive Rate"
            )

            ax.set_title(
                "ROC Curve"
            )

            ax.legend(
                loc="lower right"
            )

            ax.grid(
                alpha=.15
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

        except Exception as error:
            st.info(
                f"ROC curve unavailable: {error}"
            )

    with cm_col:

        html("""
        <div class="panel">
            <div class="panel-title">Confusion Matrix</div>
            <div class="panel-desc">
                Actual versus predicted customer churn.
            </div>
        </div>
        """)

        try:

            cm = confusion_matrix(
                y_test,
                y_pred,
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            ax.imshow(
                cm,
                interpolation="nearest",
            )

            ax.set_title(
                "Confusion Matrix"
            )

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            ax.set_xticks(
                [0, 1]
            )

            ax.set_yticks(
                [0, 1]
            )

            ax.set_xticklabels(
                ["No Churn", "Churn"]
            )

            ax.set_yticklabels(
                ["No Churn", "Churn"]
            )

            for i in range(2):
                for j in range(2):

                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

        except Exception as error:
            st.info(
                f"Confusion matrix unavailable: {error}"
            )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">EXPLAINABILITY</div>
        <div class="section-title">Feature Importance</div>
        <div class="section-desc">
            Global feature importance from the trained Random Forest.
        </div>
    </div>
    """)

    importance_df = load_optional_csv(
        IMPORTANCE_PATH
    )

    if importance_df is not None:

        feature_col = find_column(
            importance_df,
            ["feature", "features", "feature name"],
        )

        score_col = find_column(
            importance_df,
            [
                "importance",
                "feature importance",
                "importance score",
            ],
        )

        if feature_col and score_col:

            fi = importance_df[
                [feature_col, score_col]
            ].copy()

            fi.columns = [
                "Feature",
                "Importance",
            ]

        else:
            fi = pd.DataFrame()

    elif hasattr(
        model,
        "feature_importances_",
    ):

        fi = pd.DataFrame({
            "Feature": get_model_features(model),
            "Importance": model.feature_importances_,
        })

    else:
        fi = pd.DataFrame()

    if not fi.empty:

        fi["Importance"] = pd.to_numeric(
            fi["Importance"],
            errors="coerce",
        )

        fi = fi.dropna().sort_values(
            "Importance",
            ascending=False,
        )

        display_count = st.slider(
            "Features to display",
            min_value=5,
            max_value=min(
                20,
                len(fi),
            ),
            value=min(
                10,
                len(fi),
        ),
        )

        top_fi = fi.head(
            display_count
        ).sort_values(
            "Importance"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            top_fi["Feature"],
            top_fi["Importance"],
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_title(
            "Global Feature Importance"
        )

        ax.grid(
            axis="x",
            alpha=.15,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)

        st.dataframe(
            fi.head(
                display_count
            ),
            width="stretch",
            hide_index=True,
        )

        html("""
        <div class="data-note">
            Feature importance describes how the trained Random Forest uses
            features across the model. It should not be interpreted as proof
            that a feature causes churn.
        </div>
        """)

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">THRESHOLD ANALYSIS</div>
        <div class="section-title">Probability Thresholds</div>
        <div class="section-desc">
            Review how precision, recall, and F1 change at different thresholds.
        </div>
    </div>
    """)

    threshold_df = load_optional_csv(
        THRESHOLD_PATH
    )

    if threshold_df is not None:

        st.dataframe(
            threshold_df,
            width="stretch",
            hide_index=True,
        )

        threshold_col = find_column(
            threshold_df,
            ["threshold"],
        )

        f1_col = find_column(
            threshold_df,
            ["f1", "f1 score"],
        )

        if threshold_col and f1_col:

            fig, ax = plt.subplots(
                figsize=(10, 4.5)
            )

            ax.plot(
                threshold_df[threshold_col],
                threshold_df[f1_col],
                marker="o",
            )

            ax.set_xlabel(
                "Threshold"
            )

            ax.set_ylabel(
                "F1 Score"
            )

            ax.set_title(
                "Threshold vs F1 Score"
            )

            ax.grid(
                alpha=.15,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

    else:
        st.info(
            "threshold_results.csv was not found."
        )

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">DEPLOYMENT</div>
        <div class="section-title">Final Model Details</div>
    </div>

    <div class="panel">

        <div class="signal-row">
            <span class="signal-name">Model</span>
            <span class="signal-value">Random Forest Classifier</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">Task</span>
            <span class="signal-value">Binary Classification</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">Preprocessing</span>
            <span class="signal-value">One-Hot Encoding</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">Train/Test Split</span>
            <span class="signal-value">80 / 20 — Stratified</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">Random State</span>
            <span class="signal-value">42</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">Deployment</span>
            <span class="signal-value">Streamlit</span>
        </div>

    </div>
    """)


# ============================================================
# HISTORY
# ============================================================

elif page == "Prediction History":

    # Prediction History always reflects the CSV on disk — it is the
    # source of truth, not session_state. Any read failure is shown
    # to the user, never silently treated as "0 predictions".
    _history_records, _history_error = load_history_from_disk()

    if _history_error is not None:
        st.error(
            "Could not read prediction_history.csv — showing the "
            "error instead of pretending history is empty."
        )
        st.code(str(_history_error))
    else:
        st.session_state.history = _history_records
        st.session_state.history_load_error = None

    html("""
    <div style="margin:27px 0 18px;">
        <div class="section-kicker">PREDICTION HISTORY</div>
        <div class="section-title">Prediction Activity</div>
        <div class="section-desc">
            Saved predictions are stored locally in prediction_history.csv.
        </div>
    </div>
    """)

    history = st.session_state.history

    total_predictions = len(history)

    churn_predictions = sum(
        str(item.get("Prediction", "")) == "Churn"
        for item in history
    )

    stay_predictions = (
        total_predictions -
        churn_predictions
    )

    if total_predictions:
        avg_probability = sum(
            float(item.get("Churn Probability", 0.0) or 0.0)
            for item in history
        ) / total_predictions
    else:
        avg_probability = 0.0

    h1, h2, h3, h4 = st.columns(4)

    cards = [
        (
            "Total Predictions",
            total_predictions,
        ),
        (
            "Churn Predictions",
            churn_predictions,
        ),
        (
            "No Churn Predictions",
            stay_predictions,
        ),
        (
            "Average Churn Probability",
            f"{avg_probability:.2f}%",
        ),
    ]

    for col, (label, value) in zip(
        [h1, h2, h3, h4],
        cards,
    ):
        with col:
            html(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """)

    st.write("")

    refresh_col, clear_space = st.columns([1, 3])

    with refresh_col:
        if st.button(
            "Refresh History",
            key="refresh_history_btn",
            width="stretch",
        ):
            _refreshed_records, _refresh_error = load_history_from_disk()

            if _refresh_error is not None:
                st.session_state.history_load_error = str(_refresh_error)
            else:
                st.session_state.history = _refreshed_records
                st.session_state.history_load_error = None

            st.rerun()

    st.write("")

    if history:

        history_df = pd.DataFrame(
            history
        )

        # Newest predictions first.
        if "Timestamp" in history_df.columns:
            history_df = history_df.sort_values(
                "Timestamp",
                ascending=False,
            ).reset_index(drop=True)
        else:
            history_df = history_df.iloc[::-1].reset_index(drop=True)

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "Download History CSV",
                history_df.to_csv(
                    index=False
                ),
                file_name="prediction_history.csv",
                mime="text/csv",
                width="stretch",
            )

        with d2:

            if st.button(
                "Clear Prediction History",
                width="stretch",
            ):

                _clear_error = None

                try:
                    if os.path.exists(HISTORY_PATH):
                        os.remove(HISTORY_PATH)
                except Exception as error:
                    _clear_error = error

                if _clear_error is None:
                    st.session_state.history = []
                    st.rerun()
                else:
                    st.error(
                        f"Could not clear history file: {_clear_error}"
                    )

    else:

        html("""
        <div class="panel">
            <div class="panel-title">
                No Saved Predictions
            </div>
            <div class="panel-desc">
                Run a prediction and use "Save Prediction to History"
                to build your local prediction log.
            </div>
        </div>
        """)

        st.write("")

        if st.button(
            "Go to Predict",
            key="history_empty_go_to_predict",
        ):
            st.session_state.nav_page = "Predict"
            st.rerun()


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    html("""
    <div style="margin:27px 0 18px;">
        <div class="section-kicker">ABOUT THE PROJECT</div>
        <div class="section-title">Customer Churn Prediction</div>
        <div class="section-desc">
            A complete machine learning project presented as an interactive portfolio application.
        </div>
    </div>
    """)

    left, right = st.columns(
        [1.35, .85]
    )

    with left:

        html("""
        <div class="panel">

            <div class="panel-title">
                Project Goal
            </div>

            <div class="panel-desc" style="margin-top:12px;">
                The project uses supervised machine learning to estimate
                whether a customer is likely to churn based on customer
                profile, plan information, usage behavior, and support activity.
            </div>

            <div class="data-note">
                The application is designed as a demonstration of the
                complete workflow: data preparation, exploratory analysis,
                model training, comparison, tuning, evaluation, and deployment.
            </div>

        </div>
        """)

    with right:

        html(f"""
        <div class="developer-card">

            <div class="developer-kicker">
                ABOUT THE DEVELOPER
            </div>

            <div class="developer-name">
                {OWNER}
            </div>

            <div class="developer-links">

                <a
                    class="developer-link"
                    href="{LINKEDIN_URL}"
                    target="_blank"
                >
                    LinkedIn
                </a>

                <a
                    class="developer-link"
                    href="{GITHUB_URL}"
                    target="_blank"
                >
                    GitHub
                </a>

            </div>

        </div>
        """)

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">END-TO-END WORKFLOW</div>
        <div class="section-title">How the project was built</div>
    </div>

    <div class="panel">

        <div class="timeline">

            <div class="timeline-step">
                <div class="timeline-num">1</div>
                <div class="timeline-name">Load</div>
                <div class="timeline-desc">Dataset</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">2</div>
                <div class="timeline-name">EDA</div>
                <div class="timeline-desc">Explore</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">3</div>
                <div class="timeline-name">Quality</div>
                <div class="timeline-desc">Validate</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">4</div>
                <div class="timeline-name">Encode</div>
                <div class="timeline-desc">Features</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">5</div>
                <div class="timeline-name">Train</div>
                <div class="timeline-desc">Models</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">6</div>
                <div class="timeline-name">Tune</div>
                <div class="timeline-desc">Optimize</div>
            </div>

            <div class="timeline-step">
                <div class="timeline-num">7</div>
                <div class="timeline-name">Deploy</div>
                <div class="timeline-desc">Streamlit</div>
            </div>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">TECH STACK</div>
        <div class="section-title">Technologies Used</div>
    </div>
    """)

    tech_cols = st.columns(7)

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Scikit-learn",
        "Streamlit",
        "Joblib/Pickle",
    ]

    for col, technology in zip(
        tech_cols,
        technologies,
    ):
        with col:
            html(f"""
            <div class="metric-card">
                <div class="metric-label">TECHNOLOGY</div>
                <div style="
                    color:#edf2f7;
                    font-family:'Space Grotesk',sans-serif;
                    font-size:14px;
                    font-weight:700;
                    margin-top:7px;
                ">
                    {technology}
                </div>
            </div>
            """)

    # --------------------------------------------------------
    # ASSETS
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">PROJECT ASSETS</div>
        <div class="section-title">Files used by the application</div>
    </div>
    """)

    assets = [
        ("Dataset", "churn_dataset.csv", DATA_PATH),
        ("Trained Model", "customer_churn_model.pkl", MODEL_PATH),
        (
            "Model Comparison",
            "model_comparison_results.csv",
            COMPARISON_PATH,
        ),
        (
            "Threshold Results",
            "threshold_results.csv",
            THRESHOLD_PATH,
        ),
        (
            "Feature Importance",
            "feature_importance.csv",
            IMPORTANCE_PATH,
        ),
    ]

    for label, filename, path in assets:

        exists = os.path.exists(
            path
        )

        status = (
            "AVAILABLE"
            if exists
            else "MISSING"
        )

        html(f"""
        <div class="signal-row">
            <span class="signal-name">
                {label}
            </span>

            <span class="signal-value">
                {filename} — {status}
            </span>
        </div>
        """)

    # --------------------------------------------------------
    # PROJECT STRUCTURE
    # --------------------------------------------------------

    html("""
    <div style="margin:28px 0 13px;">
        <div class="section-kicker">STRUCTURE</div>
        <div class="section-title">Project Architecture</div>
    </div>

    <div class="panel">

        <div class="signal-row">
            <span class="signal-name">1. Dataset</span>
            <span class="signal-value">Customer churn records</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">2. EDA</span>
            <span class="signal-value">Distribution & relationship analysis</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">3. Preprocessing</span>
            <span class="signal-value">One-hot encoding</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">4. Modeling</span>
            <span class="signal-value">Logistic Regression, Random Forest, Gradient Boosting</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">5. Tuning</span>
            <span class="signal-value">Random Forest hyperparameter search</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">6. Evaluation</span>
            <span class="signal-value">Accuracy, Precision, Recall, F1, ROC-AUC</span>
        </div>

        <div class="signal-row">
            <span class="signal-name">7. Deployment</span>
            <span class="signal-value">Interactive Streamlit application</span>
        </div>

    </div>
    """)


# ============================================================
# FOOTER — PERSISTENT OWNERSHIP
# ============================================================

with st.container(key="footer_wrap"):

    foot_col1, foot_col2, foot_col3, foot_col4 = st.columns(
        [1.5, 1, 1, 1],
        gap="large",
    )

    with foot_col1:
        html("""
        <div class="footer-brand">
            ChurnAI
        </div>

        <div class="footer-desc">
            Customer Churn Prediction dashboard combining
            data exploration, machine learning, evaluation,
            explainability, and business-oriented insights.
        </div>
        """)

    with foot_col2:
        html("""
        <div class="footer-heading">
            PROJECT
        </div>

        <div class="footer-item">
            Customer Churn Prediction
        </div>

        <div class="footer-item">
            Random Forest
        </div>

        <div class="footer-item">
            Streamlit
        </div>
        """)

    with foot_col3:
        html("""
        <div class="footer-heading">
            QUICK LINKS
        </div>
        """)

        footer_links = [
            "Model Lab",
            "Business Insights",
            "Prediction History",
        ]

        for link_name in footer_links:
            if st.button(
                link_name,
                key=f"footer_nav_{link_name.replace(' ', '_')}",
            ):
                st.session_state.nav_page = link_name
                st.rerun()

    with foot_col4:
        html(f"""
        <div class="footer-heading">
            DEVELOPER
        </div>

        <div class="footer-item">
            {OWNER}
        </div>

        <div style="display:flex;gap:7px;margin-top:10px;">

            <a
                class="icon-link"
                href="{LINKEDIN_URL}"
                target="_blank"
            >
                in
            </a>

            <a
                class="icon-link"
                href="{GITHUB_URL}"
                target="_blank"
            >
                GH
            </a>

        </div>
        """)

    html(f"""
    <div class="footer-bottom">

        <div>
            © 2026 {OWNER}. All Rights Reserved.
        </div>

        <div>
            Built with Streamlit • Powered by Machine Learning
        </div>

    </div>
    """)