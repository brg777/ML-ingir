"""Скоринг SOC-датасета моделью курса.

Воспроизводит тестовый сплит (сверка с model_card.md), скорит
эксплуатационный период и сохраняет scored.parquet для дашборда.

Использует ТОЛЬКО API курса (course_ml). Без task_2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # корень soc-trust-analysis/

# пакет курса — на уровень выше, соседняя папка
import os
COURSE_DIR = Path(os.environ.get("COURSE_DIR", HERE.parent / "IAVD_3.7_ML_team"))
if not (COURSE_DIR / "course_ml").exists():
    raise SystemExit(f"course_ml not found under {COURSE_DIR}")
sys.path.insert(0, str(COURSE_DIR))

from course_ml.inference import CourseModel
from course_ml import data as data_mod
from course_ml import train as train_mod

DATA_PATH = HERE / "data" / "iavd_ml_dataset.parquet"
OUT_PATH  = HERE / "data" / "scored.parquet"

# Поля-срезы, которые понадобятся для вопросов руководителя SOC
SLICE_FIELDS = [
    "Наименование",              # правило корреляции
    "Критичность",
    "Организация",
    "Пиктограмма IRP",
    "Источник инцидента",
    "Тип инцидента",
    "Вендор из сием",
    "Конвейер SIEM",
]

# Эталон из model_card.md
PUBLISHED_TEST = dict(tp=390, fp=107, fn=24, tn=2652)


def apply_train_filters(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Повторяет фильтры data.build: verdict_in, rule_not_null, vendor_not_in."""
    dcfg = cfg["dataset"]
    mask = df[dcfg["target"]["column"]].isin(dcfg["filters"]["verdict_in"])
    if dcfg["filters"].get("rule_not_null"):
        mask &= df["Правило корреляции SIEM"].notna()
    for vendor in dcfg["filters"].get("vendor_not_in", []) or []:
        mask &= ~df["Вендор из сием"].eq(vendor)
    return mask


def window_mask(df: pd.DataFrame, cfg: dict, start: str, stop: str | None) -> pd.Series:
    """Маска по времени создания карточки."""
    col = cfg["dataset"]["time"]["column"]
    created = pd.to_datetime(df[col], format="ISO8601", utc=True)
    m = created >= pd.Timestamp(start, tz="UTC")
    if stop is not None:
        m &= created < pd.Timestamp(stop, tz="UTC") + pd.Timedelta(days=1)
    return m


def outcome_from(y_true: np.ndarray, proba: np.ndarray, tau: float) -> np.ndarray:
    pred = (proba >= tau).astype(int)
    return np.select(
        [(y_true == 1) & (pred == 1),
         (y_true == 0) & (pred == 1),
         (y_true == 1) & (pred == 0)],
        ["TP", "FP", "FN"], "TN")


def confusion(outcome: np.ndarray) -> dict:
    return dict(tp=int((outcome == "TP").sum()), fp=int((outcome == "FP").sum()),
                fn=int((outcome == "FN").sum()), tn=int((outcome == "TN").sum()))


def build_slice_frame(src: pd.DataFrame, cfg: dict, proba: np.ndarray,
                      y_true: np.ndarray, outcome: np.ndarray, split: str) -> pd.DataFrame:
    created = pd.to_datetime(src[cfg["dataset"]["time"]["column"]],
                             format="ISO8601", utc=True)
    out = pd.DataFrame({
        "EntityId": src["EntityId"].values,
        "created": created.values,
        "y_true": y_true,
        "proba": proba,
        "outcome": outcome,
        "split": split,
    })
    for f in SLICE_FIELDS:
        if f in src.columns:
            out[f] = src[f].values
    return out


def main() -> None:
    print(f"Course dir: {COURSE_DIR}")
    model = CourseModel.from_registry("lgbm_gate1", 1)
    cfg, tau = model.cfg, model.threshold
    print(f"Threshold τ = {tau}")
    print(f"Training window: {cfg['dataset']['window']['from']} .. {cfg['dataset']['window']['to']}")

    print(f"\nReading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    print(f"Full dataset: {df.shape}")

    # ---------- 1. Воспроизводим тестовый сплит ----------
    print("\n=== Reproducing test split ===")
    train_mask = apply_train_filters(df, cfg) & window_mask(
        df, cfg, cfg["dataset"]["window"]["from"], cfg["dataset"]["window"]["to"])
    train_df = df[train_mask].copy()
    print(f"Training-period rows (after filters): {len(train_df)}")

    prepared, y, created = data_mod.prepare(train_df, cfg)
    parts = train_mod.make_split(prepared, y, created, cfg)
    test_idx = parts["test"]
    test_df = train_df.loc[test_idx]
    print(f"Test rows: {len(test_df)} (expected ~3173)")

    test_y = y.loc[test_idx].to_numpy()
    test_proba = model.predict_proba(test_df)
    test_outcome = outcome_from(test_y, test_proba, tau)
    cm = confusion(test_outcome)
    print(f"Test confusion: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    ok = all(cm[k] == PUBLISHED_TEST[k] for k in cm)
    print(f"Matches model_card.md 390/107/24/2652: {ok}")
    if not ok:
        print("WARN: split differs from published. Check filters/window/seed.")

    test_out = build_slice_frame(test_df, cfg, test_proba, test_y, test_outcome, "test")

    # ---------- 2. Скорим эксплуатационный период ----------
    print("\n=== Scoring exploitation period ===")
    # окно эксплуатации: с 2025-09-01 по конец датасета
    oper_start = "2025-09-01"
    oper_mask = window_mask(df, cfg, oper_start, None)
    # фильтры качества, но БЕЗ verdict_in — скóрим всё, оцениваем только размеченное
    if cfg["dataset"]["filters"].get("rule_not_null"):
        oper_mask &= df["Правило корреляции SIEM"].notna()
    for vendor in cfg["dataset"]["filters"].get("vendor_not_in", []) or []:
        oper_mask &= ~df["Вендор из сием"].eq(vendor)
    oper_df = df[oper_mask].copy()
    print(f"Exploitation rows to score: {len(oper_df)} (expected ~20581)")

    oper_proba = model.predict_proba(oper_df)

    verdict_col = cfg["dataset"]["target"]["column"]
    positive = cfg["dataset"]["target"]["positive"]
    labeled_mask = oper_df[verdict_col].isin(cfg["dataset"]["filters"]["verdict_in"])
    print(f"Labeled (verdict in {cfg['dataset']['filters']['verdict_in']}): "
          f"{int(labeled_mask.sum())} (expected ~19924)")

    oper_y_all = oper_df[verdict_col].eq(positive).astype(int).to_numpy()
    oper_outcome_all = outcome_from(oper_y_all, oper_proba, tau)

    # в scored кладём только размеченное — для дашборда нужен честный y_true
    oper_labeled = oper_df[labeled_mask]
    oper_out = build_slice_frame(
        oper_labeled, cfg,
        oper_proba[labeled_mask.to_numpy()],
        oper_y_all[labeled_mask.to_numpy()],
        oper_outcome_all[labeled_mask.to_numpy()],
        "oper")
    cm_o = confusion(oper_outcome_all[labeled_mask.to_numpy()])
    print(f"Exploitation confusion (labeled): TP={cm_o['tp']} FP={cm_o['fp']} "
          f"FN={cm_o['fn']} TN={cm_o['tn']}")

    # ---------- 3. Сохраняем ----------
    scored = pd.concat([test_out, oper_out], ignore_index=True)
    scored.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"\nSaved {OUT_PATH}: {scored.shape}")
    print(f"By split: {scored['split'].value_counts().to_dict()}")
    print(f"By outcome: {scored['outcome'].value_counts().to_dict()}")
    print("\nColumns:", scored.columns.tolist())


if __name__ == "__main__":
    main()

