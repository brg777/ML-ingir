"""Агрегаты для пункта 2: инференс модели lgbm_gate1__v001.

Все функции принимают scored.parquet (или raw + scored) и возвращают
структуры, готовые для построения графиков. Визуализации здесь нет.

Соглашения:
- test = приёмка (3 173 карточки, июнь–август 2025, стратифицированный сплит)
- oper = эксплуатация (19 924 размеченные карточки, сентябрь–ноябрь 2025)
- outcome = TP/FP/FN/TN по рабочему порогу τ
- y_true = 1 если Вердикт == "Решён", 0 если "False Positive"
- ноябрь в oper неполный: данные до 2025-11-20

Рабочий порог τ = 0.019258739887821837 (из model_card.md), не меняется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------- константы ----------

THRESHOLD = 0.019258739887821837
OPER_END = "2025-11-20"
MIN_POS_FOR_SLICE = 50    # минимум реальных инцидентов в срезе для FN-rate
MIN_NEG_FOR_SLICE = 100   # минимум FP в срезе для FP-rate


# ---------- утилиты ----------

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95%-й интервал Уилсона для доли k/n. Устойчив к малым n и крайним p."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _outcome_counts(df: pd.DataFrame) -> dict:
    """TP/FP/FN/TN + доли: recall, fn_rate, precision, fp_rate, enrichment."""
    tp = int((df["outcome"] == "TP").sum())
    fp = int((df["outcome"] == "FP").sum())
    fn = int((df["outcome"] == "FN").sum())
    tn = int((df["outcome"] == "TN").sum())
    n_pos, n_neg, n = tp + fn, tn + fp, len(df)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "n_pos": n_pos, "n_neg": n_neg,
        "recall": tp / n_pos if n_pos else float("nan"),
        "fn_rate": fn / n_pos if n_pos else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "fp_rate": fp / n_neg if n_neg else float("nan"),
        "enrichment": tp / (tp + fp) if (tp + fp) else float("nan"),
        "positive_share": n_pos / n if n else float("nan"),
    }


# ---------- вопрос 1: снимок test vs oper ----------

def degradation_snapshot(scored: pd.DataFrame) -> pd.DataFrame:
    """test vs oper по 4 метрикам + 95% Wilson CI.

    Возвращает длинную таблицу: metric | split | value | ci_low | ci_high | k | n
    metric ∈ {recall, fn_rate, precision, fp_rate}.
    """
    rows = []
    for split_name in ("test", "oper"):
        c = _outcome_counts(scored[scored["split"] == split_name])
        for metric, (k, n) in {
            "recall": (c["tp"], c["n_pos"]),
            "fn_rate": (c["fn"], c["n_pos"]),
            "precision": (c["tp"], c["tp"] + c["fp"]),
            "fp_rate": (c["fp"], c["n_neg"]),
        }.items():
            lo, hi = wilson_interval(k, n)
            rows.append({
                "metric": metric, "split": split_name,
                "value": c[metric], "ci_low": lo, "ci_high": hi,
                "k": k, "n": n,
            })
    return pd.DataFrame(rows)


# ---------- вопрос 2: FN-rate по месяцам ----------

def fn_rate_by_month(scored: pd.DataFrame) -> pd.DataFrame:
    """FN-rate по месяцам эксплуатации + 95% CI + ориентир приёмки.

    Возвращает: month | n_pos | fn | fn_rate | ci_low | ci_high | is_partial
    """
    oper = scored[scored["split"] == "oper"].copy()
    oper["month"] = pd.to_datetime(oper["created"]).dt.strftime("%Y-%m")
    test = scored[scored["split"] == "test"]
    test_fn = int((test["outcome"] == "FN").sum())
    test_pos = int(test["outcome"].isin(["TP", "FN"]).sum())
    test_lo, test_hi = wilson_interval(test_fn, test_pos)

    rows = []
    for month, df in oper.groupby("month"):
        n_pos = int(df["outcome"].isin(["TP", "FN"]).sum())
        fn = int((df["outcome"] == "FN").sum())
        lo, hi = wilson_interval(fn, n_pos)
        rows.append({
            "month": month, "n_pos": n_pos, "fn": fn,
            "fn_rate": fn / n_pos if n_pos else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "is_partial": month == OPER_END[:7],
            "test_fn_rate": test_fn / test_pos,
            "test_ci_low": test_lo, "test_ci_high": test_hi,
        })
    return pd.DataFrame(rows)


# ---------- вопрос 3: обогащение и FP по месяцам ----------

def enrichment_by_month(scored: pd.DataFrame) -> pd.DataFrame:
    """Доля реальных инцидентов среди приоритетных + число FP по месяцам.

    Возвращает: month | n_priority | tp | fp | enrichment | fp_per_day | is_partial
    """
    oper = scored[scored["split"] == "oper"].copy()
    oper["month"] = pd.to_datetime(oper["created"]).dt.strftime("%Y-%m")
    test = scored[scored["split"] == "test"]
    test_priority = test[test["outcome"].isin(["TP", "FP"])]
    test_enr = (test_priority["outcome"] == "TP").mean() if len(test_priority) else float("nan")

    rows = []
    for month, df in oper.groupby("month"):
        prio = df[df["outcome"].isin(["TP", "FP"])]
        tp = int((prio["outcome"] == "TP").sum())
        fp = int((prio["outcome"] == "FP").sum())
        n_days = (pd.to_datetime(df["created"]).max()
                  - pd.to_datetime(df["created"]).min()).days + 1
        rows.append({
            "month": month, "n_priority": tp + fp, "tp": tp, "fp": fp,
            "enrichment": tp / (tp + fp) if (tp + fp) else float("nan"),
            "fp_per_day": fp / n_days if n_days else float("nan"),
            "is_partial": month == OPER_END[:7],
            "test_enrichment": test_enr,
        })
    return pd.DataFrame(rows)


# ---------- вопрос 4: ошибки по срезам ----------

def error_by_slice(scored: pd.DataFrame, field: str,
                   min_pos: int = MIN_POS_FOR_SLICE) -> pd.DataFrame:
    """FN-rate и FP-rate по значениям одного поля-среза, только n >= min_pos.

    Возвращает: value | n_pos | fn_rate | ci_low | ci_high | n_neg | fp_rate | ...
    Отсортировано по n_pos по убыванию.
    """
    oper = scored[scored["split"] == "oper"]
    rows = []
    for value, df in oper.groupby(field, dropna=False):
        n_pos = int(df["outcome"].isin(["TP", "FN"]).sum())
        n_neg = int(df["outcome"].isin(["TN", "FP"]).sum())
        if n_pos < min_pos:
            continue
        fn = int((df["outcome"] == "FN").sum())
        fp = int((df["outcome"] == "FP").sum())
        lo, hi = wilson_interval(fn, n_pos)
        fpl, fph = wilson_interval(fp, n_neg) if n_neg else (float("nan"), float("nan"))
        rows.append({
            "value": str(value), "n_pos": n_pos, "n_neg": n_neg,
            "fn": fn, "fn_rate": fn / n_pos, "ci_low": lo, "ci_high": hi,
            "fp": fp,
            "fp_rate": fp / n_neg if n_neg else float("nan"),
            "fp_ci_low": fpl, "fp_ci_high": fph,
        })
    return (pd.DataFrame(rows).sort_values("n_pos", ascending=False)
            .reset_index(drop=True))


# ---------- проверка A: базовая доля по месяцам ----------

def baseline_by_month(scored: pd.DataFrame) -> pd.DataFrame:
    """Доля инцидентов и размеченных карточек по месяцам.

    Показывает, сдвигается ли база — это влияет на precision/enrichment,
    но НЕ на FN-rate (он устойчив к сдвигу базы).
    """
    oper = scored[scored["split"] == "oper"].copy()
    oper["month"] = pd.to_datetime(oper["created"]).dt.strftime("%Y-%m")
    rows = []
    for month, df in oper.groupby("month"):
        n_pos = int(df["outcome"].isin(["TP", "FN"]).sum())
        rows.append({
            "month": month, "n": len(df),
            "n_pos": n_pos, "n_neg": len(df) - n_pos,
            "positive_share": n_pos / len(df),
        })
    return pd.DataFrame(rows)


# ---------- проверка B: деградация обогащения ----------

ENRICHMENT_FIELDS = ["Вендор из сием", "Алерт", "Значение Инцидента (группировка)"]


def enrichment_decay_by_month(raw_df: pd.DataFrame,
                              created_col: str = "Создан") -> pd.DataFrame:
    """Доля заполненности ключевых полей обогащения по месяцам.

    Проверяет гипотезу из model_card.md: «с сентября обогащение деградирует».
    Возвращает: month | <field1> | <field2> | ...
    """
    df = raw_df.copy()
    df["month"] = pd.to_datetime(df[created_col], format="ISO8601",
                                 utc=True).dt.strftime("%Y-%m")
    # только период эксплуатации
    df = df[df["month"] >= "2025-09"]
    out = {"month": []}
    for f in ENRICHMENT_FIELDS:
        if f not in df.columns:
            continue
        out[f] = []
    for month, g in df.groupby("month"):
        out["month"].append(month)
        for f in ENRICHMENT_FIELDS:
            if f in df.columns:
                out[f].append(float(g[f].notna().mean()))
    return pd.DataFrame(out)


# ---------- проверка C: порог 0.5 ----------

def oper_at_threshold(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Матрица ошибок на эксплуатации при произвольном пороге (для сравнения).

    Используется только для иллюстрации «порог не спасает», рабочий порог
    остаётся 0.0193. Возвращает: threshold | tp | fp | fn | tn | fn_rate | precision
    """
    oper = scored[scored["split"] == "oper"].copy()
    pred = (oper["proba"] >= threshold).astype(int)
    y = oper["y_true"].astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return pd.DataFrame([{
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fn_rate": fn / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "enrichment": tp / (tp + fp) if (tp + fp) else float("nan"),
    }])


from sklearn.metrics import roc_auc_score, average_precision_score

def auc_by_month(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = scored[scored["split"] == "test"]
    if not test.empty and test["y_true"].nunique() >= 2:
        rows.append({
            "month": "test", "n": len(test),
            "roc_auc": roc_auc_score(test["y_true"], test["proba"]),
            "pr_auc": average_precision_score(test["y_true"], test["proba"]),
            "positive_share": float(test["y_true"].mean()),
        })
    
    oper = scored[scored["split"] == "oper"].copy()
    if oper.empty: return pd.DataFrame(rows)
    
    oper["month"] = pd.to_datetime(oper["created"]).dt.strftime("%Y-%m")
    for month, df in oper.groupby("month"):
        if df["y_true"].nunique() < 2:
            roc, pr = np.nan, np.nan
        else:
            roc = roc_auc_score(df["y_true"], df["proba"])
            pr = average_precision_score(df["y_true"], df["proba"])
        rows.append({
            "month": month, "n": len(df),
            "roc_auc": roc, "pr_auc": pr,
            "positive_share": float(df["y_true"].mean()),
        })
    return pd.DataFrame(rows)

def recompute_outcomes(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Пересчитывает TP/FP/FN/TN на лету при изменении порога."""
    df = scored.copy()
    pred = (df["proba"] >= threshold).astype(int)
    y = df["y_true"].astype(int)
    
    df["outcome"] = np.select(
        [(y == 1) & (pred == 1), 
         (y == 0) & (pred == 1), 
         (y == 1) & (pred == 0)],
        ["TP", "FP", "FN"], 
        default="TN"
    )
    return df


