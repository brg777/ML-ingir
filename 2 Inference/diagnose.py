"""Быстрая диагностика перед финализацией вопросов.

Запускается один раз: python diagnose.py
Печатает все агрегаты, которые пойдут в дашборд.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import metrics

SCORED = Path("data/scored.parquet")
RAW = Path("data/iavd_ml_dataset.parquet")


def main() -> None:
    scored = pd.read_parquet(SCORED)
    print(f"scored.parquet: {scored.shape}")
    print(f"split: {scored['split'].value_counts().to_dict()}")
    print(f"outcome: {scored['outcome'].value_counts().to_dict()}")
    print(f"max(created): {scored['created'].max()}\n")

    # ---- вопрос 1 ----
    print("=== Q1: деградация test → oper ===")
    print(metrics.degradation_snapshot(scored).to_string(index=False))

    # ---- вопрос 2 ----
    print("\n=== Q2: FN-rate по месяцам (устойчивость?) ===")
    print(metrics.fn_rate_by_month(scored).to_string(index=False))

    # ---- вопрос 3 ----
    print("\n=== Q3: обогащение и FP по месяцам ===")
    print(metrics.enrichment_by_month(scored).to_string(index=False))

    # ---- вопрос 4 ----
    print("\n=== Q4: срезы (n_pos >= 50) ===")
    for field in ["Тип инцидента", "Критичность", "Пиктограмма IRP"]:
        if field in scored.columns:
            df = metrics.error_by_slice(scored, field)
            print(f"\n--- {field}: {len(df)} срезов ---")
            print(df.to_string(index=False))

    # ---- проверка A: базовая доля ----
    print("\n=== Check A: базовая доля по месяцам ===")
    print(metrics.baseline_by_month(scored).to_string(index=False))

    # ---- проверка B: деградация обогащения ----
    if RAW.exists():
        print("\n=== Check B: заполненность обогащения по месяцам ===")
        raw = pd.read_parquet(RAW, columns=["Создан", "Вендор из сием",
                                            "Алерт", "Значение Инцидента (группировка)"])
        print(metrics.enrichment_decay_by_month(raw).to_string(index=False))

    # ---- проверка C: порог 0.5 ----
    print("\n=== Check C: порог 0.5 на эксплуатации (только иллюстрация) ===")
    print(metrics.oper_at_threshold(scored, 0.5).to_string(index=False))

     # ---- Check D: AUC по месяцам ----
    print("\n=== Check D: AUC по месяцам (ranking не ухудшился?) ===")
    print(metrics.auc_by_month(scored).to_string(index=False))


if __name__ == "__main__":
    main()
