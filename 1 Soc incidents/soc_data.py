"""Подготовка данных и агрегаты для дашборда SOC."""
import numpy as np
import pandas as pd

COLS = ["Создан", "Вердикт", "Дата закрытия (факт)", "Критичность", "Ответственный",
        "Организация", "Наименование", "Период взятия в работу",
        "SLA просрочен (взятие в работу)"]
VERDICT = {"False Positive": "False Positive", "Не указан": "Не указан",
           "Решён": "Решён", "Дубликат": "Дубликат"}
VERDICT_ORDER = ["False Positive", "Не указан", "Решён", "Дубликат", "Другое"]
CRIT_ORDER = ["Критическая", "Высокая", "Средняя", "Низкая"]


def _ts(s):
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, unit="us")


def load(path: str) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=COLS)
    t, tc = _ts(raw["Создан"]), _ts(raw["Дата закрытия (факт)"])
    d = pd.DataFrame({
        "month": t.dt.strftime("%Y-%m"), "hour": t.dt.hour, "wd": t.dt.weekday,
        "verdict": raw["Вердикт"].map(VERDICT).fillna("Другое"),
        "crit": raw["Критичность"],
        "org": raw["Организация"].fillna("Не задано"),
        "rule": raw["Наименование"],
        "dur": (tc - t).dt.total_seconds() / 60,
        "react": raw["Период взятия в работу"] / 60000,
        "breach": raw["SLA просрочен (взятие в работу)"].astype(str).map({"True": 1, "False": 0}),
    })
    d.loc[d.dur < 0, "dur"] = np.nan
    d.loc[d.react < 0, "react"] = np.nan
    rank = raw["Ответственный"].value_counts().index
    anon = {n: f"A{i + 1:02d}" for i, n in enumerate(rank)}
    d["analyst"] = raw["Ответственный"].map(anon)
    d = d[d.month >= "2024-05"]
    return d.reset_index(drop=True)


def kpis(d):
    n = len(d)
    vc = d.verdict.value_counts()
    rules = d[~d.rule.fillna("").str.contains("~~", regex=False)].groupby("rule").size().sort_values(ascending=False)
    r80 = int((rules.cumsum() / rules.sum() < 0.8).sum() + 1) if len(rules) else 0
    return {"n": n, "fp": vc.get("False Positive", 0) / n, "na": vc.get("Не указан", 0) / n,
            "res": vc.get("Решён", 0) / n, "breach": d.breach.mean(), "r80": r80, "rules": len(rules)}


def monthly(d):
    return d.groupby(["month", "verdict"]).size().rename("n").reset_index()


def rules(d, min_n=300):
    r = d[~d.rule.fillna("").str.contains("~~", regex=False)].groupby("rule").agg(
        n=("verdict", "size"), fp_share=("verdict", lambda s: (s == "False Positive").mean() * 100))
    r["fp_cards"] = (r.n * r.fp_share / 100).round()
    return r[r.n >= min_n].reset_index()


def sla_by_crit(d):
    g = d[d.crit.isin(CRIT_ORDER)].groupby("crit").agg(
        cards=("breach", "size"), breach=("breach", lambda s: s.mean() * 100), med_wait=("react", "median"))
    return g.reindex(CRIT_ORDER).reset_index().round(1)


def sla_by_hour(d):
    return d.groupby("hour").breach.mean().mul(100).round(1).rename("breach").reset_index()


def dur_by_verdict(d):
    g = d[d.verdict != "Другое"].groupby("verdict").dur
    out = pd.DataFrame({"Карточек": d[d.verdict != "Другое"].groupby("verdict").size(),
                        "Медиана, мин": g.median(), "P90, мин": g.quantile(0.9)}).round(1)
    return out.reset_index().rename(columns={"verdict": "Вердикт"})


def heat(d):
    h = d.groupby(["wd", "hour"]).size().unstack(fill_value=0)
    return h.reindex(index=range(7), columns=range(24), fill_value=0)


def analysts(d, min_cards=1000):
    a = d.dropna(subset=["analyst"]).groupby("analyst").agg(
        n=("verdict", "size"),
        fp=("verdict", lambda s: (s == "False Positive").mean() * 100),
        na=("verdict", lambda s: (s == "Не указан").mean() * 100))
    a = a[a.n >= min_cards].sort_values("n").reset_index()
    a["other"] = (100 - a.fp - a.na).clip(lower=0)
    a["label"] = a.analyst + " · " + a.n.astype(str)
    return a
