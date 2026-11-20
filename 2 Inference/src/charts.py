"""Plotly-фигуры для дашборда пункта 2.

Каждая функция — данные → go.Figure. Никакого HTML, вёрстки и Taipy здесь нет.
Обоснование выбора каждого графика — в словаре WHY внизу файла.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import metrics
from . import palette as pal


# ---------- утилиты ----------

def _clean_slice_value(v) -> str:
    """'["Нарушение политики"]' -> 'Нарушение политики'."""
    return str(v).strip("[]\"'").strip()


def _base_layout(height: int = 400) -> dict:
    return dict(
        height=height,
        margin=dict(l=50, r=20, t=80, b=80), # Больше места сверху и снизу
        plot_bgcolor=pal.BACKGROUND,
        paper_bgcolor=pal.BACKGROUND,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=pal.TEXT),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Легенда уходит вниз под график
            xanchor="center",
            x=0.5,
            font_size=11
        ),
    )


def _style_axes(fig):
    fig.update_xaxes(gridcolor=pal.GRID, zeroline=False,
                     linecolor=pal.GRID, tickfont_size=11)
    fig.update_yaxes(gridcolor=pal.GRID, zeroline=False,
                     linecolor=pal.GRID, tickfont_size=11)


# ---------- Q1: деградация test → oper ----------

def chart_q1_degradation(scored: pd.DataFrame) -> go.Figure:
    """Парные точки test → oper по 4 метрикам (small multiples 2×2).

    Каждая панель — своя шкала, чтобы ×2,7 по FN-rate не потерялось
    на фоне 94 % recall.
    """
    df = metrics.degradation_snapshot(scored)
    panels = [
        ("fn_rate",   "Доля пропусков",       pal.MISSED),
        ("recall",    "Полнота",              pal.TRUE_POSITIVE),
        ("precision", "Точность",             pal.TRUE_POSITIVE),
        ("fp_rate",   "Доля ложных тревог",   pal.FALSE_POSITIVE),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[p[1] for p in panels],
                        vertical_spacing=0.22, horizontal_spacing=0.12)

    for i, (metric, label, color) in enumerate(panels):
        row, col = i // 2 + 1, i % 2 + 1
        sub = df[df["metric"] == metric]
        t = sub[sub["split"] == "test"].iloc[0]
        o = sub[sub["split"] == "oper"].iloc[0]

        fig.add_trace(go.Scatter(
            x=["Приёмка", "Эксплуатация"],
            y=[t["value"] * 100, o["value"] * 100],
            mode="lines+markers",
            line=dict(color="#9ca3af", width=2, dash="dot"),
            marker=dict(size=[12, 14], color=[pal.ACCEPTANCE, color],
                        line=dict(width=2, color="white")),
            error_y=dict(
                type="data", symmetric=False,
                array=[(t["ci_high"] - t["value"]) * 100,
                       (o["ci_high"] - o["value"]) * 100],
                arrayminus=[(t["value"] - t["ci_low"]) * 100,
                            (o["value"] - o["ci_low"]) * 100],
                color=pal.TEXT_MUTED, thickness=1.5, width=6,
            ),
            showlegend=False,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ), row=row, col=col)

        # подпись «было → стало» в subplot title
        delta = o["value"] - t["value"]
        arrow = "↑" if delta > 0 else "↓"
        fig.layout.annotations[i].text = (
            f"<b>{label}</b><br>"
            f"<span style='font-size:11px;color:{pal.TEXT_MUTED}'>"
            f"{t['value']*100:.1f}% → {o['value']*100:.1f}% "
            f"{arrow} {abs(delta)*100:.1f} п.п.</span>"
        )

    fig.update_layout(**_base_layout(height=440))
    fig.update_yaxes(ticksuffix="%")
    _style_axes(fig)
    return fig


# ---------- Q2: ranking vs operating point ----------

def chart_q2_ranking_vs_operating(scored: pd.DataFrame, threshold: float, highlight_month: str = None) -> go.Figure:
    auc = metrics.auc_by_month(scored)
    fnr = metrics.fn_rate_by_month(scored)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Ranking: ROC-AUC по месяцам</b><br>"
            "<span style='font-size:11px;color:#6b7280'>стабилен — модель не «сломалась»</span>",
            f"<b>Operating point: доля пропусков (FN-rate)</b><br>"
            f"<span style='font-size:11px;color:#6b7280'>при τ = {threshold:.4f}</span>",
        ),
        horizontal_spacing=0.12,
    )

    test_auc = float(auc[auc["month"] == "test"].iloc[0]["roc_auc"]) if "test" in auc["month"].values else 0.977
    oper_auc = auc[auc["month"] != "test"]
    
    fig.add_trace(go.Scatter(
        x=oper_auc["month"], y=oper_auc["roc_auc"],
        mode="lines+markers",
        marker=dict(size=14, color=pal.TRUE_POSITIVE, line=dict(width=2, color="white")),
        line=dict(color=pal.TRUE_POSITIVE, width=2.5),
        name="ROC-AUC (эксплуатация)",
        hovertemplate="%{x}: ROC-AUC %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=test_auc, line_dash="dash", line_color=pal.ACCEPTANCE,
                  annotation_text=f"приёмка: {test_auc:.3f}", annotation_position="bottom right",
                  annotation_font_size=11, row=1, col=1)

    test_lo = float(fnr.iloc[0]["test_ci_low"]) * 100
    test_hi = float(fnr.iloc[0]["test_ci_high"]) * 100
    fig.add_hrect(y0=test_lo, y1=test_hi, fillcolor=pal.ACCEPTANCE, opacity=0.18, line_width=0, row=1, col=2)

    for _, r in fnr.iterrows():
        is_highlight = (highlight_month and r["month"] == highlight_month)
        is_partial = bool(r["is_partial"])
        fig.add_trace(go.Scatter(
            x=[r["month"]], y=[r["fn_rate"] * 100],
            mode="markers",
            marker=dict(
                size=18 if is_highlight else 14,
                color=pal.MISSED if not is_partial else "white",
                line=dict(width=3 if is_highlight else 2, color=pal.MISSED),
            ),
            error_y=dict(type="data", symmetric=False, array=[(r["ci_high"] - r["fn_rate"]) * 100],
                         arrayminus=[(r["fn_rate"] - r["ci_low"]) * 100], color=pal.MISSED, thickness=2, width=6),
            showlegend=False,
            hovertemplate=(f"{r['month']}: FN {r['fn']}/{r['n_pos']} = {r['fn_rate']*100:.1f}%<extra></extra>"),
        ), row=1, col=2)

    fig.add_trace(go.Scatter(x=fnr["month"], y=fnr["fn_rate"] * 100, mode="lines",
                             line=dict(color=pal.MISSED, width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=2)

    fig.update_layout(**_base_layout(height=450))
    fig.update_xaxes(type='category')
    fig.update_yaxes(title_text="ROC-AUC", range=[0.85, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="Доля пропусков, %", ticksuffix="%", row=1, col=2)
    _style_axes(fig)
    return fig

def chart_q3_load(scored: pd.DataFrame, highlight_month: str = None) -> go.Figure:
    enr = metrics.enrichment_by_month(scored)
    if enr.empty: return go.Figure()
    
    spread = enr["fp_per_day"].max() / max(enr["fp_per_day"].min(), 1e-6)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Доля реальных инцидентов в приоритете</b><br><span style='font-size:11px;color:#6b7280'>сколько из 100 приоритетных — реальные</span>",
            f"<b>Ложных срабатываний в день</b><br><span style='font-size:11px;color:{pal.MISSED}'>разброс ×{spread:.1f} между месяцами</span>",
        ),
        horizontal_spacing=0.12,
    )

    test_enr = float(enr.iloc[0]["test_enrichment"])
    
    # Подсветка выбранного месяца
    colors_enr = [pal.TRUE_POSITIVE if (highlight_month and m == highlight_month) else ("#fcd34d" if not p else pal.PARTIAL) for m, p in zip(enr["month"], enr["is_partial"])]
    colors_fp = [pal.FALSE_POSITIVE if (highlight_month and m == highlight_month) else ("#93c5fd" if not p else pal.PARTIAL) for m, p in zip(enr["month"], enr["is_partial"])]

    fig.add_trace(go.Bar(x=enr["month"], y=enr["enrichment"] * 100, marker=dict(color=colors_enr),
                         text=[f"{v*100:.0f}%" for v in enr["enrichment"]], textposition="outside", name="Доля реальных в приоритете"), row=1, col=1)
    fig.add_hline(y=test_enr * 100, line_dash="dash", line_color=pal.ACCEPTANCE,
                  annotation_text=f"приёмка: {test_enr*100:.0f}%", annotation_position="bottom right", row=1, col=1)

    fig.add_trace(go.Bar(x=enr["month"], y=enr["fp_per_day"], marker=dict(color=colors_fp),
                         text=[f"{v:.0f}" for v in enr["fp_per_day"]], textposition="outside", name="FP в день"), row=1, col=2)

    fig.update_layout(**_base_layout(height=450))
    fig.update_xaxes(type='category')
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="FP / день", row=1, col=2)
    _style_axes(fig)
    return fig

# ---------- Q4: срезы с систематическими ошибками ----------

def chart_q4_slices(scored: pd.DataFrame, field: str = "Тип инцидента", threshold: float = 0.0193) -> go.Figure:
    """Точечная диаграмма срезов: FN-rate с 95% CI по одному полю.

    Интервалы отделяют системную слепоту от шума:
    если CI среза не пересекается с остальными — ошибка системная.
    """
    df = metrics.error_by_slice(scored, field)
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["label"] = df["value"].map(_clean_slice_value)
    df = df.sort_values("fn_rate", ascending=True).reset_index(drop=True)

    test = scored[scored["split"] == "test"]
    test_fn = int((test["outcome"] == "FN").sum())
    test_pos = int(test["outcome"].isin(["TP", "FN"]).sum())
    test_rate = test_fn / test_pos if test_pos else 0.0
    lo, hi = metrics.wilson_interval(test_fn, test_pos)

    fig = go.Figure()

    # полоса-ориентир приёмки
    fig.add_vrect(x0=lo * 100, x1=hi * 100,
                  fillcolor=pal.ACCEPTANCE, opacity=0.18, line_width=0)
    
    fig.add_vline(x=test_rate * 100, line_dash="dash", line_color=pal.ACCEPTANCE,
                  annotation_text=f"приёмка: {test_rate*100:.1f}%",
                  annotation_position="top",      # <-- верните это
                  annotation_font_size=11)        # <-- и это

    # точки срезов
    colors = [pal.MISSED if r > 0.5 else (pal.FALSE_POSITIVE if r > 0.15 else pal.TRUE_POSITIVE)
              for r in df["fn_rate"]]
    fig.add_trace(go.Scatter(
        x=df["fn_rate"] * 100, y=df["label"],
        mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=2, color="white")),
        error_x=dict(
            type="data", symmetric=False,
            array=(df["ci_high"] - df["fn_rate"]) * 100,
            arrayminus=(df["fn_rate"] - df["ci_low"]) * 100,
            color=pal.TEXT_MUTED, thickness=1.5, width=6,
        ),
        text=[f"{r['label']}<br>FN {r['fn']}/{r['n_pos']} = {r['fn_rate']*100:.1f}%"
              for _, r in df.iterrows()],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для выбранных фильтров", 
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        return fig

    fig.update_layout(**_base_layout(height=440))
    fig.update_xaxes(title_text="Доля пропусков (FN-rate), %", ticksuffix="%")
    fig.update_layout(yaxis=dict(autorange=True))
    _style_axes(fig)
    return fig


# ---------- KPI-плитки ----------

def kpi_tiles(scored: pd.DataFrame) -> list[dict]:
    """4 KPI-плитки для верхней полосы дашборда (безопасные к пустым данным)."""
    empty_tile = {
        "label": "Нет данных", "value": "—", "sub": "выборка пуста", 
        "delta": "", "color": pal.TEXT_MUTED, "signal": "neutral"
    }
    
    if scored.empty:
        return [empty_tile] * 4

    snap = metrics.degradation_snapshot(scored)
    enr = metrics.enrichment_by_month(scored)
    slices = metrics.error_by_slice(scored, "Тип инцидента")

    # Проверяем, что есть и приёмка (test), и эксплуатация (oper)
    test_fn = snap[(snap["metric"] == "fn_rate") & (snap["split"] == "test")]
    oper_fn = snap[(snap["metric"] == "fn_rate") & (snap["split"] == "oper")]

    if test_fn.empty or oper_fn.empty or enr.empty:
        return [empty_tile] * 4

    fn_t = test_fn.iloc[0]
    fn_o = oper_fn.iloc[0]
    
    # Защита от деления на ноль, если на приёмке 0 пропусков (например, при пороге 0)
    ratio = fn_o["value"] / fn_t["value"] if fn_t["value"] > 0 else float('inf')
    ratio_str = f"×{ratio:.1f}" if ratio != float('inf') else "∞"

    enr_t = float(enr.iloc[0]["test_enrichment"])
    enr_o = float(enr["enrichment"].mean())
    
    fp_spread = float(enr["fp_per_day"].max() / max(enr["fp_per_day"].min(), 1e-6))
    worst = slices.sort_values("fn_rate", ascending=False).iloc[0] if not slices.empty else None

    return [
        {
            "label": "Доля пропусков (FN-rate)",
            "value": f"{fn_o['value']*100:.1f}%",
            "sub": f"было {fn_t['value']*100:.1f}% на приёмке",
            "delta": ratio_str,
            "color": pal.MISSED,
            "signal": "bad",
        },
        {
            "label": "Доля реальных в приоритете",
            "value": f"{enr_o*100:.1f}%",
            "sub": f"было {enr_t*100:.1f}% на приёмке",
            "delta": f"−{(enr_t-enr_o)*100:.0f} п.п.",
            "color": pal.FALSE_POSITIVE,
            "signal": "bad",
        },
        {
            "label": "Разброс нагрузки (FP/день)",
            "value": f"×{fp_spread:.1f}",
            "sub": f"от {enr['fp_per_day'].min():.0f} до {enr['fp_per_day'].max():.0f} в день",
            "delta": "непредсказуемо",
            "color": pal.FALSE_POSITIVE,
            "signal": "bad",
        },
        {
            "label": "Худший класс по пропускам",
            "value": (f"{worst['fn_rate']*100:.0f}%" if worst is not None else "—"),
            "sub": (_clean_slice_value(worst["value"]) if worst is not None else "—"),
            "delta": "системная слепота",
            "color": pal.MISSED,
            "signal": "bad",
        },
    ]


# ---------- обоснования «почему этот график» ----------

WHY = {
    "q1": (
        "**Почему парные точки?**\n\n"
        "Руководителю нужно увидеть направление и величину сдвига одной карточкой, а не читать таблицу. "
        "Каждая панель имеет свою шкалу: без этого ×2,7 по FN-rate потерялось бы на фоне 94 % recall.\n\n"
        "Интервалы Уилсона показывают, что деградация статистически значима. Доля пропусков выделена цветом как главная метрика SOC."
    ),
    "q2": (
        "**Почему две панели рядом?**\n\n"
        "Это два разных диагноза:\n"
        "* **ROC-AUC стабилен** (0,91–0,92) — модель не сломалась в ранжировании.\n"
        "* **FN-rate скачет** (19 % → 10 %) — operating point при фиксированном пороге τ съезжает.\n\n"
        "Один график без другого вводил бы в заблуждение: падение FN в ноябре выглядело бы как улучшение, хотя на самом деле это over-flagging."
    ),
    "q3": (
        "**Почему два столбца рядом?**\n\n"
        "Они отвечают на разные вопросы:\n"
        "* **Enrichment** — каково качество очереди (доля реальных инцидентов в приоритете).\n"
        "* **FP в день** — какой абсолютный объём мусора разбирает смена.\n\n"
        "Один без другого неполон: enrichment упал до 24 %, но чтобы понять, что это значит для смены, нужен FP/день — он вырос с 12 до 85. Разброс ×6,8 вынесен в подпись — это аргумент против планирования нагрузки «по среднему»."
    ),
    "q4": (
        "**Почему точечная диаграмма с интервалами?**\n\n"
        "Интервалы — единственный способ на глаз отличить системную слепоту от шума. Если CI среза не пересекается с остальными — различие системное.\n\n"
        "Точки вместо баров удобнее для длинных названий классов и переключателя поля. Порог n ≥ 50 отсекает шумные срезы, где интервал шире самого эффекта."
    ),
}


SLICE_FIELDS = ["Тип инцидента", "Критичность", "Пиктограмма IRP"]
