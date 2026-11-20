"""Taipy-дашборд: доверие к модели-советнику SOC lgbm_gate1__v001."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import taipy.gui.builder as tgb
from taipy.gui import Gui
from src import charts, metrics

HERE = Path(__file__).resolve().parent
SCORED = HERE / "data" / "scored.parquet"

scored_raw = pd.read_parquet(SCORED)
scored_raw["Организация"] = scored_raw["Организация"].apply(
    lambda v: json.loads(v).get("name", v) if isinstance(v, str) and v.startswith("{") else v
)

# ---------- Состояние ----------
THRESH_OPTIONS = ["Рабочий τ (0.0193)", "Высокий (0.5)", "Строгий (0.8)"]
threshold_opt = THRESH_OPTIONS[0]
filter_month = "Все"
filter_org = "Все"
selected_slice = "Тип инцидента"

oper_months = pd.to_datetime(scored_raw[scored_raw["split"] == "oper"]["created"]).dt.strftime("%Y-%m").unique()
months = ["Все"] + sorted(oper_months.tolist())
orgs = ["Все"] + sorted(scored_raw["Организация"].dropna().unique().tolist())
SLICE_FIELDS = ["Тип инцидента", "Критичность", "Пиктограмма IRP"]

# Переменные KPI (инициализация)
k0_label, k0_value, k0_sub, k0_delta, k0_class = "", "", "", "", ""
k1_label, k1_value, k1_sub, k1_delta, k1_class = "", "", "", "", ""
k2_label, k2_value, k2_sub, k2_delta, k2_class = "", "", "", "", ""
k3_label, k3_value, k3_sub, k3_delta, k3_class = "", "", "", "", ""

fig_q1, fig_q2, fig_q3, fig_q4 = None, None, None, None

def get_threshold(opt: str) -> float:
    if opt == "Рабочий τ (0.0193)": return metrics.THRESHOLD
    if opt == "Высокий (0.5)": return 0.5
    return 0.8

def update_dashboard(state, var_name=None, value=None):
    thresh = get_threshold(state.threshold_opt)
    raw = state.scored_raw

    # 1. Приёмка (test) — ВСЕГДА полностью
    test_df = metrics.recompute_outcomes(raw[raw["split"] == "test"].copy(), thresh)
    
    # 2. Эксплуатация (oper)
    oper_df = raw[raw["split"] == "oper"].copy()
    
    # 3. Для графиков Q2 и Q3 берем ВСЮ эксплуатацию (чтобы видеть тренд)
    oper_full_df = metrics.recompute_outcomes(oper_df.copy(), thresh)
    df_full = pd.concat([test_df, oper_full_df])

    # 4. Для KPI и Q1/Q4 применяем фильтры
    oper_filtered_df = oper_df.copy()
    if state.filter_month != "Все":
        oper_filtered_df = oper_filtered_df[pd.to_datetime(oper_filtered_df["created"]).dt.strftime("%Y-%m") == state.filter_month]
    if state.filter_org != "Все":
        oper_filtered_df = oper_filtered_df[oper_filtered_df["Организация"] == state.filter_org]
    oper_filtered_df = metrics.recompute_outcomes(oper_filtered_df, thresh)
    df_filtered = pd.concat([test_df, oper_filtered_df])

    # 5. Обновляем KPI через setattr
    tiles = charts.kpi_tiles(df_filtered)
    for i, t in enumerate(tiles):
        setattr(state, f"k{i}_label", t["label"])
        setattr(state, f"k{i}_value", t["value"])
        setattr(state, f"k{i}_sub", t["sub"])
        setattr(state, f"k{i}_delta", t["delta"])
        setattr(state, f"k{i}_class", "kpi-bad" if t["signal"] == "bad" else "kpi-good")

    # 6. Обновляем графики
    state.fig_q1 = charts.chart_q1_degradation(df_filtered)
    state.fig_q2 = charts.chart_q2_ranking_vs_operating(df_full, thresh, highlight_month=state.filter_month if state.filter_month != "Все" else None)
    state.fig_q3 = charts.chart_q3_load(df_full, highlight_month=state.filter_month if state.filter_month != "Все" else None)
    state.fig_q4 = charts.chart_q4_slices(df_filtered, state.selected_slice, thresh)

# Инициализация
class DummyState:
    scored_raw = scored_raw
    threshold_opt = threshold_opt
    filter_month = filter_month
    filter_org = filter_org
    selected_slice = selected_slice

init_state = DummyState()
update_dashboard(init_state)
# Переносим значения в глобальные переменные Taipy
for i in range(4):
    for attr in ["label", "value", "sub", "delta", "class"]:
        globals()[f"k{i}_{attr}"] = getattr(init_state, f"k{i}_{attr}", "")
fig_q1, fig_q2, fig_q3, fig_q4 = init_state.fig_q1, init_state.fig_q2, init_state.fig_q3, init_state.fig_q4

# ---------- CSS ----------
CUSTOM_CSS = """
<style>
    .page-wrap { padding: 24px 40px; max-width: 1600px; margin: 0 auto; box-sizing: border-box; }
    .h1 { font-size: 26px; font-weight: 700; color: #1f2937; margin: 8px 0 4px 0; }
    .h2 { font-size: 18px; font-weight: 600; color: #1f2937; margin: 28px 0 8px 0; }
    .subtitle { font-size: 13px; color: #6b7280; margin-bottom: 24px; }
    .kpi-row { gap: 16px; margin: 20px 0 28px 0; }
    .kpi-card { background: #ffffff; border-radius: 10px; padding: 18px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #e5e7eb; min-height: 120px; }
    .kpi-label { font-size: 11px; color: #6b7280; text-transform: uppercase; font-weight: 600; }
    .kpi-value { font-size: 34px; font-weight: 700; margin: 6px 0 2px 0; line-height: 1.1; }
    .kpi-sub { font-size: 12px; color: #9ca3af; }
    .kpi-delta { font-size: 12px; font-weight: 600; margin-top: 6px; }
    .kpi-bad .kpi-value, .kpi-bad .kpi-delta { color: #a3162b; }
    .kpi-good .kpi-value, .kpi-good .kpi-delta { color: #f59e0b; }
    .why { font-size: 13px; color: #374151; background: #f9fafb; border-left: 4px solid #3b82f6;
           padding: 16px 20px; border-radius: 4px; margin: 16px 0 24px 0; line-height: 1.6; }
    .why ul { margin: 4px 0 8px 16px; padding: 0; } .why li { margin-bottom: 4px; }
    .verdict { font-size: 14px; color: #1f2937; line-height: 1.75; max-width: 900px; }
    .filters-row { align-items: center; padding: 16px 24px; background: white; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 24px; }
</style>
"""

# ---------- UI ----------
def filters_ui():
    with tgb.layout(columns="1 1 1", gap="2rem", class_name="filters-row"):
        tgb.selector(value="{threshold_opt}", lov="{THRESH_OPTIONS}", label="Сценарий порога", on_change=update_dashboard)
        tgb.selector(value="{filter_month}", lov="{months}", label="Месяц", on_change=update_dashboard)
        tgb.selector(value="{filter_org}", lov="{orgs}", label="Организация", on_change=update_dashboard)

def kpi_card(i: int):
    with tgb.part(class_name="{k" + str(i) + "_class} kpi-card"):
        tgb.text("{k" + str(i) + "_label}", class_name="kpi-label")
        tgb.text("{k" + str(i) + "_value}", class_name="kpi-value")
        tgb.text("{k" + str(i) + "_sub}", class_name="kpi-sub")
        tgb.text("{k" + str(i) + "_delta}", class_name="kpi-delta")

with tgb.Page() as overview_page:
    tgb.html("div", CUSTOM_CSS)
    tgb.navbar()
    with tgb.part(class_name="page-wrap"):
        tgb.text("Доверие к модели-советнику SOC", class_name="h1")
        tgb.text("Модель lgbm_gate1__v001 · порог τ = {threshold_opt} · эксплуатация 01.09–20.11.2025", class_name="subtitle")
        filters_ui()
        with tgb.layout(columns="1 1 1 1", class_name="kpi-row"):
            for i in range(4): kpi_card(i)
        
        tgb.text("Вердикт", class_name="h2")
        tgb.text(
            "**Доверять как советчику для приоритизации — да.** "
            "ROC-AUC стабилен 0,91–0,92 на всём периоде эксплуатации: модель по-прежнему ранжирует карточки правильно.\n\n"
            "**Доверять как единственному фильтру или для планирования нагрузки — нет.** "
            "Операционная работа непредсказуема: доля пропусков колеблется по месяцам, "
            "а FP в день меняется в разы. Попробуйте переключить порог в фильтре — "
            "вы увидите, что идеальной точки нет: сигнал обогащения в данных деградировал, и модель «плывет».",
            mode="markdown", class_name="verdict",
        )

with tgb.Page() as q1_page:
    tgb.html("div", CUSTOM_CSS)
    tgb.navbar()
    with tgb.part(class_name="page-wrap"):
        tgb.text("Q1. Насколько хуже в бою, чем на приёмке?", class_name="h1")
        filters_ui()
        with tgb.part(class_name="why"):
            tgb.text(charts.WHY["q1"], mode="markdown")
        tgb.chart(figure="{fig_q1}")

with tgb.Page() as q2_page:
    tgb.html("div", CUSTOM_CSS)
    tgb.navbar()
    with tgb.part(class_name="page-wrap"):
        tgb.text("Q2. Ranking стабилен — а operating point скачет", class_name="h1")
        filters_ui()
        with tgb.part(class_name="why"):
            tgb.text(charts.WHY["q2"], mode="markdown")
        tgb.chart(figure="{fig_q2}")

with tgb.Page() as q3_page:
    tgb.html("div", CUSTOM_CSS)
    tgb.navbar()
    with tgb.part(class_name="page-wrap"):
        tgb.text("Q3. Сколько работы модель создаёт аналитикам?", class_name="h1")
        filters_ui()
        with tgb.part(class_name="why"):
            tgb.text(charts.WHY["q3"], mode="markdown")
        tgb.chart(figure="{fig_q3}")

with tgb.Page() as q4_page:
    tgb.html("div", CUSTOM_CSS)
    tgb.navbar()
    with tgb.part(class_name="page-wrap"):
        tgb.text("Q4. Где модель слепнет систематически?", class_name="h1")
        filters_ui()
        with tgb.part(class_name="why"):
            tgb.text(charts.WHY["q4"], mode="markdown")
        tgb.selector(value="{selected_slice}", lov="{SLICE_FIELDS}", label="Поле среза", on_change=update_dashboard, width="300px")
        tgb.chart(figure="{fig_q4}")

pages = {"overview": overview_page, "q1_degradation": q1_page, "q2_ranking": q2_page, "q3_load": q3_page, "q4_slices": q4_page}

if __name__ == "__main__":
    Gui(pages=pages).run(title="SOC Trust Analysis", host="0.0.0.0", port=5003, dark_mode=False, use_reloader=False)