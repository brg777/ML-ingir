"""
Дашборд SHAP: группа 3.
Запуск: python app.py -> http://127.0.0.1:5000
"""
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from taipy.gui import Gui, Markdown

with open("shap_artifacts.pkl", "rb") as f:
    A = pickle.load(f)

entity_ids = A["entity_ids"]
feature_names = A["feature_names"]
sv_group = A["sv_group"]
y_true = A["y_true"]
proba = A["proba"]
outcome = A["outcome"]
base = A["base"]
df_raw = A["df_raw_group"]
FEATURE_LABELS = A["feature_labels"]
ARTIFACT = A["artifact_features"]

THR = 0.005

def humanize(f):
    label = FEATURE_LABELS.get(f, f)
    return label.replace(" ((!) артефакт)", "").replace(" ((!) мета)", "")

def classify(f):
    if f in ARTIFACT: return "артефакт"
    if f.startswith("len_") or f.startswith("freq_"): return "мета"
    return "поведение"

def find_li(eid):
    return int(np.where(entity_ids == eid)[0][0])

def top_k(li, k=8):
    sv = sv_group[li]
    order = np.argsort(np.abs(sv))[::-1][:k]
    return [(feature_names[j], float(sv[j]), classify(feature_names[j])) for j in order]

n_cards = len(entity_ids)
tp = sum(1 for o in outcome if o == "TP")
fp = sum(1 for o in outcome if o == "FP")
fn = sum(1 for o in outcome if o == "FN")
tn = sum(1 for o in outcome if o == "TN")
recall = tp / (tp + fn) if (tp + fn) else 0
precision = tp / (tp + fp) if (tp + fp) else 0
mean_abs = float(np.abs(sv_group).mean())

kpi_recall    = f"{recall*100:.0f} %"
kpi_precision = f"{precision*100:.0f} %"
kpi_mean      = f"{mean_abs:.2f}"

summary = pd.DataFrame({
    "EntityId": entity_ids.astype(str),
    "Факт":     ["инцидент" if y == 1 else "FP" for y in y_true],
    "Proba":    [f"{p:.3g}" for p in proba],
    "Модель":   ["инцидент" if p >= THR else "FP" for p in proba],
    "Итог":     outcome,
})

_ma = pd.DataFrame({
    "feature": feature_names,
    "m": np.abs(sv_group).mean(0),
}).sort_values("m", ascending=False).head(15)
_ma["label"] = _ma.feature.map(humanize)
_ma["kind"]  = _ma.feature.map(classify)

fig_global = px.bar(
    _ma, x="m", y="label", orientation="h", color="kind",
    color_discrete_map={"поведение": "#2ca02c", "артефакт": "#d62728", "мета": "#ff7f0e"},
    labels={"m": "Средний |SHAP|", "label": "", "kind": ""},
)
fig_global.update_layout(
    yaxis={"autorange": "reversed"}, height=460,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", y=-0.18, x=0),
)

def build_waterfall(eid, k=8):
    li = find_li(eid)
    sv = sv_group[li]
    order = np.argsort(np.abs(sv))[::-1][:k]
    labels = [humanize(feature_names[j]) for j in order]
    vals = sv[order]
    total = float(sv.sum())
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(vals) + ["total"],
        y=labels + ["ИТОГО"],
        x=vals.tolist() + [total],
        text=[f"{v:+.2f}" for v in vals] + [f"{total:+.2f}"],
        connector={"line": {"color": "gray"}},
        decreasing={"marker": {"color": "#d62728"}},
        increasing={"marker": {"color": "#2ca02c"}},
        totals={"marker": {"color": "#1f77b4"}},
    ))
    fig.update_layout(
        xaxis_title="SHAP  (+ к инциденту / − к FP)",
        height=440, margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

def card_narrative(eid):
    li = find_li(eid)
    tops = top_k(li, 5)
    pos = [t for t in tops if t[1] > 0][:3]
    neg = [t for t in tops if t[1] < 0][:3]
    lines = [f"Объяснение для карточки {eid} (вердикт: {outcome[li]}, proba: {proba[li]:.3g}):", ""]
    if pos:
        lines.append("Факторы в пользу «Инцидента» (+):")
        for f, v, k in pos:
            lines.append(f"  • {humanize(f)} (вклад: {v:+.2f})")
    if neg:
        lines.append("\nФакторы в пользу «Ложного срабатывания» (−):")
        for f, v, k in neg:
            lines.append(f"  • {humanize(f)} (вклад: {v:+.2f})")
    return "\n".join(lines)

RAW_FIELDS = [
    "Наименование", "Тип инцидента", "Критичность", "Источник инцидента",
    "Теги", "Алерт", "Команда", "CMDLINE", "commandName",
    "processName", "Привилегии аккаунта", "suser", "duser",
    "Hostname Цели", "Hostname атакующего",
]

def raw_ctx(eid):
    r = df_raw[df_raw.EntityId == eid]
    if len(r) == 0:
        return "— нет данных —"
    row = r.iloc[0]
    lines = []
    for c in RAW_FIELDS:
        if c in row.index and pd.notna(row[c]):
            v = str(row[c])
            if len(v) > 300: v = v[:300] + "…"
            lines.append(f"{c}: {v}")
    return "\n".join(lines) or "—"

card_options  = [f"{e} · {o}" for e, o in zip(entity_ids, outcome)]
selected_card = card_options[find_li(527608)]

fig_waterfall   = build_waterfall(527608)
text_narrative  = card_narrative(527608)
raw_text        = raw_ctx(527608)

def on_card_change(state, var_name, var_value):
    eid = int(var_value.split(" ")[0])
    state.fig_waterfall  = build_waterfall(eid)
    state.text_narrative = card_narrative(eid)
    state.raw_text       = raw_ctx(eid)

page = Markdown("""
# SOC · SHAP · группа 3

<|layout|columns=1 1 1 1|
<|
**Карточек**

<|{n_cards}|text|>
|>
<|
**Recall**

<|{kpi_recall}|text|>
|>
<|
**Precision**

<|{kpi_precision}|text|>
|>
<|
**Средний abs(SHAP)**

<|{kpi_mean}|text|>
|>
|>

---

## 1. Обзор группы

<|{summary}|table|>

<br/>

<|chart|figure={fig_global}|height=480px|>

---

## 2. Разбор конкретной карточки

Выберите карточку из списка: **<|{selected_card}|selector|lov={card_options}|on_change=on_card_change|dropdown=True|>**

<br/>

<|chart|figure={fig_waterfall}|height=450px|>

### Детализация SHAP для выбранной карточки

<|{text_narrative}|text|mode=pre|>

### Сырые значения полей из базы

<|{raw_text}|text|mode=pre|>
""")

if __name__ == "__main__":
    Gui(page).run(title="SOC · SHAP · группа 3", port=5000, use_reloader=False)