"""Taipy-дашборд по карточкам SOC."""
import os

import plotly.express as px
import plotly.graph_objects as go
import taipy.gui.builder as tgb
from taipy.gui import Gui, notify

import soc_data as S

PATH = os.environ.get("SOC_PARQUET", "iavd_incidents_raw_target.parquet")
DF = S.load(PATH)
COLORS = {
    "False Positive": "#9aa3b2",
    "Не указан": "#e8a33d",
    "Решён": "#2f7d6d",
    "Дубликат": "#7a8ae0",
    "Другое": "#d64545",
}
ORGS = ["Все"] + sorted(DF.org.unique())

org, min_n = "Все", 300
k_fp = k_na = k_res = k_sla = k_rules = k_n = ""
fig_month = fig_rules = fig_top = fig_crit = fig_hour = fig_heat = fig_an = None
tbl_dur = None


def _layout(fig, h=380):
    fig.update_layout(
        height=h,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def update(state):
    current_min_n = state.min_n if state.min_n is not None else 300
    d = DF if state.org == "Все" else DF[DF.org == state.org]

    if d.empty:
        state.k_n = "0"
        state.k_fp = state.k_na = state.k_res = state.k_sla = "—"
        state.k_rules = "0 из 0"
        state.fig_month = _layout(go.Figure().update_layout(title="Нет данных"))
        state.fig_rules = _layout(go.Figure().update_layout(title="Нет данных"))
        state.fig_top = _layout(go.Figure().update_layout(title="Нет данных"))
        state.fig_crit = _layout(go.Figure().update_layout(title="Нет данных"))
        state.fig_hour = _layout(go.Figure().update_layout(title="Нет данных"))
        state.fig_heat = _layout(go.Figure().update_layout(title="Нет данных"))
        state.fig_an = _layout(go.Figure().update_layout(title="Нет данных"))
        state.tbl_dur = None
        return

    k = S.kpis(d)
    
    pct = lambda x: f"{x * 100:.1f}%".replace(".", ",") if x == x else "—"
    state.k_n = f"{k['n']:,}".replace(",", " ")
    state.k_fp = pct(k["fp"])
    state.k_na = pct(k["na"])
    state.k_res = pct(k["res"])
    state.k_sla = f"{k['breach'] * 100:.0f}%" if k["breach"] == k["breach"] else "—"
    state.k_rules = f"{k['r80']} из {k['rules']}"


    m = S.monthly(d)
    state.fig_month = _layout(
        px.bar(
            m, x="month", y="n", color="verdict",
            color_discrete_map=COLORS,
            category_orders={"verdict": S.VERDICT_ORDER},
            labels={"month": "Месяц", "n": "Карточек", "verdict": "Вердикт"},
            title="Поток карточек по месяцам и вердиктам",
        )
    )


    r = S.rules(d, current_min_n)
    if r.empty:
        state.fig_rules = _layout(go.Figure().update_layout(title="Нет правил при данном пороге"))
        state.fig_top = _layout(go.Figure().update_layout(title="—"))
    else:
        state.fig_rules = _layout(
            px.scatter(
                r, x="n", y="fp_share", log_x=True, hover_name="rule",
                size="fp_cards", color="fp_share", color_continuous_scale="RdYlGn_r",
                labels={"n": "Карточек (log)", "fp_share": "Доля FP, %"},
                title="Правила: объём × доля FP (верхний правый угол — шум)",
            )
        )
        state.fig_rules.update_xaxes(
            type="log", tickvals=[10, 100, 1000, 10000], ticktext=["10", "100", "1k", "10k"]
        )

        top = r.sort_values("fp_cards", ascending=False).head(10).iloc[::-1]
        state.fig_top = _layout(
            px.bar(
                top, x="fp_cards", y="rule", orientation="h",
                color="fp_share", color_continuous_scale="RdYlGn_r", text="fp_share",
                labels={"fp_cards": "FP-карточек", "rule": "", "fp_share": "% FP"},
                title="Топ-10 правил по числу FP-карточек",
            ).update_traces(texttemplate="%{text:.1f}%", textposition="auto")
        )


    c = S.sla_by_crit(d)
    state.fig_crit = _layout(
        px.bar(
            c, x="crit", y="breach", text="breach", color="crit",
            color_discrete_sequence=["#d64545", "#e8a33d", "#2b5fd9", "#9aa3b2"],
            labels={"crit": "Критичность", "breach": "% просрочен SLA"},
            title="Просрочен SLA взятия в работу, % по критичности",
        ).update_traces(texttemplate="%{text:.1f}%", textposition="outside"),
        340,
    )
    state.fig_hour = _layout(
        px.line(
            S.sla_by_hour(d), x="hour", y="breach", markers=True,
            labels={"hour": "Час создания", "breach": "% просрочен SLA"},
            title="Просрочен SLA, % по часу создания",
        ),
        340,
    )
    state.tbl_dur = S.dur_by_verdict(d)


    h = S.heat(d)
    state.fig_heat = _layout(
        go.Figure(
            go.Heatmap(
                z=h.values, x=list(h.columns),
                y=["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"],
                colorscale="Blues", colorbar=dict(title="карточек"),
            )
        ).update_layout(
            title="Поступление карточек: день недели × час",
            yaxis_autorange="reversed",
        ),
        340,
    )


    a = S.analysts(d)
    if a.empty:
        state.fig_an = _layout(go.Figure().update_layout(title="Нет данных по аналитикам"))
    else:
        h_an = max(400, 24 * len(a) + 120)
        state.fig_an = _layout(
            go.Figure([
                go.Bar(name="False Positive", y=a.label, x=a.fp, orientation="h", marker_color=COLORS["False Positive"]),
                go.Bar(name="Не указан", y=a.label, x=a.na, orientation="h", marker_color=COLORS["Не указан"]),
                go.Bar(name="Остальное", y=a.label, x=a.other, orientation="h", marker_color=COLORS["Решён"]),
            ]).update_layout(
                barmode="stack",
                title="Структура вердиктов по аналитикам (обезличено), %",
                xaxis=dict(range=[0, 100]),
                yaxis=dict(automargin=True),
            ),
            h_an,
        )


def on_filter(state):
    update(state)
    notify(state, "i", "Дашборд обновлён")


def on_init(state):
    update(state)


with tgb.Page() as page:
    tgb.text("# Карточки SOC: что в потоке и где теряется время", mode="md")

    with tgb.layout(columns="1 1"):
        tgb.selector(
            value="{org}", lov="{ORGS}", dropdown=True, label="Организация", on_change=on_filter,
        )

        with tgb.part():
            tgb.text("Мин. карточек на правило (вопрос 2)", mode="md", class_name="mb-2")
            tgb.slider(
                value="{min_n}", min=50, max=2000, step=50, on_change=on_filter,
            )


    with tgb.layout(columns="1 1 1 1 1 1"):
        with tgb.part():
            tgb.text("Карточек", mode="md", class_name="text-sm text-gray-400")
            tgb.text("{k_n}", mode="md", class_name="text-2xl font-bold")
            
        with tgb.part():
            tgb.text("FP", mode="md", class_name="text-sm text-gray-400")
            tgb.text("{k_fp}", mode="md", class_name=f"text-2xl font-bold text-[{COLORS['False Positive']}]")
            
        with tgb.part():
            tgb.text("«Не указан»", mode="md", class_name="text-sm text-gray-400")
            tgb.text("{k_na}", mode="md", class_name=f"text-2xl font-bold text-[{COLORS['Не указан']}]")
            
        with tgb.part():
            tgb.text("Решено", mode="md", class_name="text-sm text-gray-400")
            tgb.text("{k_res}", mode="md", class_name=f"text-2xl font-bold text-[{COLORS['Решён']}]")
            
        with tgb.part():
            tgb.text("SLA просрочен", mode="md", class_name="text-sm text-gray-400")
            tgb.text("{k_sla}", mode="md", class_name=f"text-2xl font-bold text-[{COLORS['Другое']}]")
            
        with tgb.part():
            tgb.text("Правил на 80% потока", mode="md", class_name="text-sm text-gray-400")
            tgb.text("{k_rules}", mode="md", class_name="text-2xl font-bold")


    tgb.text("## 1. Из чего состоит поток и как он менялся?", mode="md")
    tgb.chart(figure="{fig_month}")


    tgb.text("## 2. Какие правила дают больше всего шума?", mode="md")
    with tgb.layout(columns="1 1"):
        tgb.chart(figure="{fig_rules}")
        tgb.chart(figure="{fig_top}")


    tgb.text("## 3. Успевают ли взять в работу самое важное?", mode="md")
    with tgb.layout(columns="1 1"):
        tgb.chart(figure="{fig_crit}")
        tgb.chart(figure="{fig_hour}")
    
    tgb.text("### Время до закрытия по вердиктам", mode="md")
    tgb.table(data="{tbl_dur}", show_all=True, width="100%")
    tgb.text(
        "Время до закрытия — календарное (создана → закрыта), не трудозатраты аналитика.",
        mode="md",
    )


    tgb.text("## 4. Когда приходит нагрузка?", mode="md")
    tgb.chart(figure="{fig_heat}")


    tgb.text("## 5. Одинаково ли аналитики закрывают карточки?", mode="md")
    tgb.chart(figure="{fig_an}")

if __name__ == "__main__":
    gui = Gui(page)
    gui.run(title="SOC dashboard", use_reloader=False, port=5000, on_init=on_init)