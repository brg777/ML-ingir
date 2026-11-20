"""Быстрая проверка: строит все графики в HTML и сохраняет."""
from pathlib import Path
import pandas as pd
from src import charts

scored = pd.read_parquet(Path("data/scored.parquet"))
out = Path("screenshots"); out.mkdir(exist_ok=True)

charts.chart_q1_degradation(scored).write_html(out / "q1.html")
charts.chart_q2_ranking_vs_operating(scored).write_html(out / "q2.html")
charts.chart_q3_load(scored).write_html(out / "q3.html")
for f in charts.SLICE_FIELDS:
    charts.chart_q4_slices(scored, f).write_html(
        out / f"q4_{f.replace(' ', '_')}.html")

import json
(out / "kpi.json").write_text(
    json.dumps(charts.kpi_tiles(scored), ensure_ascii=False, indent=2))

print("Готово:", list(out.glob("*.html")), "kpi.json")
