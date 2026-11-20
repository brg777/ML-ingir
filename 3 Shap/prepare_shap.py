"""
prepare_shap.py — БЕЗ модели. Берём готовые SHAP из shap_cards.csv.
Запуск:  python prepare_shap.py
Результат: shap_artifacts.pkl
"""
import pickle
import numpy as np
import pandas as pd

GROUP_IDS = [523950, 523961, 527330, 521060, 521667, 528067, 521952, 527608]
BASE = -8.852417

shap_df = pd.read_csv("shap_cards.csv", sep=";", encoding="utf-8-sig")
shap_df["EntityId"] = shap_df["EntityId"].astype(int)

shap_df["__order"] = shap_df["EntityId"].map({e: i for i, e in enumerate(GROUP_IDS)})
shap_team = shap_df[shap_df.EntityId.isin(GROUP_IDS)].sort_values("__order").reset_index(drop=True)

META = {"EntityId", "y_true", "proba", "outcome", "base", "__order"}
feature_names = [c for c in shap_team.columns if c not in META]
sv_group = shap_team[feature_names].values.astype(float)
entity_ids = shap_team["EntityId"].values.astype(int)
y_true = shap_team["y_true"].values.astype(int)
proba = shap_team["proba"].values.astype(float)
outcome = shap_team["outcome"].tolist()

import pyarrow.parquet as pq

parquet_file = pq.ParquetFile("cards_raw.parquet")

raw_parts = []

for batch in parquet_file.iter_batches(batch_size=10000):
    chunk = batch.to_pandas()
    chunk["EntityId"] = pd.to_numeric(
        chunk["EntityId"], errors="coerce"
    )
    chunk = chunk[chunk["EntityId"].isin(GROUP_IDS)]
    if not chunk.empty:
        raw_parts.append(chunk)

if raw_parts:
    raw_team = pd.concat(raw_parts, ignore_index=True)
    raw_team["EntityId"] = raw_team["EntityId"].astype(int)
else:
    raise ValueError("Не найдены записи для указанных EntityId")

raw_team["__order"] = raw_team["EntityId"].map(
    {e: i for i, e in enumerate(GROUP_IDS)}
)

FEATURE_LABELS = {
    "Создан": "Время создания ((!) артефакт)",
    "Hostname атакующего": "Хост атакующего",
    "Hostname Цели": "Хост цели",
    "src": "Src (адрес/учётка)",
    "dst": "Dst (адрес/учётка)",
    "suser": "Учётка источника",
    "duser": "Учётка цели",
    "freq_Наименование": "Частота правила",
    "freq_Вендор": "Частота вендора",
    "len_Значение Инцидента (группировка)": "Размер группировки инцидента",
    "len_Значение Атаки (группировка)": "Размер группировки атак",
    "Алерт": "Текст алерта",
    "Команда": "Выполненная команда",
    "CMDLINE": "Командная строка",
    "commandName": "Имя утилиты",
    "processName": "Имя процесса",
    "Наименование": "Имя правила корреляции",
    "cmd_entropy": "Энтропия команды",
    "cmd_chars": "Длина команды (символы)",
    "cmd_words": "Длина команды (слова)",
    "cmd_has_b64": "Команда содержит base64",
    "cmd_has_url": "Команда содержит URL",
    "cmd_has_pipe": "Команда содержит pipe",
    "alert_chars": "Длина алерта",
    "alert_words": "Длина алерта (слова)",
    "event_age_days": "Возраст события (дни)",
    "event_duration_s": "Длительность события (сек)",
    "create_lag_s": "Задержка создания (сек)",
    "src_is_private": "Src в приватной сети",
    "dst_is_private": "Dst в приватной сети",
    "dst_external_count": "Внешних dst",
    "ip_count_src": "Кол-во src IP",
    "ip_count_dst": "Кол-во dst IP",
    "host_depth": "Глубина хоста",
    "host_is_server": "Хост — сервер",
    "is_business_hours": "Рабочие часы",
    "is_night": "Ночное время",
    "categoryMain": "Категория (main)",
    "categoryHigh": "Категория (high)",
    "categoryLow": "Категория (low)",
    "Пиктограмма IRP": "Пиктограмма IRP",
    "Критичность": "Критичность",
    "Критичность инцидента в источнике": "Критичность в источнике",
    "Источник инцидента": "Источник (SIEM)",
    "Вендор из сием": "Вендор SIEM",
    "Конвейер SIEM": "Конвейер SIEM",
    "Организация": "Организация",
    "object": "Тип объекта",
    "object.name": "Имя объекта",
    "objectValue": "Значение объекта",
    "objectLinks": "Связи объектов",
    "ProcessPath": "Путь процесса",
    "Имя родителя процесса": "Родительский процесс",
    "Путь родителя процесса": "Путь родителя",
    "len_src": "Длина src",
    "len_dst": "Длина dst",
    "len_suser": "Длина suser",
    "len_duser": "Длина duser",
    "len_CMDLINE": "Длина CMDLINE",
    "len_Команда": "Длина команды",
    "len_commandName": "Длина утилиты",
    "len_processName": "Длина процесса",
    "len_FilePath": "Длина пути",
    "len_FileName": "Длина файла",
    "len_objectLinks": "Длина связей",
    "len_Алерт": "Длина алерта",
    "path_depth": "Глубина пути",
    "path_in_temp": "Путь в Temp",
    "has_suser": "Есть suser",
    "has_duser": "Есть duser",
    "has_malicious": "Есть malicious-флаг",
    "n_rules": "Кол-во правил",
}

ARTIFACT_FEATURES = {
    "Создан", "Hostname атакующего", "Hostname Цели", "src", "dst",
    "suser", "duser", "freq_Наименование", "freq_Вендор",
    "len_Значение Инцидента (группировка)", "len_Значение Атаки (группировка)",
    "n_rules",
}

with open("shap_artifacts.pkl", "wb") as f:
    pickle.dump({
        "entity_ids": entity_ids,
        "feature_names": feature_names,
        "sv_group": sv_group,
        "y_true": y_true,
        "proba": proba,
        "outcome": outcome,
        "base": BASE,
        "df_raw_group": raw_team,
        "feature_labels": FEATURE_LABELS,
        "artifact_features": ARTIFACT_FEATURES,
    }, f)

print("✓ shap_artifacts.pkl готов")
print(f"  карточек: {len(entity_ids)}, признаков: {len(feature_names)}")
print(f"  Σ|SHAP| mean для группы: {np.abs(sv_group).mean():.3f}")