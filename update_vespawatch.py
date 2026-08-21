"""Refresh analysis_vespawatch.html from the INBO VespaWatch service."""
import json
import re
from pathlib import Path

import pandas as pd
import requests

QUERY_URL = (
    "https://gisservices.inbo.be/arcgis/rest/services/"
    "VespaWatch/VespaWatch_view/FeatureServer/0/query"
)
PROVINCES = ["Antwerpen", "Limburg", "Oost-Vlaanderen", "Vlaams-Brabant", "West-Vlaanderen"]
AREAS_KM2 = {"Antwerpen": 2867, "Limburg": 2427, "Oost-Vlaanderen": 3007, "Vlaams-Brabant": 2106, "West-Vlaanderen": 3197}
MONTHS = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"]


def fetch() -> pd.DataFrame:
    params = {"where": "validatie_status_consensus IN ('goedgekeurd', 'onzeker') AND nest_type IS NOT NULL", "outFields": "OBJECTID,provincie,nest_type,melding_observatie_datum", "returnGeometry": "false", "resultRecordCount": 2000, "f": "json"}
    records, offset = [], 0
    while True:
        params["resultOffset"] = offset
        response = requests.get(QUERY_URL, params=params, timeout=120)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        features = result.get("features", [])
        records.extend(f["attributes"] for f in features)
        if not features or not result.get("exceededTransferLimit", False):
            break
        offset += len(features)
    df = pd.DataFrame(records).fillna("")
    df["datum"] = pd.to_datetime(df["melding_observatie_datum"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datum"])
    df["nest_type"] = df["nest_type"].astype(str).str.replace("_", " ", regex=False)
    return df[(df["nest_type"] != "") & (df["nest_type"] != "inactief leeg nest")].copy()


def tables(df: pd.DataFrame) -> dict:
    df["jaar"], df["maand"] = df["datum"].dt.year, df["datum"].dt.month
    by_month = df.pivot_table(index="jaar", columns="maand", values="OBJECTID", aggfunc="count", fill_value=0).reindex(columns=range(1, 13), fill_value=0).rename(columns=dict(enumerate(MONTHS, 1))).astype(int)
    by_month["totaal"] = by_month.sum(axis=1)
    by_prov = df[df["provincie"].isin(PROVINCES)].pivot_table(index="jaar", columns="provincie", values="OBJECTID", aggfunc="count", fill_value=0).reindex(columns=PROVINCES, fill_value=0).astype(int)
    by_prov["totaal"] = by_prov.sum(axis=1)
    today = pd.Timestamp.now(tz="Europe/Brussels")
    period = df[(df.datum.dt.month < today.month) | ((df.datum.dt.month == today.month) & (df.datum.dt.day <= today.day))]
    ytd = period[period["provincie"].isin(PROVINCES)].pivot_table(index="jaar", columns="provincie", values="OBJECTID", aggfunc="count", fill_value=0).reindex(columns=PROVINCES, fill_value=0).astype(int)
    # Always include every year from 2017 through the current year.
    years = range(2017, today.year + 1)
    ytd = ytd.reindex(years, fill_value=0)
    ytd["totaal"] = ytd.sum(axis=1)
    per_km2 = ytd.copy()
    for province in PROVINCES:
        per_km2[province] = (per_km2[province] / AREAS_KM2[province]).round(3)
    per_km2["totaal"] = per_km2[PROVINCES].sum(axis=1).round(3)
    return {"generated": str(today.date()), "as_of": f"1 januari t/m {today.day} {MONTHS[today.month - 1]}", "areas_km2": AREAS_KM2, "tables": {"month": by_month.reset_index().to_dict("records"), "province": by_prov.reset_index().to_dict("records"), "ytd": ytd.reset_index().to_dict("records"), "area_ytd": per_km2.reset_index().to_dict("records")}}


def main() -> None:
    path = Path("analysis_vespawatch.html")
    html = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\nconst fmt=", html)
    if not match:
        raise RuntimeError("Could not find embedded report data.")
    payload = json.dumps(tables(fetch()), ensure_ascii=False, separators=(",", ":"))
    html = html[:match.start(1)] + payload + html[match.end(1):]
    html = re.sub(r"<div class=\"badge\" id=\"date\">.*?</div>", "<div class=\"badge\" id=\"date\"></div>", html, count=1)
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
