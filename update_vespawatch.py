"""Refresh VespaWatch analysis outputs from the INBO VespaWatch service."""
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import requests

QUERY_URL = (
    "https://gisservices.inbo.be/arcgis/rest/services/"
    "VespaWatch/VespaWatch_view/FeatureServer/0/query"
)
MUNICIPALITIES_URL = (
    "https://geodata.antwerpen.be/arcgis/rest/services/P_Publiek/P_basemap/MapServer/8/query"
)
PROVINCES = ["Antwerpen", "Limburg", "Oost-Vlaanderen", "Vlaams-Brabant", "West-Vlaanderen"]
AREAS_KM2 = {
    "Antwerpen": 2867,
    "Limburg": 2427,
    "Oost-Vlaanderen": 3007,
    "Vlaams-Brabant": 2106,
    "West-Vlaanderen": 3197,
}
MONTHS = [
    "januari", "februari", "maart", "april", "mei", "juni",
    "juli", "augustus", "september", "oktober", "november", "december",
]
ROOT = Path(__file__).resolve().parent
MUNICIPALITIES_CACHE = ROOT / "data" / "gemeentegrenzen.geojson"
HEATMAP_HTML = ROOT / "heatmap_vespawatch.html"
ANALYSIS_HTML = ROOT / "analysis_vespawatch.html"
SITE_NAV = (
    '<a href="index.html">Hive weight</a> | '
    '<a href="asian_hornet_observations.html">Asian Hornet map</a> | '
    '<a href="analysis_vespawatch.html">VespaWatch analysis</a> | '
    '<a href="heatmap_vespawatch.html">Flanders heatmap</a>'
)
SITE_NAV_CHROME = (
    '<style>'
    'html,body{margin:0;background:#fff;color:#696969;'
    'font-family:"Courier New",monospace;font-size:10px}'
    '.site-nav{text-align:center;padding:8px 12px 12px;line-height:1.6}'
    '.site-nav a{color:#447adb;text-decoration:none}'
    '.site-nav a:hover{text-decoration:underline}'
    '</style>'
    f'<nav class="site-nav">{SITE_NAV}</nav>'
)


def _with_site_nav(html: str, dark: bool = False) -> str:
    if 'class="site-nav"' in html:
        return html
    chrome = (
        '<style>'
        '.site-nav{text-align:center;padding:8px 12px 12px;line-height:1.6;'
        'font-family:"Courier New",monospace;font-size:10px}'
        '.site-nav a{color:#58a6ff;text-decoration:none}'
        '.site-nav a:hover{text-decoration:underline}'
        '</style>'
        f'<nav class="site-nav">{SITE_NAV}</nav>'
        if dark else SITE_NAV_CHROME
    )
    if "</body>" in html:
        return html.replace("</body>", chrome + "</body>", 1)
    return html + chrome


def fetch() -> pd.DataFrame:
    params = {
        "where": (
            # "validatie_status_consensus IN ('goedgekeurd', 'onzeker') "
            # "AND nest_type IS NOT NULL"
            "1=1"
        ),
        "outFields": (
            "OBJECTID,provincie,gemeente,breedtegraad,lengtegraad,"
            "nest_type,melding_observatie_datum"
        ),
        "returnGeometry": "false",
        "resultRecordCount": 2000,
        "f": "json",
    }
    records, offset = [], 0
    while True:
        params["resultOffset"] = offset
        response = requests.get(QUERY_URL, params=params, timeout=120)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        features = result.get("features", [])
        records.extend(feature["attributes"] for feature in features)
        if not features or not result.get("exceededTransferLimit", False):
            break
        offset += len(features)

    df = pd.DataFrame(records).fillna("")
    df["datum"] = pd.to_datetime(df["melding_observatie_datum"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datum"])
    df["nest_type"] = df["nest_type"].astype(str).str.replace("_", " ", regex=False)
    # return df[(df["nest_type"] != "") & (df["nest_type"] != "inactief leeg nest")].copy()         # !!!!!!!!!
    return df.copy()


def tables(df: pd.DataFrame) -> dict:
    df["jaar"], df["maand"] = df["datum"].dt.year, df["datum"].dt.month
    by_month = (
        df.pivot_table(index="jaar", columns="maand", values="OBJECTID", aggfunc="count", fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
        .rename(columns=dict(enumerate(MONTHS, 1)))
        .astype(int)
    )
    by_month["totaal"] = by_month.sum(axis=1)
    by_prov = (
        df[df["provincie"].isin(PROVINCES)]
        .pivot_table(index="jaar", columns="provincie", values="OBJECTID", aggfunc="count", fill_value=0)
        .reindex(columns=PROVINCES, fill_value=0)
        .astype(int)
    )
    by_prov["totaal"] = by_prov.sum(axis=1)
    today = pd.Timestamp.now(tz="Europe/Brussels")
    period = df[
        (df.datum.dt.month < today.month)
        | ((df.datum.dt.month == today.month) & (df.datum.dt.day <= today.day))
    ]
    ytd = (
        period[period["provincie"].isin(PROVINCES)]
        .pivot_table(index="jaar", columns="provincie", values="OBJECTID", aggfunc="count", fill_value=0)
        .reindex(columns=PROVINCES, fill_value=0)
        .astype(int)
    )
    years = range(2017, today.year + 1)
    ytd = ytd.reindex(years, fill_value=0)
    ytd["totaal"] = ytd.sum(axis=1)

    per_km2 = ytd.copy()
    for province in PROVINCES:
        per_km2[province] = (per_km2[province] / AREAS_KM2[province]).round(3)
    # per_km2["totaal"] = per_km2[PROVINCES].sum(axis=1).round(3)
    per_km2["totaal"] = ytd["totaal"] / sum(AREAS_KM2.values())
    return {
        "generated": str(today.date()),
        "as_of": f"1 januari t/m {today.day} {MONTHS[today.month - 1]}",
        "areas_km2": AREAS_KM2,
        "tables": {
            "month": by_month.reset_index().to_dict("records"),
            "province": by_prov.reset_index().to_dict("records"),
            "ytd": ytd.reset_index().to_dict("records"),
            "area_ytd": per_km2.reset_index().to_dict("records"),
        },
    }


def municipality_counts(df: pd.DataFrame, municipalities: gpd.GeoDataFrame) -> pd.Series:
    counts = (
        df["gemeente"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    unmatched = counts.index.difference(municipalities["NAAM"])
    if unmatched.empty:
        return counts

    remaining = df[df["gemeente"].astype(str).str.strip().isin(unmatched)].copy()
    remaining = remaining[remaining["breedtegraad"].notna() & remaining["lengtegraad"].notna()]
    if remaining.empty:
        return counts

    points = gpd.GeoDataFrame(
        remaining,
        geometry=gpd.points_from_xy(remaining["lengtegraad"], remaining["breedtegraad"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        municipalities[["NAAM", "geometry"]],
        how="left",
        predicate="within",
    )
    spatial_counts = joined.groupby("NAAM")["OBJECTID"].count()
    matched = counts.drop(unmatched, errors="ignore")
    return pd.concat([matched, spatial_counts]).groupby(level=0).sum()


def load_municipalities() -> gpd.GeoDataFrame:
    if not MUNICIPALITIES_CACHE.exists():
        params = {
            "where": "1=1",
            "outFields": "NAAM,NISCODE",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        response = requests.get(MUNICIPALITIES_URL, params=params, timeout=120)
        response.raise_for_status()
        MUNICIPALITIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MUNICIPALITIES_CACHE.write_text(response.text, encoding="utf-8")

    municipalities = gpd.read_file(MUNICIPALITIES_CACHE)
    municipalities["NAAM"] = municipalities["NAAM"].astype(str).str.strip()
    municipalities["NISCODE"] = municipalities["NISCODE"].astype(str).str.strip()

    # Reproject to Belgian Lambert 72 (EPSG:31370) for accurate area calculation
    municipalities_proj = municipalities.to_crs("EPSG:31370")
    municipalities["area_km2"] = municipalities_proj.geometry.area / 1_000_000

    municipalities["geometry"] = municipalities.geometry.simplify(0.001, preserve_topology=True)
    return municipalities


def write_heatmap(df: pd.DataFrame) -> None:
    today = pd.Timestamp.now(tz="Europe/Brussels")
    years = range(2017, today.year + 1)

    # Load municipalities and calculate area ONCE (Optimization)
    municipalities = load_municipalities()
    geojson = json.loads(municipalities.to_json())

    # Prepare dataframe with year column for the animation slider
    df_plot = df.copy()
    df_plot["jaar"] = df_plot["datum"].dt.year

    frames_data = []
    for year in years:
        df_year = df_plot[df_plot["jaar"] == year]
        counts = municipality_counts(df_year, municipalities)

        year_data = municipalities[["NAAM", "NISCODE", "area_km2", "geometry"]].copy()
        year_data["observations"] = year_data["NAAM"].map(counts).fillna(0)

        # Calculate observations per km²
        year_data["obs_per_km2"] = year_data["observations"] / year_data["area_km2"]
        # Convert to string here specifically for the Plotly animation slider
        year_data["jaar"] = str(year)
        frames_data.append(year_data)

    # Combine all years into a single dataframe for the animated plot
    plot_data = pd.concat(frames_data, ignore_index=True)
    plot_data["obs_per_km2"] = plot_data["obs_per_km2"].round(3)
    plot_data["area_km2"] = plot_data["area_km2"].round(3)

    # Determine a dynamic max value for the color scale (adds 20% headroom)
    max_obs_per_km2 = plot_data["obs_per_km2"].max()
    # color_max = max(1.0, round(max_obs_per_km2 * 1.2, 1))
    color_max = 18

    fig = px.choropleth_map(
        plot_data,
        geojson=geojson,
        locations="NISCODE",
        featureidkey="properties.NISCODE",
        color="obs_per_km2",
        hover_name="NAAM",
        hover_data={
            "obs_per_km2": True,
            "observations": True,
            "area_km2": True,
            "NISCODE": False
        },
        color_continuous_scale="YlOrRd",
        range_color=(1, color_max),
        labels={
            "obs_per_km2": "waarnemingen per km²",
            "observations": "totaal waarnemingen",
            "area_km2": "oppervlakte (km²)"
        },
        animation_frame="jaar",                                                                     # Creates the interactive year slider
        title=(
            "VespaWatch waarnemingen per km² per gemeente in Vlaanderen<br>"
            "<sup>alle waarnemingen (dus incl. onzekere en niet gevalideerde en dubbele waarnemingen, lege nesten, ...</sup>"
        ),
        center={"lat": 51.0, "lon": 4.5},
        zoom=8,
        height=820,
    )
    fig.update_layout(
        font=dict(family="Courier New, monospace", size=10, color="#696969"),
        title_x=0.5,
        margin={"r": 0, "t": 70, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "waarnemingen / km²"},
    )
    fig.write_html(HEATMAP_HTML, include_plotlyjs="cdn")
    HEATMAP_HTML.write_text(_with_site_nav(HEATMAP_HTML.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Wrote {HEATMAP_HTML} (max {max_obs_per_km2:.2f} per km²).")


def update_analysis_html(payload: dict, path: Path = ANALYSIS_HTML) -> None:
    if not path.exists():
        return
    html = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\nconst fmt=", html)
    if not match:
        raise RuntimeError("Could not find embedded report data.")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = html[: match.start(1)] + encoded + html[match.end(1):]
    html = re.sub(
        r'<div class="badge" id="date">.*?</div>',
        '<div class="badge" id="date"></div>',
        html,
        count=1,
    )
    html = _with_site_nav(html, dark=True)
    path.write_text(html, encoding="utf-8")
    print(f"Updated {path}.")


def main() -> None:
    df = fetch()
    write_heatmap(df)
    update_analysis_html(tables(df))


if __name__ == "__main__":
    main()
