"""Refresh VespaWatch analysis outputs from the INBO VespaWatch service."""
import json
import re
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import requests
import plotly.express as px
# px.set_mapbox_access_token("pk.eyJ1IjoicHZhbmhldmVsIiwiYSI6ImNqZnZnanZjcjR3ZnEycXFmaTFycmx4MzAifQ.0jurH4Sa_VFi8RrbTL_bGA")
# import plotly.io as pio
# pio.set_mapbox_access_token(os.environ["MAPBOX_TOKEN"])


QUERY_URL = (
    "https://gisservices.inbo.be/arcgis/rest/services/"
    "VespaWatch/VespaWatch_view/FeatureServer/0/query"
)
# MUNICIPALITIES_URL = (                                                                              # does not include Antwerp!
#     "https://geodata.antwerpen.be/arcgis/rest/services/P_Publiek/P_basemap/MapServer/8/query"
# )
MUNICIPALITIES_URL = ("https://geo.api.vlaanderen.be/VRBG/ogc/features/v1/collections/Refgem/items")

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
RESULTS = [
    'niet_bestrijdbaar',
    'succesvol_bestreden',
    'niet_succesvol_bestreden',
    'ongekend',
]
ROOT = Path(__file__).resolve().parent
# MUNICIPALITIES_CACHE = ROOT / "data" / "gemeentegrenzen.geojson"
MUNICIPALITIES_CACHE = ROOT / "data" / "belgische_gemeenten_vlaanderen.geojson"
HEATMAP_HTML = ROOT / "heatmap_vespawatch.html"
HEATMAP_TEMPLATE_HTML = ROOT / "heatmap_template.html"
ANALYSIS_HTML = ROOT / "analysis_vespawatch.html"
HEATMAP1_HTML = ROOT / "heatmap_beekeepers.html"
HEATMAP1_TEMPLATE_HTML = ROOT / "heatmap1_template.html"
SITE_NAV = (
    '<a href="index.html">Hive weight</a> | '
    '<a href="asian_hornet_observations.html">AH observations near Hofstade</a> | '
    '<a href="analysis_vespawatch.html">AH observations tables</a> | '
    '<a href="heatmap_vespawatch.html">AH observations heatmap</a> | '
    '<a href="heatmap_beekeepers.html">Beekeepers heatmap</a>'
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


# def _with_site_nav(html: str, dark: bool = False) -> str:
#     if 'class="site-nav"' in html:
#         return html
#     chrome = (
#         '<style>'
#         '.site-nav{text-align:center;padding:8px 12px 12px;line-height:1.6;'
#         'font-family:"Courier New",monospace;font-size:10px}'
#         '.site-nav a{color:#58a6ff;text-decoration:none}'
#         '.site-nav a:hover{text-decoration:underline}'
#         '</style>'
#         f'<nav class="site-nav">{SITE_NAV}</nav>'
#         if dark else SITE_NAV_CHROME
#     )
#     if "</body>" in html:
#         return html.replace("</body>", chrome + "</body>", 1)
#     return html + chrome


def fetch() -> pd.DataFrame:
    params = {
        "where": (
            # "validatie_status_consensus IN ('goedgekeurd', 'onzeker') "
            # "AND nest_type IS NOT NULL"
            "1=1"
        ),
        "outFields": (
            "OBJECTID,provincie,gemeente,breedtegraad,lengtegraad,"
            "nest_type,melding_observatie_datum,bestrijding_resultaat"
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

    by_result = df.pivot_table(
        index="jaar", columns="bestrijding_resultaat", values="OBJECTID", aggfunc="count", fill_value=0
    ).reindex(columns=RESULTS, fill_value=0).astype(int)
    by_result["totaal"] = by_result.sum(axis=1)
    by_result.index.name = "jaar"
    by_result.columns.name = None

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
    per_km2["totaal"] = ytd["totaal"] / sum(AREAS_KM2.values())

    allowed_types = ["inactief leeg nest", "actief secundair nest"]
    df_filtered = df[df["nest_type"].isin(allowed_types)].copy()
    df_filtered["latitude"] = pd.to_numeric(df_filtered["breedtegraad"], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(df_filtered["lengtegraad"], errors="coerce")
    df_clean = df_filtered.dropna(subset=["latitude", "longitude", "datum"]).copy()

    def get_hornet_season(date):
        if date.month >= 6:
            return f"{date.year}-{date.year + 1}"
        else:
            return f"{date.year - 1}-{date.year}"

    df_clean["season"] = df_clean["datum"].apply(get_hornet_season)
    earth_radius_meters = 6371000
    max_distance_meters = 50
    eps_rad = max_distance_meters / earth_radius_meters
    clustered_records = []
    for season_name, df_season in df_clean.groupby("season"):
        if len(df_season) < 2:
            continue
        coords_rad = np.deg2rad(df_season[["latitude", "longitude"]].values)
        db = DBSCAN(eps=eps_rad, min_samples=2, metric="haversine")
        cluster_labels = db.fit_predict(coords_rad)
        df_season_result = df_season.copy()
        df_season_result["cluster_id"] = cluster_labels
        df_clusters = df_season_result[df_season_result["cluster_id"] != -1].copy()
        if not df_clusters.empty:
            df_clusters["unique_cluster_name"] = (
                df_clusters["season"]
                + "_Cluster_"
                + df_clusters["cluster_id"].astype(str)
            )
            clustered_records.append(df_clusters)
    if clustered_records:
        df_final_clusters = pd.concat(clustered_records)
        cluster_counts = (
            df_final_clusters.groupby("unique_cluster_name")
            .size()
            .reset_index(name="cluster_grootte")
        )
        df_final_clusters = df_final_clusters.merge(
            cluster_counts, on="unique_cluster_name"
        )
        df_final_clusters = df_final_clusters.sort_values(
            ["season", "unique_cluster_name"]
        )
    obs_per_season = df_clean.groupby("season").size().rename("aantal waarnemingen")
    clusters_stats = (
        df_final_clusters[df_final_clusters["cluster_grootte"] > 1]
        .groupby("season")
        .agg(
            aantal_clusters=("unique_cluster_name", "nunique"),  # Telt unieke cluster namen
            gemiddelde_grootte=("cluster_grootte", "mean")       # Berekent het gemiddelde van de grootte
        )
        .rename(columns={"aantal_clusters": "aantal 50 m radius clusters groter dan 1", "gemiddelde_grootte": "gemiddelde clustergrootte"})
        .astype({"aantal 50 m radius clusters groter dan 1": int})
        .round({"gemiddelde clustergrootte": 1})
    )
    clusters_stats = clusters_stats.join(obs_per_season).fillna(0)
    clusters_stats["aantal waarnemingen"] = clusters_stats["aantal waarnemingen"].astype(int)
    clusters_stats.index.name = "jaar"
    clusters_stats.columns.name = None

    return {
        "generated": str(today.date()),
        "as_of": f"1 januari t/m {today.day} {MONTHS[today.month - 1]}",
        "areas_km2": AREAS_KM2,
        "tables": {
            "result": by_result.reset_index().to_dict("records"),
            "month": by_month.reset_index().to_dict("records"),
            "province": by_prov.reset_index().to_dict("records"),
            "ytd": ytd.reset_index().to_dict("records"),
            "area_ytd": per_km2.reset_index().to_dict("records"),
            "clusters_stats": clusters_stats.reset_index().to_dict("records"),
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
    # Forceer herdownload als het bestand corrupt is of HTML-tags bevat van eerdere pogingen
    is_corrupted = False
    if MUNICIPALITIES_CACHE.exists():
        with open(MUNICIPALITIES_CACHE, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if "<!DOCTYPE html>" in first_line or "<html" in first_line:
                is_corrupted = True

    if not MUNICIPALITIES_CACHE.exists() or is_corrupted:
        # Vraag direct om GeoJSON-formaat en zet de limiet hoog genoeg (Vlaanderen heeft ~280+ gemeenten)
        params = {
            "f": "application/geo+json",
            "limit": "500",
        }
        response = requests.get(MUNICIPALITIES_URL, params=params, timeout=120)
        response.raise_for_status()

        # Extra veiligheidscontrole op Content-Type
        if "html" in response.headers.get("Content-Type", "").lower():
            raise ValueError(
                "De server gaf onverwacht een HTML-pagina terug in plaats van GeoJSON."
            )

        MUNICIPALITIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MUNICIPALITIES_CACHE.write_text(response.text, encoding="utf-8")

    # Geopandas leest het schone GeoJSON-bestand in
    municipalities = gpd.read_file(MUNICIPALITIES_CACHE)

    # Kolomnamen standaardiseren naar hoofdletters voor compatibiliteit met de rest van je code
    if "NAAM" not in municipalities.columns and "naam" in municipalities.columns:
        municipalities = municipalities.rename(columns={"naam": "NAAM"})
    if (
        "NISCODE" not in municipalities.columns
        and "niscode" in municipalities.columns
    ):
        municipalities = municipalities.rename(columns={"niscode": "NISCODE"})

    # Data opschonen
    municipalities["NAAM"] = municipalities["NAAM"].astype(str).str.strip()
    municipalities["NISCODE"] = (
        municipalities["NISCODE"].astype(str).str.strip()
    )

    # Projecteer naar Lambert 72 (EPSG:31370) voor een nauwkeurige oppervlakteberekening
    municipalities_proj = municipalities.to_crs("EPSG:31370")
    municipalities["area_km2"] = municipalities_proj.geometry.area / 1_000_000

    # Geometrie vereenvoudigen voor soepele rendering in de VespaWatch animatie/slider
    municipalities["geometry"] = municipalities.geometry.simplify(
        0.001, preserve_topology=True
    )
    return municipalities


def write_heatmap(df: pd.DataFrame) -> None:
    today = pd.Timestamp.now(tz="Europe/Brussels")
    years = range(2017, today.year + 1)

    # Load municipalities and calculate area ONCE (Optimization)
    municipalities = load_municipalities()
    # Selecteer alle kolommen die een datum/tijd bevatten en zet ze om naar tekst (string)
    for col in municipalities.select_dtypes(include=["datetime", "datetimetz"]).columns:
        municipalities[col] = municipalities[col].astype(str)

    # Nu werkt het omzetten naar JSON zonder foutmeldingen
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

    fig = px.choropleth_mapbox(
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
            f"Waarnemingen van AH nesten per km² per gemeente in Vlaanderen  (geüpdatet op {today:%Y-%m-%d %H:%M:%S%z})<br>"
            "<sup>Bron: INBO VespaWatch FeatureServer<br>"
            "alle waarnemingen (dus incl. de niet gevalideerde, de onzekere, de dubbele, de lege nesten, enz.</sup>"
        ),
        center={"lat": 51.2, "lon": 4.3},
        zoom=7,
        # height=820,
        mapbox_style="open-street-map"
    )
    fig.update_layout(
        mapbox={
            "accesstoken": os.environ["MAPBOX_TOKEN"],
            "style": "open-street-map"
        }
    )
    fig.update_layout(
        font=dict(family="Courier New, monospace", size=10, color="#696969"),
        title_x=0.5,
        margin={"r": 0, "t": 70, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "waarnemingen / km²"},
    )
    # fig.write_html(HEATMAP_HTML, include_plotlyjs="cdn")
    # HEATMAP_HTML.write_text(_with_site_nav(HEATMAP_HTML.read_text(encoding="utf-8")), encoding="utf-8")

    # plotly_div = fig.to_html(include_plotlyjs=False, full_html=False)
    plotly_div = fig.to_html(include_plotlyjs="cdn", full_html=False)
    with open(HEATMAP_TEMPLATE_HTML, "r", encoding="utf-8") as f:
        html_template = f.read()
    final_html = html_template.replace("{content}", plotly_div)
    with open(HEATMAP_HTML, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Wrote {HEATMAP_HTML} (max {max_obs_per_km2:.2f} per km²).")


def beekeepers_heatmap() -> None:
    today = pd.Timestamp.now(tz="Europe/Brussels")
    # Load municipalities and calculate area ONCE (Optimization)
    municipalities = load_municipalities()
    # Selecteer alle kolommen die een datum/tijd bevatten en zet ze om naar tekst (string)
    for col in municipalities.select_dtypes(include=["datetime", "datetimetz"]).columns:
        municipalities[col] = municipalities[col].astype(str)

    # Nu werkt het omzetten naar JSON zonder foutmeldingen
    geojson = json.loads(municipalities.to_json())
    df_plot = pd.read_csv("inter_actieve_actoren_NL.csv", encoding="ISO-8859-1")
    df_plot = df_plot[df_plot["PAP Omschrijving"] == "Imker - houden bijen"]
    cols = [
        'OP Uniek Nr Id ',
        'LNO Uniek Nr ',
        # 'PAP Id',
        # 'PAP Omschrijving',
        # 'PAP ACT Code',
        # 'PAP ACT Omschrijving',
        # 'PAP PLA Code',
        # 'PAP PLA Omschrijving',
        # 'PAP PRD Code',
        # 'PAP PRD Omschrijving',
        # 'TYP ERK Erkenning code',
        # 'TYP ERK omschrijving',
        # 'TYP ERK Erkenning vorm Omschrijving',
        # 'TYP ERK Erkenning Vorm Code',
        'PC Postcode ',
        'GEM Naam ',
        'PR Naam ',
        'ERK Nummer ',
        'ERK Begindatum ',
        'Datum vandaag',
    ]
    df_plot = df_plot[cols]
    df_plot = df_plot.rename(columns={"GEM Naam ": "NAAM"})
    df_plot = df_plot[df_plot["NAAM"].isin(municipalities["NAAM"])]
    counts = (
        df_plot["NAAM"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    plot_data = municipalities[["NAAM", "NISCODE", "area_km2", "geometry"]].copy()
    plot_data["beekeepers"] = plot_data["NAAM"].map(counts).fillna(0)
    plot_data["beeks_per_km2"] = plot_data["beekeepers"] / plot_data["area_km2"]
    plot_data["beeks_per_km2"] = plot_data["beeks_per_km2"].round(3)
    plot_data["area_km2"] = plot_data["area_km2"].round(3)
    max_beeks_per_km2 = plot_data["beeks_per_km2"].max()
    color_max = max(1.0, round(max_beeks_per_km2 * 1.2, 1))

    fig = px.choropleth_mapbox(
        plot_data,
        geojson=geojson,
        locations="NISCODE",
        featureidkey="properties.NISCODE",
        color="beeks_per_km2",
        hover_name="NAAM",
        hover_data={
            "beeks_per_km2": True,
            "beekeepers": True,
            "area_km2": True,
            "NISCODE": False
        },
        color_continuous_scale="YlOrRd",
        range_color=(0, color_max),
        labels={
            "beeks_per_km2": "imkers per km²",
            "beekeepers": "aantal imkers",
            "area_km2": "oppervlakte (km²)"
        },
        title=(
            f"Aantal imkers per km² per gemeente in Vlaanderen  (geüpdatet op {today:%Y-%m-%d %H:%M:%S%z})<br>"
            "<sup>Bron: FAVV inter_actieve_actoren_NL.csv</sup>"
        ),
        center={"lat": 51.2, "lon": 4.3},
        zoom=7,
        # height=820,
        mapbox_style="open-street-map"
    )
    fig.update_layout(
        mapbox={
            "accesstoken": os.environ["MAPBOX_TOKEN"],
            "style": "open-street-map"
        }
    )
    fig.update_layout(
        font=dict(family="Courier New, monospace", size=10, color="#696969"),
        title_x=0.5,
        margin={"r": 0, "t": 70, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "imkers / km²"},
    )
    plotly_div = fig.to_html(include_plotlyjs="cdn", full_html=False)
    with open(HEATMAP1_TEMPLATE_HTML, "r", encoding="utf-8") as f:
        html_template = f.read()
    final_html = html_template.replace("{content}", plotly_div)
    with open(HEATMAP1_HTML, "w", encoding="utf-8") as f:
        f.write(final_html)


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
    # html = _with_site_nav(html, dark=True)
    path.write_text(html, encoding="utf-8")
    print(f"Updated {path}.")


def main() -> None:
    df = fetch()
    write_heatmap(df)
    update_analysis_html(tables(df))
    beekeepers_heatmap()


if __name__ == "__main__":
    main()
