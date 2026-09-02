"""
Klad code voor update_vespawatch.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import requests

QUERY_URL = (
    "https://gisservices.inbo.be/arcgis/rest/services/"
    "VespaWatch/VespaWatch_view/FeatureServer/0/query"
)
PROVINCES = [
    "Antwerpen",
    "Limburg",
    "Oost-Vlaanderen",
    "Vlaams-Brabant",
    "West-Vlaanderen",
]
PROVINCE_AREAS_KM2 = {
    "Antwerpen": 2867,
    "Limburg": 2427,
    "Oost-Vlaanderen": 3007,
    "Vlaams-Brabant": 2106,
    "West-Vlaanderen": 3197,
}
MONTHS = list(range(1, 13))
MONTH_NAMES = {
    1: "januari", 2: "februari", 3: "maart", 4: "april",
    5: "mei", 6: "juni", 7: "juli", 8: "augustus",
    9: "september", 10: "oktober", 11: "november", 12: "december",
}
RESULTS = [
    'niet_bestrijdbaar',
    'succesvol_bestreden',
    'niet_succesvol_bestreden',
    'ongekend',
]


def gisservices_inbo() -> pd.DataFrame:
    """Fetch all approved/uncertain observations and return them as df.

    The province restriction from the original function is deliberately not
    used here: province is needed as a column for the second summary table.
    Pagination makes this work when the service has more than 2,000 records.
    """
    where = (
        # "validatie_status_consensus IN ('goedgekeurd', 'onzeker') "
        # "AND nest_type IS NOT NULL"
        "1=1"
    )
    fields = [
        "OBJECTID", "breedtegraad", "lengtegraad", "provincie", "gemeente",
        "nest_grootte", "nest_hoogte", "nest_locatie", "nest_type",
        "melding_observatie_datum", "bestrijding_datum", "bestrijding_resultaat",
        "bestrijder_naam", "bestrijding_product", "validatie_status_consensus",
        "bron_url", "GlobalID", "id_extern",
    ]
    params = {
        "where": where,
        "outFields": ",".join(fields),
        "returnGeometry": "false",
        "orderByFields": "OBJECTID",
        "resultRecordCount": 2000,
        "f": "json",
    }
    records = []
    offset = 0
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

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["OBJECTID", "province", "municipality", "nest_type", "datum"])

    df = df.fillna("")
    df = df.rename(columns={
        "breedtegraad": "latitude",
        "lengtegraad": "longitude",
        "bron_url": "url",
        "gemeente": "municipality",
        "provincie": "province",
    })
    df["OBJECTID"] = df["OBJECTID"].astype(str)
    df["datum"] = pd.to_datetime(
        df["melding_observatie_datum"], unit="ms", errors="coerce"
    )
    df["year"] = df["datum"].dt.year
    df["month"] = df["datum"].dt.month
    df["nest_type"] = df["nest_type"].astype(str).str.replace("_", " ", regex=False)
    return df


def create_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create year x month and year x province observation-count tables."""
    # valid = df[df["nest_type"].ne("inactief leeg nest") & df["nest_type"].ne("")].copy()
    valid = df

    by_result = valid.pivot_table(
        index="year", columns="bestrijding_resultaat", values="OBJECTID", aggfunc="count", fill_value=0
    ).reindex(columns=RESULTS, fill_value=0).astype(int)
    by_result["totaal"] = by_result.sum(axis=1)
    by_result.index.name = "jaar"
    by_result.columns.name = None

    by_month = valid.pivot_table(
        index="year", columns="month", values="OBJECTID", aggfunc="count", fill_value=0
    ).reindex(columns=MONTHS, fill_value=0).rename(columns=MONTH_NAMES).astype(int)
    by_month["totaal"] = by_month.sum(axis=1)
    by_month.index.name = "jaar"
    by_month.columns.name = None

    by_province = valid[valid["province"].isin(PROVINCES)].pivot_table(
        index="year", columns="province", values="OBJECTID", aggfunc="count", fill_value=0
    ).reindex(columns=PROVINCES, fill_value=0).astype(int)
    by_province["totaal"] = by_province.sum(axis=1)
    by_province.index.name = "jaar"
    by_province.columns.name = None

    # Compare each year over the same period as the current year: from
    # January 1 through today's month/day. This avoids comparing a partial
    # current year with complete historical years.
    today = pd.Timestamp.today()
    current_period = valid[
        (valid["month"] < today.month)
        | ((valid["month"] == today.month) & (valid["datum"].dt.day <= today.day))
    ]
    by_province_till_current_date = current_period[current_period["province"].isin(PROVINCES)].pivot_table(
        index="year",
        columns="province",
        values="OBJECTID",
        aggfunc="count",
        fill_value=0,
    ).reindex(columns=PROVINCES, fill_value=0).astype(int)
    by_province_till_current_date["totaal"] = by_province_till_current_date.sum(axis=1)
    by_province_till_current_date.index.name = "jaar"
    by_province_till_current_date.columns.name = None

    by_province_per_km2 = by_province_till_current_date.copy()
    for province in PROVINCES:
        by_province_per_km2[province] = (
            by_province_per_km2[province] / PROVINCE_AREAS_KM2[province]
        ).round(3)
    by_province_per_km2["totaal"] = (
        by_province_per_km2[PROVINCES].sum(axis=1).round(3)
    )

    return (
        by_result,
        by_month,
        by_province,
        by_province_till_current_date,
        by_province_per_km2,
    )


def find_duplicates(df):
    
    # 1. Filter direct op de gewenste nest_types en converteer coördinaten
    allowed_types = ["inactief leeg nest", "actief secundair nest"]
    df_filtered = df[df["nest_type"].isin(allowed_types)].copy()
    
    df_filtered["latitude"] = pd.to_numeric(df_filtered["latitude"], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(df_filtered["longitude"], errors="coerce")
    
    # Verwijder rijen zonder geldige coördinaten of datum
    df_clean = df_filtered.dropna(subset=["latitude", "longitude", "datum"]).copy()
    
    
    # 2. Functie om het hoornaarseizoen te bepalen (Juni t/m Mei volgend jaar)
    def get_hornet_season(date):
        if date.month >= 6:
            return f"{date.year}-{date.year + 1}"
        else:
            return f"{date.year - 1}-{date.year}"
    
    
    # Voeg de seizoenskolom toe
    df_clean["season"] = df_clean["datum"].apply(get_hornet_season)
    
    # Instellingen voor de BallTree (500 meter naar radialen)
    earth_radius_meters = 6371000
    max_distance_meters = 200
    radius_rad = max_distance_meters / earth_radius_meters
    
    # Lijst voor de resultaten
    all_season_duplicates = []
    
    # 3. Loop door elk uniek seizoen
    for season_name, df_season in df_clean.groupby("season"):
        if len(df_season) < 2:
            continue
    
        # Reset index om correct binnen de subset te mappen, behoud de originele index
        df_season_reset = df_season.reset_index(drop=False)
    
        # Converteer naar radialen
        coords_rad = np.deg2rad(df_season_reset[["latitude", "longitude"]].values)
    
        # Bouw en query de BallTree
        tree = BallTree(coords_rad, metric="haversine")
        indices = tree.query_radius(coords_rad, r=radius_rad)
    
        seen_pairs = set()
    
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if i != neighbor:
                    pair = tuple(sorted((i, neighbor)))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
    
                        row_a = df_season_reset.iloc[i]
                        row_b = df_season_reset.iloc[neighbor]
    
                        all_season_duplicates.append(
                            {
                                "season": season_name,
                                "index_a": row_a["index"],  # Originele index in df
                                "id_a": row_a["OBJECTID"],
                                "nest_type_a": row_a["nest_type"],
                                "datum_a": row_a["datum"],
                                "index_b": row_b[
                                    "index"
                                ],  # Originele index in df
                                "id_b": row_b["OBJECTID"],
                                "nest_type_b": row_b["nest_type"],
                                "datum_b": row_b["datum"],
                                "lat_a": row_a["latitude"],
                                "lon_a": row_a["longitude"],
                                "lat_b": row_b["latitude"],
                                "lon_b": row_b["longitude"],
                            }
                        )
    
    # 4. Resultaat omzetten naar een DataFrame
    df_duplicates = pd.DataFrame(all_season_duplicates)
    
    # Toon een samenvatting van de resultaten
    if not df_duplicates.empty:
        print("Aantal specifieke nest-duplicaten per hoornaarseizoen:")
        print(df_duplicates.groupby("season").size())
    else:
        print("Geen duplicaten gevonden binnen deze criteria.")
    
    return df_duplicates


def find_clusters():
    
    # 1. Filteren op nest_type en numerieke conversie
    allowed_types = ["inactief leeg nest", "actief secundair nest"]
    df_filtered = df[df["nest_type"].isin(allowed_types)].copy()
    
    df_filtered["latitude"] = pd.to_numeric(df_filtered["latitude"], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(df_filtered["longitude"], errors="coerce")
    
    df_clean = df_filtered.dropna(subset=["latitude", "longitude", "datum"]).copy()
    
    
    # 2. Functie voor hoornaarseizoen (juni t/m mei volgend jaar)
    def get_hornet_season(date):
        if date.month >= 6:
            return f"{date.year}-{date.year + 1}"
        else:
            return f"{date.year - 1}-{date.year}"
    
    
    df_clean["season"] = df_clean["datum"].apply(get_hornet_season)
    
    # 3. DBSCAN parameters instellen (100 meter naar radialen)
    earth_radius_meters = 6371000
    max_distance_meters = 50
    eps_rad = max_distance_meters / earth_radius_meters
    
    # Lijst om de rijen met cluster-informatie in op te slaan
    clustered_records = []
    
    # 4. Loop per seizoen en pas DBSCAN toe
    for season_name, df_season in df_clean.groupby("season"):
        # DBSCAN vereist minimaal 2 punten om een cluster te vormen (min_samples=2)
        if len(df_season) < 2:
            continue
    
        # Converteer coördinaten naar radialen
        coords_rad = np.deg2rad(df_season[["latitude", "longitude"]].values)
    
        # Start DBSCAN met Haversine afstand
        db = DBSCAN(eps=eps_rad, min_samples=2, metric="haversine")
        cluster_labels = db.fit_predict(coords_rad)
    
        # Voeg cluster-informatie toe aan de seizoensdata
        df_season_result = df_season.copy()
        df_season_result["cluster_id"] = cluster_labels
    
        # Filter de ruis (-1 betekent dat het punt geen buur heeft binnen 100m)
        df_clusters = df_season_result[df_season_result["cluster_id"] != -1].copy()
    
        # Maak een unieke cluster-naam per seizoen (bijv. "2023-2024_Cluster_0")
        if not df_clusters.empty:
            df_clusters["unique_cluster_name"] = (
                df_clusters["season"]
                + "_Cluster_"
                + df_clusters["cluster_id"].astype(str)
            )
            clustered_records.append(df_clusters)
    
    # 5. Combineer alle clusters in één DataFrame
    if clustered_records:
        df_final_clusters = pd.concat(clustered_records)
    
        # Bereken de clustergrootte (aantal meldingen per cluster)
        cluster_counts = (
            df_final_clusters.groupby("unique_cluster_name")
            .size()
            .reset_index(name="cluster_grootte")
        )
        df_final_clusters = df_final_clusters.merge(
            cluster_counts, on="unique_cluster_name"
        )
    
        # Sorteer resultaat voor betere leesbaarheid
        df_final_clusters = df_final_clusters.sort_values(
            ["season", "unique_cluster_name"]
        )
    
        # Toon statistieken per seizoen
        print("Aantal clusters (>1 melding) per seizoen:")
        print(df_final_clusters.groupby("season")["unique_cluster_name"].nunique())
    
        print("\nGemiddelde clustergrootte per seizoen:")
        print(df_final_clusters.groupby("season")["cluster_grootte"].mean())
    else:
        df_final_clusters = pd.DataFrame()
        print("Geen clusters gevonden binnen de gestelde criteria.")

    return df_final_clusters 




def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent  # repo root (this script now lives in python/)
    df = gisservices_inbo()
    (
        observations_by_result,
        observations_by_month,
        observations_by_province,
        observations_by_province_till_current_date,
        observations_by_province_per_km2,
    ) = create_tables(df)
    observations_by_month.to_csv(output_dir / "observations_by_month.csv")
    observations_by_province.to_csv(output_dir / "observations_by_province.csv")
    observations_by_province_till_current_date.to_csv(
        output_dir / "observations_by_province_till_current_date.csv"
    )
    observations_by_province_per_km2.to_csv(
        output_dir / "observations_by_province_per_km2.csv"
    )
    print("Observations by result:")
    print(observations_by_result.to_string())
    print("Observations by month:")
    print(observations_by_month.to_string())
    print("\nObservations by province:")
    print(observations_by_province.to_string())
    print("\nObservations by province till current date:")
    print(observations_by_province_till_current_date.to_string())
    print("\nObservations by province per km2 till current date:")
    print(observations_by_province_per_km2.to_string())


if __name__ == "__main__":
    main()
