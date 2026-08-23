"""VespaWatch analysis based on the INBO ArcGIS FeatureServer export."""
from pathlib import Path

import pandas as pd
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
        by_month,
        by_province,
        by_province_till_current_date,
        by_province_per_km2,
    )


def main() -> None:
    output_dir = Path(__file__).parent
    df = gisservices_inbo()
    (
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
