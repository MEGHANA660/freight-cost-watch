"""Weekly cost-per-tonne-km per route plus the two comparison baselines."""
import pandas as pd

from . import config

REQUIRED = ["origin", "destination", "route_type", "quantity_tonnes",
            "distance_km", "freight_cost_inr", "shipment_date"]


def load_shipments(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"shipment file missing columns: {missing}")
    df = df.dropna(subset=REQUIRED)
    df = df[(df.quantity_tonnes > 0) & (df.distance_km > 0) & (df.freight_cost_inr > 0)]
    df = df.copy()
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    # Weeks run Monday-Sunday; week_of is that week's Monday.
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_route_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    """cost_per_tonne_km = total cost / total (tonnes x km) for the route-week."""
    w = (shipments.groupby(["route", "route_type", "week_of"], as_index=False)
         .agg(cost_inr=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum"),
              shipments=("route", "size")))
    w["cost_per_tonne_km"] = w["cost_inr"] / w["tonne_km"]
    return w.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    key = ["route", "route_type"]

    # vs own history: mean of the prior (up to) 8 weekly values, current week excluded.
    prior = w.groupby(key)["cost_per_tonne_km"].shift(1)
    w["_prior"] = prior
    w["own_history_avg"] = (w.groupby(key)["_prior"]
                            .transform(lambda s: s.rolling(config.HISTORY_WEEKS, min_periods=1).mean()))
    w["history_weeks_used"] = (w.groupby(key)["_prior"]
                               .transform(lambda s: s.rolling(config.HISTORY_WEEKS, min_periods=1).count()))
    w["vs_own_pct"] = (w["cost_per_tonne_km"] / w["own_history_avg"] - 1) * 100

    # vs similar routes: mean of OTHER routes with same route_type in the same week.
    grp = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    total, count = grp.transform("sum"), grp.transform("count")
    w["peer_avg"] = ((total - w["cost_per_tonne_km"]) / (count - 1)).where(count > 1)  # NaN if no peers
    w["vs_peer_pct"] = (w["cost_per_tonne_km"] / w["peer_avg"] - 1) * 100
    return w.drop(columns="_prior")
