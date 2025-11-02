#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFiLlama Bridges
=====================================

Zweck
-----
Lädt öffentliche Bridge-Daten von DeFiLlama, erzeugt tägliche Zeitreihen pro Bridge
und fasst diese zu Monats- und Jahreswerten zusammen. Zusätzlich werden die
Top-N-Brücken je Monat und je Jahr (gemessen an `totalUSD`) als separate CSVs
exportiert.

Wesentliche Eigenschaften
-------------------------
- Zeitraumfilter per CLI (Default: 2022-10 .. 2025-08).
- Top-N frei wählbar (Default: 5).

- Ausgaben:
  - all_bridges_daily.csv
  - bridges_monthly.csv
  - bridges_yearly.csv
  - bridges_monthly_top_<N>.csv
  - bridges_yearly_top_<N>.csv

Beispiel
--------
$ python defillama_bridges_top_n.py --top-n 35 --start-month 2022-10 --end-month 2025-09

Voraussetzungen
---------------
Python 3.9+
pip install requests pandas
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BRIDGES_URL = "https://bridges.llama.fi/bridges"
BRIDGE_VOLUME_URL = "https://bridges.llama.fi/bridgevolume/all"

DEFAULT_TOP_N = 5
DEFAULT_START_MONTH = "2022-10"
DEFAULT_END_MONTH = "2025-08"

def nz(v: Any, fb: Any = 0) -> Any:
    if v is None:
        return fb
    if isinstance(v, str) and v.strip() == "":
        return fb
    return v

def to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0.0
        # naive Normalisierungsversuche
        candidates = (
            s.replace(" ", ""),
            s.replace(".", "").replace(",", "."),  # 1.234,56 -> 1234.56
            s.replace(",", ""),                    # 1,234.56 -> 1234.56
        )
        for cand in candidates:
            try:
                return float(cand)
            except Exception:
                pass
    return 0.0

def http_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    initial_delay: float = 0.4,
) -> Any:
    """GET mit Retries. Gibt JSON zurück oder wirft Exception."""
    delay = initial_delay
    last_exc: Optional[Exception] = None
    headers = {"User-Agent": "defillama-bridges-topn/1.0 (+no-tracking)"}
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=90, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise RuntimeError(f"GET failed for {url} params={params}: {last_exc}")

def fetch_bridges_overview() -> List[Dict[str, Any]]:
    """Liest /bridges Übersicht (Bezeichner, Anzeigenamen, …)."""
    data = http_get_json(BRIDGES_URL)
    if not isinstance(data, dict) or "bridges" not in data:
        raise RuntimeError(f"Unerwartetes Format von {BRIDGES_URL}: {str(data)[:200]}")
    return data["bridges"]

def fetch_bridge_daily_all(bridge_id: int) -> List[Dict[str, Any]]:
    """Liest tägliche Historie einer Bridge.

    Rückgabe: Liste von Objekten mit Unix-`date`, `depositUSD`, `withdrawUSD`, etc.
    """
    data = http_get_json(BRIDGE_VOLUME_URL, params={"id": bridge_id})
    if not isinstance(data, list):
        raise RuntimeError(f"Unerwartetes Format für bridge id={bridge_id}: {str(data)[:200]}")
    if not data or "date" not in data[0]:
        raise RuntimeError(f"Keine 'date'-Felder für bridge id={bridge_id}: {str(data[:2])}")
    return data

def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Schreibt eine Liste von Dicts als CSV mit fester Spaltenreihenfolge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def collect_daily_rows(ids_all: List[int], name_map: Dict[int, str], delay_ms: int) -> List[Dict[str, Any]]:
    """Holt die tägliche Historie für alle Bridges in `ids_all`.

    Args:
        ids_all: Liste der Bridge-IDs.
        name_map: Map id -> Anzeigename.
        delay_ms: Pause zwischen API-Calls.

    Returns:
        Liste mit Tageszeilen (bridgeId, bridgeName, date, depositUSD, withdrawUSD, totalUSD).
    """
    rows: List[Dict[str, Any]] = []

    for i, bid in enumerate(ids_all, start=1):
        print(f"[{i}/{len(ids_all)}] Hole Historie: bridgeId={bid}")
        try:
            hist = fetch_bridge_daily_all(bid)
            for r in hist:
                try:
                    ts = int(nz(r.get("date"), 0))
                except Exception:
                    ts = 0
                date_iso = dt.datetime.utcfromtimestamp(ts).date().isoformat() if ts > 0 else "1970-01-01"

                dep = to_float(nz(r.get("depositUSD"), 0))
                wd = to_float(nz(r.get("withdrawUSD"), 0))

                rows.append({
                    "bridgeId": bid,
                    "bridgeName": name_map.get(bid, f"bridge-{bid}"),
                    "date": date_iso,
                    "depositUSD": dep,
                    "withdrawUSD": wd,
                    "totalUSD": dep + wd,
                    "netUSD": dep - wd,
                })
        except Exception as e:
            print(f"  Warnung: Übersprungen: ID {bid} -> {e}", file=sys.stderr)

        if delay_ms and delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    return rows


def aggregate_monthly_yearly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erstellt Monats- und Jahres-Zusammenfassung aus Tagesdaten."""
    monthly = (df.groupby(["bridgeId", "bridgeName", "month"], as_index=False)
                 .agg(depositUSD=("depositUSD", "sum"),
                      withdrawUSD=("withdrawUSD", "sum"),
                      totalUSD=("totalUSD", "sum"),
                      netUSD=("netUSD", "sum")))
    monthly = monthly[["month", "bridgeId", "bridgeName", "depositUSD", "withdrawUSD", "totalUSD", "netUSD"]]

    yearly = (df.groupby(["bridgeId", "bridgeName", "year"], as_index=False)
                .agg(depositUSD=("depositUSD", "sum"),
                     withdrawUSD=("withdrawUSD", "sum"),
                     totalUSD=("totalUSD", "sum"),
                     netUSD=("netUSD", "sum")))
    yearly = yearly[["year", "bridgeId", "bridgeName", "depositUSD", "withdrawUSD", "totalUSD", "netUSD"]]
    yearly = yearly.sort_values(["year", "totalUSD"], ascending=[True, False])

    return monthly, yearly


def top_n_per_period(
    df: pd.DataFrame,
    period_col: str,
    top_n: int
) -> pd.DataFrame:
    """Erzeugt Top-N je Zeitraum (Monat oder Jahr) nach totalUSD."""
    df_sorted = df.sort_values([period_col, "totalUSD"], ascending=[True, False])
    return (
        df_sorted
        .groupby(period_col, as_index=False, sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ----------------------------- CLI & main ----------------------------------

def parse_args() -> argparse.Namespace:
    """Parst die Kommandozeilenargumente."""
    ap = argparse.ArgumentParser(
        description="DeFiLlama Bridges ETL – lade Zeitreihen und aggregiere Top-N je Monat/Jahr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--max-bridges", type=int, default=0,
                    help="0 = alle (ansonsten nur die ersten N IDs).")
    ap.add_argument("--delay-ms", type=int, default=200,
                    help="Pause zwischen API-Calls in Millisekunden.")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Ausgabeordner für CSVs.")
    ap.add_argument("--start-month", type=str, default=DEFAULT_START_MONTH,
                    help="Filter Start (yyyy-MM).")
    ap.add_argument("--end-month", type=str, default=DEFAULT_END_MONTH,
                    help="Filter Ende (yyyy-MM, inkl.).")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="Top-N Bridges pro Monat/Jahr.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    top_n = int(args.top_n)

    # /bridges Übersicht
    print("Hole /bridges Übersicht ...", flush=True)
    overview = fetch_bridges_overview()
    if not overview:
        print("Konnte /bridges nicht laden.", file=sys.stderr)
        sys.exit(2)

    # Name-Map id -> Anzeigename
    name_map: Dict[int, str] = {}
    ids_all: List[int] = []
    for b in overview:
        bid = int(b.get("id"))
        ids_all.append(bid)
        disp = b.get("displayName") or b.get("name") or b.get("slug") or f"bridge-{bid}"
        name_map[bid] = disp

    if args.max_bridges and args.max_bridges > 0:
        print(f"Begrenze auf erste {args.max_bridges} Bridges (nach Auflistungsreihenfolge).")
        ids_all = ids_all[: args.max_bridges]

    # Tagesreihen laden
    rows = collect_daily_rows(ids_all=ids_all, name_map=name_map, delay_ms=args.delay_ms)

    # CSV-Datei speichern
    daily_csv = outdir / "all_bridges_daily.csv"
    save_csv(daily_csv, rows, ["bridgeId", "bridgeName", "date", "depositUSD", "withdrawUSD", "totalUSD", "netUSD"])
    print(f"Saved -> {daily_csv}")

    if not rows:
        print("Keine Zeitreihen geladen – Abbruch.", file=sys.stderr)
        sys.exit(1)

    # DataFrame + Filter (Zeitraum)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%Y-%m")

    if args.start_month:
        df = df[df["month"] >= args.start_month]
    if args.end_month:
        df = df[df["month"] <= args.end_month]

    for col in ["depositUSD", "withdrawUSD", "totalUSD", "netUSD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Monats- & Jahresdaten
    monthly, yearly = aggregate_monthly_yearly(df)
    monthly_csv = outdir / "bridges_monthly.csv"
    yearly_csv = outdir / "bridges_yearly.csv"
    monthly.to_csv(monthly_csv, index=False, encoding="utf-8")
    yearly.to_csv(yearly_csv, index=False, encoding="utf-8")
    print(f"Saved -> {monthly_csv}")
    print(f"Saved -> {yearly_csv}")

    # Top-N je Monat
    monthly_top_n = (
        monthly.sort_values(["month", "totalUSD"], ascending=[True, False])
        .groupby("month", as_index=False, sort=False)
        .head(top_n)
    )

    month_totals = (
        monthly.groupby("month", as_index=False)["totalUSD"].sum()
        .rename(columns={"totalUSD": "month_totalUSD"})
    )

    monthly_top_n = monthly_top_n.merge(month_totals, on="month", how="left")
    monthly_top_n["shares"] = (
            (monthly_top_n["totalUSD"] / monthly_top_n["month_totalUSD"]) * 100.0
    ).fillna(0.0).round(2)
    monthly_top_n = monthly_top_n.drop(columns=["month_totalUSD"])

    monthly_top_csv = outdir / f"bridges_monthly_top_{top_n}.csv"
    monthly_top_n[["month", "bridgeId", "bridgeName", "totalUSD", "netUSD", "shares"]] \
        .to_csv(monthly_top_csv, index=False, encoding="utf-8")
    print(f"Saved -> {monthly_top_csv}")

    # Top-N je Jahr
    yearly_top_n = (
        yearly.sort_values(["year", "totalUSD"], ascending=[True, False])
        .groupby("year", as_index=False, sort=False)
        .head(top_n)
    )

    year_totals = (
        yearly.groupby("year", as_index=False)["totalUSD"].sum()
        .rename(columns={"totalUSD": "year_totalUSD"})
    )

    yearly_top_n = yearly_top_n.merge(year_totals, on="year", how="left")
    yearly_top_n["shares"] = (
            (yearly_top_n["totalUSD"] / yearly_top_n["year_totalUSD"]) * 100.0
    ).fillna(0.0).round(2)
    yearly_top_n = yearly_top_n.drop(columns=["year_totalUSD"])

    yearly_top_csv = outdir / f"bridges_yearly_top_{top_n}.csv"
    yearly_top_n[["year", "bridgeId", "bridgeName", "totalUSD", "netUSD", "shares"]] \
        .to_csv(yearly_top_csv, index=False, encoding="utf-8")
    print(f"Saved -> {yearly_top_csv}")

    # Status in Konsole
    active_months = monthly[monthly["totalUSD"] > 0].sort_values("month")["month"].tolist()
    if active_months:
        last_active_month = active_months[-1]
        print(f"\nLetzter aktiver Monat: {last_active_month}")
        print(
            monthly_top_n[monthly_top_n["month"] == last_active_month]
            .sort_values("totalUSD", ascending=False)
            .to_string(index=False)
        )

    if not yearly_top_n.empty:
        last_year = int(yearly_top_n["year"].max())
        print(f"\nTop-{top_n} im letzten Jahr: {last_year}")
        print(
            yearly_top_n[yearly_top_n["year"] == last_year]
            .sort_values("totalUSD", ascending=False)
            .to_string(index=False)
        )

    print(f"\nFertig. Dateien im Ordner: {outdir.resolve()}")


if __name__ == "__main__":
    main()
