#!/usr/bin/env python3
"""
Fetch player prop odds from Sportcast/BetOnline as fast as possible.

Key improvements for speed:
  * Concurrent Requests: uses a configurable thread pool (default 16) to fan out
    calls to RequestBetPriceUI.
  * Session Re-use per Worker: each worker re-uses a requests.Session with the
    correct headers to avoid TLS setup costs.
  * Streaming Emission: responses are written as they arrive; no need to wait
    for all futures to complete before persisting results.

WARNING: The number of combinations can explode (O(n^k)). Use combo-size with
care. The default configuration is tuned for aggressive throughput, but the API
may still rate-limit or throttle if too many requests are fired.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests


BASE_URL = "https://public-prod-gen2.sportcastlive.com"
MARKETS_ENDPOINT = "/public/getmarketsV2/"
PRICE_ENDPOINT = "/public/RequestBetPriceUI"

# Default concurrency tuned to keep total runtime low while avoiding request
# failures. Users can override via CLI.
DEFAULT_MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)


@dataclass
class SelectionNode:
    Id: int
    Selection: str
    EntityId: int = 0
    GlobalIdLong: Optional[str] = None
    GlobalIdShort: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict) -> "SelectionNode":
        return cls(
            Id=payload["Id"],
            Selection=payload.get("UntranslatedValue") or payload.get("Value"),
            EntityId=payload.get("EntityId") or 0,
            GlobalIdLong=payload.get("GlobalIdLong"),
            GlobalIdShort=payload.get("GlobalIdShort"),
        )


@dataclass
class MarketDetail:
    MarketId: int
    MarketName: str
    MarketLabelId: int
    AllowOrCombo: bool = False
    MultipleSelection: bool = False
    BetSelections: List[SelectionNode] = field(default_factory=list)

    def as_payload(self) -> Dict:
        payload = {
            "MarketId": self.MarketId,
            "MarketName": self.MarketName,
            "MarketLabelId": self.MarketLabelId,
            "AllowOrCombo": self.AllowOrCombo,
            "MultipleSelection": self.MultipleSelection,
            "BetSelections": [asdict(node) for node in self.BetSelections],
        }
        return payload


class SportcastClient:
    def __init__(self, api_key: str, fixture_id: int, culture: str, sport_id: int, client_id: Optional[int]):
        self.api_key = api_key
        self.fixture_id = fixture_id
        self.culture = culture
        self.sport_id = sport_id
        self.client_id = client_id

        self.base_headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/markets?key={api_key}&fixtureId={fixture_id}&odds=AmericanPrice&brand=betonline",
            "User-Agent": "Mozilla/5.0 (compatible; sportcast-scraper/2.0)",
            "Sc-FixtureId": str(fixture_id),
            "Sc-SportId": self._sport_name_from_id(sport_id),
        }

        self.session = self._new_session()

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.base_headers)
        return session

    @staticmethod
    def _sport_name_from_id(sport_id: int) -> str:
        mapping = {
            0: "Unknown",
            1: "Soccer",
            2: "AussieRules",
            3: "RugbyLeague",
            4: "Tennis",
            5: "Basketball",
            6: "Golf",
            7: "Soccer",
            8: "NFL",
            9: "Baseball",
            10: "IceHockey",
            11: "Cricket",
            14: "RugbyUnion",
            18: "MixedMartialArts",
        }
        return mapping.get(sport_id, "Unknown")

    def fetch_player_markets(self) -> List[Dict]:
        params = {
            "key": self.api_key,
            "fixtureId": self.fixture_id,
            "culture": self.culture,
            "returnFilters": "false",
            "marketLabel": "0",
        }
        resp = self.session.get(BASE_URL + MARKETS_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        return [market for market in payload["PayLoad"] if market.get("IsPlayerMarket")]

    def fetch_market_tree(self, label_id: int) -> Dict:
        params = {
            "key": self.api_key,
            "fixtureId": self.fixture_id,
            "culture": self.culture,
            "returnFilters": "true",
            "marketLabel": str(label_id),
        }
        resp = self.session.get(BASE_URL + MARKETS_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()["PayLoad"][0]

    def request_price(self, market_details: Sequence[MarketDetail], session: Optional[requests.Session] = None) -> Dict:
        payload = {
            "FixtureId": self.fixture_id,
            "Key": self.api_key,
            "Sport": self.sport_id,
            "Culture": self.culture,
            "ReturnBetSlip": False,
            "ReturnValidationMatrix": False,
            "ReturnAllTranslations": False,
            "ReturnMarkets": False,
            "MarketDetails": [md.as_payload() for md in market_details],
        }
        if self.client_id is not None:
            payload["ClientId"] = self.client_id
        active_session = session or self.session
        resp = active_session.post(BASE_URL + PRICE_ENDPOINT, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()


def extract_selection_paths(market_payload: Dict) -> Iterator[List[SelectionNode]]:
    def _traverse(node: Dict, path: List[SelectionNode]) -> Iterator[List[SelectionNode]]:
        current = SelectionNode.from_payload(node)
        path = path + [current]
        items = node.get("Items")
        if not items or not items.get("Items"):
            yield path
            return
        for child in items["Items"]:
            yield from _traverse(child, path)

    for player in market_payload.get("Filter", {}).get("Items", []):
        yield from _traverse(player, [])


def build_market_details(market_payload: Dict) -> List[MarketDetail]:
    base_kwargs = {
        "MarketId": market_payload["Id"],
        "MarketName": market_payload["UntranslatedLabel"],
        "MarketLabelId": market_payload["LabelId"],
        "AllowOrCombo": market_payload.get("AllowOrCombos", False),
        "MultipleSelection": market_payload.get("MultipleSelection", False),
    }

    details = []
    for path in extract_selection_paths(market_payload):
        md = MarketDetail(**base_kwargs)
        md.BetSelections = path
        details.append(md)
    return details


def combos_iterator(details: List[MarketDetail], combo_size: int) -> Iterable[Tuple[MarketDetail, ...]]:
    if combo_size <= 1:
        for item in details:
            yield (item,)
    else:
        yield from itertools.combinations(details, combo_size)


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe.strip("_") or "unknown"


def write_player_breakdown(result: Dict, store: Dict[str, Dict[str, List[Dict]]]) -> None:
    response = result.get("response", {})
    payload = response.get("PayLoad") or {}
    for entry in result.get("combo", []):
        player = entry["Selections"][0]
        market = entry["MarketName"]
        store.setdefault(player, {}).setdefault(market, []).append(
            {
                "combo": result["combo"],
                "price": payload.get("Price"),
                "priceDetails": payload.get("PriceDetails"),
                "status": payload.get("Status"),
                "rawResponse": response,
            }
        )


def persist_player_breakdown(store: Dict[str, Dict[str, List[Dict]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for player, markets in store.items():
        player_dir = output_dir / sanitize_name(player)
        player_dir.mkdir(parents=True, exist_ok=True)
        for market_name, combos in markets.items():
            path = player_dir / f"{sanitize_name(market_name)}.json"
            path.write_text(json.dumps(combos, indent=2, ensure_ascii=False))


def run(
    api_key: str,
    fixture_id: int,
    culture: str,
    sport_id: int,
    client_id: Optional[int],
    combo_size: int,
    sleep_seconds: float,
    output_path: Optional[Path],
    output_dir: Optional[Path],
    max_workers: int,
    quiet: bool,
) -> None:
    start = time.perf_counter()
    client = SportcastClient(api_key, fixture_id, culture, sport_id, client_id)

    player_markets = client.fetch_player_markets()
    all_market_details: List[MarketDetail] = []
    for market in player_markets:
        market_tree = client.fetch_market_tree(market["LabelId"])
        all_market_details.extend(build_market_details(market_tree))

    combo_iter = combos_iterator(all_market_details, combo_size)
    total_jobs = 0
    lock = threading.Lock()
    aggregated_results: List[Dict] = []
    player_store: Dict[str, Dict[str, List[Dict]]] = {}

    def process_combo(combo: Tuple[MarketDetail, ...]) -> Dict:
        local_session = client._new_session() if max_workers > 1 else client.session
        try:
            resp = client.request_price(combo, session=local_session)
        except requests.HTTPError as exc:
            resp = {"Status": -1, "Error": str(exc)}
        finally:
            if local_session is not client.session:
                local_session.close()
        selection_descriptions = [
            {
                "MarketName": md.MarketName,
                "Selections": [node.Selection for node in md.BetSelections],
            }
            for md in combo
        ]
        return {"combo": selection_descriptions, "response": resp}

    if max_workers <= 1:
        for combo in combo_iter:
            total_jobs += 1
            result = process_combo(combo)
            aggregated_results.append(result)
            write_player_breakdown(result, player_store)
            if not quiet:
                print(json.dumps(result, ensure_ascii=False))
            if sleep_seconds:
                time.sleep(sleep_seconds)
    else:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs lazily to avoid blowing up memory.
            future_to_combo = {}
            chunk = max_workers * 4
            for combo in combo_iter:
                future = executor.submit(process_combo, combo)
                future_to_combo[future] = combo
                total_jobs += 1

                if len(future_to_combo) >= chunk:
                    for done in futures.as_completed(list(future_to_combo.keys())):
                        combo = future_to_combo.pop(done)
                        try:
                            result = done.result()
                        except Exception as exc:  # noqa: BLE001 - best effort logging
                            result = {
                                "combo": [
                                    {
                                        "MarketName": md.MarketName,
                                        "Selections": [node.Selection for node in md.BetSelections],
                                    }
                                    for md in combo
                                ],
                                "response": {"Status": -1, "Error": str(exc)},
                            }
                        aggregated_results.append(result)
                        write_player_breakdown(result, player_store)
                        if not quiet:
                            with lock:
                                print(json.dumps(result, ensure_ascii=False))
                        if sleep_seconds:
                            time.sleep(sleep_seconds)

            # Drain remaining futures.
            for done in futures.as_completed(future_to_combo):
                combo = future_to_combo[done]
                try:
                    result = done.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "combo": [
                            {
                                "MarketName": md.MarketName,
                                "Selections": [node.Selection for node in md.BetSelections],
                            }
                            for md in combo
                        ],
                        "response": {"Status": -1, "Error": str(exc)},
                    }
                aggregated_results.append(result)
                write_player_breakdown(result, player_store)
                if not quiet:
                    with lock:
                        print(json.dumps(result, ensure_ascii=False))
                if sleep_seconds:
                    time.sleep(sleep_seconds)

    if output_path:
        output_path.write_text(json.dumps(aggregated_results, ensure_ascii=False, indent=2))
    if output_dir:
        persist_player_breakdown(player_store, output_dir)

    elapsed = time.perf_counter() - start
    if not quiet:
        print(f"\nCompleted {total_jobs} price requests in {elapsed:.2f}s (workers={max_workers})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blazing-fast player prop price fetcher.")
    parser.add_argument("--api-key", required=True, help="Sportcast API key.")
    parser.add_argument("--fixture-id", required=True, type=int, help="Fixture identifier.")
    parser.add_argument("--sport-id", type=int, default=5, help="Numeric sport id (default: 5).")
    parser.add_argument("--culture", default="en-US", help="Culture string (default: en-US).")
    parser.add_argument("--client-id", type=int, help="Optional client id to include in the payload.")
    parser.add_argument("--combo-size", type=int, default=1, help="Selections per request (>=1).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between processed results.")
    parser.add_argument("--output", type=Path, help="Write aggregated JSON to this file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="If set, write per-player JSON files into this directory (organised by folders).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Thread pool size (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-result stdout (useful when writing to output files only).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(
        api_key=args.api_key,
        fixture_id=args.fixture_id,
        culture=args.culture,
        sport_id=args.sport_id,
        client_id=args.client_id,
        combo_size=args.combo_size,
        sleep_seconds=args.sleep,
        output_path=args.output,
        output_dir=args.output_dir,
        max_workers=max(1, args.max_workers),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

