from uniswap_fetcher_rs import UniswapFetcher
from db.db_manager import DBManager
import os
from collections import defaultdict
from datetime import datetime, timezone
import time
import pandas as pd
from typing import Dict, List, Union
from utils.utils import (
    signed_hex_to_int,
    unsigned_hex_to_int,
    tick_to_sqrt_price,
    calc_prices_token0_by_token1,
    calc_prices_token1_by_token0,
    normalize_with_deciamls,
    apply_abs_to_list,
)

TIME_INTERVAL = 10 * 60
START_TIMESTAMP = int(datetime(2021, 5, 4).replace(tzinfo=timezone.utc).timestamp())
DAY_SECONDS = 86400
STABLECOINS = [
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI in Ethereum Mainnet
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC in Ethereum Mainnet
    "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC2
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC3
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT in Ethereum Mainnet
]


class PoolDataFetcher:
    def __init__(self) -> None:
        self.db_manager = DBManager()
        self.uniswap_fetcher = UniswapFetcher(os.getenv("ETHEREUM_RPC_NODE_URL"))

    def add_new_time_range(self) -> None:
        """
        Add a new timetable entry to the database.
        """
        last_time_range = self.db_manager.fetch_last_time_range()
        if last_time_range is None:
            start = START_TIMESTAMP
            end = START_TIMESTAMP + DAY_SECONDS
        else:
            start = last_time_range["end"]
            end = last_time_range["end"] + DAY_SECONDS
        print(f"Adding new timetable entry between {start} and {end}")
        self.db_manager.add_timetable_entry(start, end)

        print(f"Fetching token pairs between {start} and {end}")

        token_pairs = (
            self.uniswap_fetcher.get_pool_created_events_between_two_timestamps(
                int(start), int(end)
            )
        )
        self.db_manager.reset_token_pairs()
        self.db_manager.add_token_pairs(token_pairs)

        return {"start": start, "end": end}

    def get_incomplete_token_pairs(self, start: int, end: int) -> list[dict[str, str]]:
        """
        Get the token pairs for the miner modules.

        Args:
            start: The start datetime.
            end: The end datetime.

        Returns:
            The token pairs to fetch from rpc node.
        """
        token_pairs = self.db_manager.fetch_incompleted_token_pairs()
        if not token_pairs:
            self.db_manager.mark_time_range_as_complete(start, end)
            return None
        return token_pairs[:1]

    def save_pool_and_metrics_data(
        self, prob: dict, answer: dict, metrics: tuple
    ) -> None:
        """
        Save the pool data to the database.

        Args:
            prob: The token_pair and datetime to fetch.
            answer: The fetched data from rpc node.
        """
        token_pairs = prob.get("token_pairs", None)
        onchain_data = answer.get("data", None)
        pool_metrics, token_metrics, daily_pool_metrics, current_pool_metrics = metrics
        print("saving pool data to database ...")
        self.db_manager.add_or_update_current_pool_metrics(daily_pool_metrics)
        start_time = datetime.now()
        self.db_manager.add_pool_event_and_metrics_data(onchain_data, pool_metrics)
        end_time = datetime.now()
        print("Time taken to save pool data: ", end_time - start_time)
        self.db_manager.add_or_update_token_metrics(token_metrics)
        end_time = datetime.now()
        print("Time taken to save token metrics: ", end_time - start_time)
        self.db_manager.mark_token_pairs_as_complete(token_pairs)
        end_time = datetime.now()
        print("Time taken to mark token pairs as complete: ", end_time - start_time)
        self.db_manager.add_or_update_current_token_metrics(current_pool_metrics)
        print("pool data saved successfully.")

    def get_next_token_pairs(self, time_range: dict) -> dict:
        """
        Get a token pair to fetch.

        Returns:
            Token pair and time range.
        """
        token_pairs = self.get_incomplete_token_pairs(
            time_range["start"], time_range["end"]
        )
        if not token_pairs:
            return None

        req_token_pairs = []
        req_pool_addresses = []
        req_pools_map = {}
        for token_pair in token_pairs:
            req_token_pairs.append(
                (token_pair["token0"], token_pair["token1"], token_pair["fee"])
            )
            req_pool_addresses.append(token_pair["pool_address"])
            req_pools_map[token_pair["pool_address"]] = {
                "token0": token_pair["token0"],
                "token1": token_pair["token1"],
                "is_stablecoin": token_pair["is_stablecoin"],
                "token0_decimals": token_pair["token0_decimals"],
                "token1_decimals": token_pair["token1_decimals"],
            }

        return {
            "token_pairs": req_token_pairs,
            "pool_addresses": req_pool_addresses,
            "pools_map": req_pools_map,
            "start_datetime": int(time_range["start"]),
            "end_datetime": int(time_range["end"]),
        }

    def get_current_pool_metrics(
        self, pool_addresses: list[str]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get the current pool metrics."""
        return self.db_manager.fetch_current_pool_metrics(pool_addresses)

    def get_current_token_metrics(
        self, token_addresses: list[str]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get the current token metrics."""
        return self.db_manager.fetch_current_token_metrics(token_addresses)

    def get_token_metrics(
        self, token_address: str, start: int, end: int
    ) -> List[Dict[str, Union[int, float, str]]]:
        """Get the token metrics."""
        return self.db_manager.fetch_token_metrics(token_address, start, end)

    def process_time_range(self, time_range: dict):
        print(
            f'Processing time range between {time_range["start"]} and {time_range["end"]}'
        )
        prob = self.get_next_token_pairs(time_range)
        if prob is None:
            return None
        print(f"querying uniswap_fetcher with problem: {prob}")
        answer = self.uniswap_fetcher.fetch_pool_data(
            prob["token_pairs"], prob["start_datetime"], prob["end_datetime"]
        )
        print("received answer")
        current_pool_metrics = self.get_current_pool_metrics(prob["pool_addresses"])
        current_token_metrics = self.get_current_token_metrics(
            [
                prob["pools_map"][pool_address]["token0"]
                for pool_address in prob["pool_addresses"]
            ]
            + [
                prob["pools_map"][pool_address]["token1"]
                for pool_address in prob["pool_addresses"]
            ]
        )
        metrics = self.generate_metrics(
            answer,
            prob["pools_map"],
            prob["pool_addresses"],
            current_pool_metrics,
            current_token_metrics,
            time_range["start"],
            time_range["end"],
        )
        print("saving data...")
        self.save_pool_and_metrics_data(prob, answer, metrics)

    def generate_metrics(
        self,
        pool_data: Dict,
        pools_map: Dict[str, Dict[str, Union[str, bool, int]]],
        prob_pool_addresses: list,
        current_pool_metrics: Dict,
        current_token_metrics: Dict,
        start: int,
        end: int,
        interval: int = 300,
    ) -> tuple:
        """
        Generate metrics from the pool data.

        Args:
            pool_data: The pool data to generate metrics from on-chain pool data.
            start: The start datetime for aggregation.
            end: The end datetime for aggregation.
            interval: The interval for aggregation.
        """
        derived_token_metrics = {}
        for pool_map in pools_map.values():
            if not pool_map.get("is_stablecoin"):
                db_token_metrics = self.db_manager.fetch_token_metrics(
                    pool_map.get("token0"), start + interval, end
                )
                db_token_metrics.extend(
                    self.db_manager.fetch_token_metrics(
                        pool_map.get("token1"), start + interval, end
                    )
                )
                for token_metric in db_token_metrics:
                    if token_metric["token_address"] not in derived_token_metrics:
                        derived_token_metrics[token_metric["token_address"]] = {
                            token_metric["timestamp"]: token_metric.get(
                                "close_price", 0.0
                            )
                        }
                    else:
                        derived_token_metrics[token_metric["token_address"]].update(
                            {
                                token_metric["timestamp"]: token_metric.get(
                                    "close_price", 0.0
                                )
                            }
                        )
        # raise Exception("stop")
        data = pool_data.get("data", None)
        daily_pool_metrics = {}
        aggregated_data = defaultdict(lambda: defaultdict(list))
        datetime_series = pd.date_range(
            start=datetime.fromtimestamp(start + interval, tz=timezone.utc),
            end=datetime.fromtimestamp(end, tz=timezone.utc),
            freq=f"{interval}s",
        )
        for pool_address in prob_pool_addresses:
            daily_pool_metrics[pool_address] = {
                "events_count": 1,
                "last_price": current_pool_metrics.get(pool_address, {}).get(
                    "price", 0.0
                ),
                "total_liquidity_token0": current_pool_metrics.get(
                    pool_address, {}
                ).get("liquidity_token0", 0.0),
                "total_liquidity_token1": current_pool_metrics.get(
                    pool_address, {}
                ).get("liquidity_token1", 0.0),
                "total_volume_token0": current_pool_metrics.get(pool_address, {}).get(
                    "volume_token0", 0.0
                ),
                "total_volume_token1": current_pool_metrics.get(pool_address, {}).get(
                    "volume_token1", 0.0
                ),
            }
            for date_time in datetime_series:
                timestamp = int(date_time.timestamp())
                key = (pool_address, timestamp)
                aggregated_data[key]["total_liquidity"] = []
                aggregated_data[key]["token0_liquidity"] = []
                aggregated_data[key]["token1_liquidity"] = []
                aggregated_data[key]["amount0"] = []
                aggregated_data[key]["amount1"] = []
                aggregated_data[key]["sqrt_price_x96"] = []
                aggregated_data[key]["type"] = None
        # print(f'printing aggregated data: {aggregated_data}')
        # print(f"daily_pool_metrics: {daily_pool_metrics}")
        for event in data:
            round_timestamp = event.get("timestamp") // interval * interval + interval
            pool_address = event.get("pool_address")
            daily_pool_metrics[pool_address]["events_count"] += 1
            key = (pool_address, round_timestamp)
            if key not in aggregated_data:
                raise Exception(f"Key {key} not found in aggregated_data")
            if event.get("event").get("type") == "swap":
                aggregated_data[key]["sqrt_price_x96"].append(
                    event.get("event").get("data").get("sqrt_price_x96")
                )
                aggregated_data[key]["amount0"].append(
                    signed_hex_to_int(event.get("event").get("data").get("amount0"))
                )
                aggregated_data[key]["amount1"].append(
                    signed_hex_to_int(event.get("event").get("data").get("amount1"))
                )
            else:
                amount = unsigned_hex_to_int(
                    event.get("event").get("data").get("amount", "0x0")
                )
                tick_lower = event.get("event").get("data").get("tick_lower")
                tick_upper = event.get("event").get("data").get("tick_upper")
                sqrt_price_lower = tick_to_sqrt_price(tick_lower)
                sqrt_price_upper = tick_to_sqrt_price(tick_upper)
                liquidity_token0 = (
                    amount
                    * (sqrt_price_upper - sqrt_price_lower)
                    / (sqrt_price_upper * sqrt_price_lower)
                )
                liquidity_token1 = (
                    amount * (sqrt_price_upper - sqrt_price_lower) / sqrt_price_upper
                )
                event_type = event.get("event").get("type")
                if event_type == "burn":
                    liquidity_token0 = -liquidity_token0
                    liquidity_token1 = -liquidity_token1
                aggregated_data[key]["token0_liquidity"].append(liquidity_token0)
                aggregated_data[key]["token1_liquidity"].append(liquidity_token1)
                aggregated_data[key]["total_liquidity"].append(amount)

        pool_metrics = []
        token_metrics = []

        for key, value in aggregated_data.items():
            pool_address, timestamp = key
            volume_token0 = normalize_with_deciamls(
                sum(apply_abs_to_list(value["amount0"])),
                pools_map[pool_address].get("token0_decimals"),
            )
            volume_token1 = normalize_with_deciamls(
                sum(apply_abs_to_list(value["amount1"])),
                pools_map[pool_address].get("token1_decimals"),
            )
            liquidity_token0 = normalize_with_deciamls(
                sum(value["token0_liquidity"]),
                pools_map[pool_address].get("token0_decimals"),
            )
            liquidity_token1 = normalize_with_deciamls(
                sum(value["token1_liquidity"]),
                pools_map[pool_address].get("token1_decimals"),
            )
            token0_address = pools_map[pool_address].get("token0")
            token1_address = pools_map[pool_address].get("token1")
            prices_ratio_token0 = calc_prices_token0_by_token1(
                value["sqrt_price_x96"],
                pools_map[pool_address].get("token0_decimals"),
                pools_map[pool_address].get("token1_decimals"),
            )
            prices_ratio_token1 = calc_prices_token1_by_token0(
                value["sqrt_price_x96"],
                pools_map[pool_address].get("token0_decimals"),
                pools_map[pool_address].get("token1_decimals"),
            )
            if token0_address in STABLECOINS or token1_address in STABLECOINS:
                stable_token = (
                    token0_address if token0_address in STABLECOINS else token1_address
                )
                non_stable_token = (
                    token0_address
                    if token0_address not in STABLECOINS
                    else token1_address
                )

                stable_price = 1.0
                if len(prices_ratio_token0) == 0:
                    prices_ratio_token0 = [
                        0.0,
                    ]
                if len(prices_ratio_token1) == 0:
                    prices_ratio_token1 = [
                        0.0,
                    ]
                close_non_stable_price = (
                    prices_ratio_token1[-1]
                    if token0_address in STABLECOINS
                    else prices_ratio_token0[-1]
                )
                high_non_stable_price = (
                    max(prices_ratio_token1)
                    if token0_address in STABLECOINS
                    else max(prices_ratio_token0)
                )
                low_non_stable_price = (
                    min(prices_ratio_token1)
                    if token0_address in STABLECOINS
                    else min(prices_ratio_token0)
                )
                if token0_address == stable_token:
                    stable_token_volume = volume_token0
                    stable_token_liquidity = liquidity_token0
                    non_stable_token_volume = volume_token1
                    non_stable_token_liquidity = liquidity_token1
                else:
                    stable_token_volume = volume_token1
                    stable_token_liquidity = liquidity_token1
                    non_stable_token_volume = volume_token0
                    non_stable_token_liquidity = liquidity_token0

                if stable_token in current_token_metrics:
                    current_token_metrics[stable_token]["close_price"] = (
                        stable_price
                        if stable_price != 0.0
                        else current_token_metrics[stable_token]["close_price"]
                    )
                    current_token_metrics[stable_token]["total_volume"] += (
                        stable_token_volume
                    )
                    current_token_metrics[stable_token]["total_liquidity"] += (
                        stable_token_liquidity
                    )
                else:
                    current_token_metrics[stable_token] = {
                        "close_price": stable_price,
                        "total_volume": stable_token_volume,
                        "total_liquidity": stable_token_liquidity,
                    }
                if non_stable_token in current_token_metrics:
                    current_token_metrics[non_stable_token]["close_price"] = (
                        close_non_stable_price
                        if close_non_stable_price != 0.0
                        else current_token_metrics[non_stable_token]["close_price"]
                    )
                    current_token_metrics[non_stable_token]["total_volume"] += (
                        non_stable_token_volume
                    )
                    current_token_metrics[non_stable_token]["total_liquidity"] += (
                        non_stable_token_liquidity
                    )
                else:
                    current_token_metrics[non_stable_token] = {
                        "close_price": close_non_stable_price,
                        "total_volume": non_stable_token_volume,
                        "total_liquidity": non_stable_token_liquidity,
                    }

                token_metrics.append(
                    {
                        "token_address": stable_token,
                        "timestamp": timestamp,
                        "total_volume": current_token_metrics[stable_token][
                            "total_volume"
                        ],
                        "total_liquidity": current_token_metrics[stable_token][
                            "total_liquidity"
                        ],
                        "high_price": stable_price
                        if stable_price != 0.0
                        else current_token_metrics[stable_token]["close_price"],
                        "low_price": stable_price
                        if stable_price != 0.0
                        else current_token_metrics[stable_token]["close_price"],
                        "close_price": stable_price
                        if stable_price != 0.0
                        else current_token_metrics[stable_token]["close_price"],
                        "is_derived": False,
                    }
                )
                token_metrics.append(
                    {
                        "token_address": non_stable_token,
                        "timestamp": timestamp,
                        "total_volume": current_token_metrics[non_stable_token][
                            "total_volume"
                        ],
                        "total_liquidity": current_token_metrics[non_stable_token][
                            "total_liquidity"
                        ],
                        "high_price": high_non_stable_price
                        if high_non_stable_price != 0.0
                        else current_token_metrics[non_stable_token]["close_price"],
                        "low_price": low_non_stable_price
                        if low_non_stable_price != 0.0
                        else current_token_metrics[non_stable_token]["close_price"],
                        "close_price": close_non_stable_price
                        if close_non_stable_price != 0.0
                        else current_token_metrics[non_stable_token]["close_price"],
                        "is_derived": False,
                    }
                )

                if token0_address == stable_token:
                    daily_pool_metrics[pool_address]["last_price"] = (
                        close_non_stable_price
                        if close_non_stable_price != 0.0
                        else daily_pool_metrics[pool_address]["last_price"]
                    )
                    daily_pool_metrics[pool_address]["total_liquidity_token0"] += (
                        stable_token_liquidity
                    )
                    daily_pool_metrics[pool_address]["total_liquidity_token1"] += (
                        non_stable_token_liquidity
                    )
                    daily_pool_metrics[pool_address]["total_volume_token0"] += (
                        stable_token_volume
                    )
                    daily_pool_metrics[pool_address]["total_volume_token1"] += (
                        non_stable_token_volume
                    )
                    pool_metrics.append(
                        {
                            "pool_address": pool_address,
                            "timestamp": timestamp,
                            "price": close_non_stable_price
                            if close_non_stable_price != 0.0
                            else daily_pool_metrics[pool_address]["last_price"],
                            "volume_token0": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_volume_token0", 0.0),
                            "volume_token1": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_volume_token1", 0.0),
                            "liquidity_token0": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_liquidity_token0", 0.0),
                            "liquidity_token1": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_liquidity_token1", 0.0),
                        }
                    )

                else:
                    daily_pool_metrics[pool_address]["last_price"] = (
                        close_non_stable_price
                        if close_non_stable_price != 0.0
                        else daily_pool_metrics[pool_address]["last_price"]
                    )
                    daily_pool_metrics[pool_address]["total_liquidity_token0"] += (
                        non_stable_token_liquidity
                    )
                    daily_pool_metrics[pool_address]["total_liquidity_token1"] += (
                        stable_token_liquidity
                    )
                    daily_pool_metrics[pool_address]["total_volume_token0"] += (
                        non_stable_token_volume
                    )
                    daily_pool_metrics[pool_address]["total_volume_token1"] += (
                        stable_token_volume
                    )
                    pool_metrics.append(
                        {
                            "pool_address": pool_address,
                            "timestamp": timestamp,
                            "price": close_non_stable_price
                            if close_non_stable_price != 0.0
                            else daily_pool_metrics[pool_address]["last_price"],
                            "volume_token0": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_volume_token0", 0.0),
                            "volume_token1": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_volume_token1", 0.0),
                            "liquidity_token0": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_liquidity_token0", 0.0),
                            "liquidity_token1": daily_pool_metrics.get(
                                pool_address, {}
                            ).get("total_liquidity_token1", 0.0),
                        }
                    )
            else:
                price_token0 = derived_token_metrics.get(token0_address, {}).get(
                    timestamp, None
                )
                price_token1 = derived_token_metrics.get(token1_address, {}).get(
                    timestamp, None
                )
                if not price_token0 or not price_token1:
                    continue

                if len(prices_ratio_token0) == 0:
                    prices_ratio_token0 = [
                        0.0,
                    ]
                    prices_ratio_token1 = [
                        0.0,
                    ]
                is_token0_derived = True
                is_token1_derived = True
                close_price_token0 = 0.0
                close_price_token1 = 0.0
                if price_token0 and not price_token1:
                    close_price_token1 = price_token0 / prices_ratio_token0[-1]
                    high_price_token1 = price_token0 / min(prices_ratio_token0)
                    low_price_token1 = price_token0 / max(prices_ratio_token0)
                    is_token1_derived = False

                elif price_token1 and not price_token0:
                    close_price_token0 = price_token1 * prices_ratio_token0[-1]
                    high_price_token0 = price_token1 * max(prices_ratio_token0)
                    low_price_token0 = price_token1 * min(prices_ratio_token0)
                    is_token0_derived = False

                if token0_address in current_token_metrics:
                    current_token_metrics[token0_address]["close_price"] = (
                        close_price_token0
                        if close_price_token0 != 0.0
                        else current_token_metrics[token0_address]["close_price"]
                    )
                    current_token_metrics[token0_address]["total_volume"] += (
                        volume_token0
                    )
                    current_token_metrics[token0_address]["total_liquidity"] += (
                        liquidity_token0
                    )
                else:
                    current_token_metrics[token0_address] = {
                        "close_price": price_token0,
                        "total_volume": volume_token0,
                        "total_liquidity": liquidity_token0,
                    }

                if token1_address in current_token_metrics:
                    current_token_metrics[token1_address]["close_price"] = (
                        close_price_token1
                        if close_price_token1 != 0.0
                        else current_token_metrics[token1_address]["close_price"]
                    )
                    current_token_metrics[token1_address]["total_volume"] += (
                        volume_token1
                    )
                    current_token_metrics[token1_address]["total_liquidity"] += (
                        liquidity_token1
                    )
                else:
                    current_token_metrics[token1_address] = {
                        "close_price": price_token1,
                        "total_volume": volume_token1,
                        "total_liquidity": liquidity_token1,
                    }

                if is_token0_derived:
                    token_metrics.append(
                        {
                            "token_address": token0_address,
                            "timestamp": timestamp,
                            "total_volume": current_token_metrics[token0_address][
                                "total_volume"
                            ],
                            "total_liquidity": current_token_metrics[token0_address][
                                "total_liquidity"
                            ],
                            "is_derived": True,
                        }
                    )
                else:
                    token_metrics.append(
                        {
                            "token_address": token0_address,
                            "timestamp": timestamp,
                            "total_volume": current_token_metrics[token0_address][
                                "total_volume"
                            ],
                            "total_liquidity": current_token_metrics[token0_address][
                                "total_liquidity"
                            ],
                            "high_price": high_price_token0
                            if high_price_token0 != 0.0
                            else current_token_metrics[token0_address]["close_price"],
                            "low_price": low_price_token0
                            if low_price_token0 != 0.0
                            else current_token_metrics[token0_address]["close_price"],
                            "close_price": close_price_token0
                            if close_price_token0 != 0.0
                            else current_token_metrics[token0_address]["close_price"],
                            "is_derived": False,
                        }
                    )

                if is_token1_derived:
                    token_metrics.append(
                        {
                            "token_address": token1_address,
                            "timestamp": timestamp,
                            "total_volume": current_token_metrics[token1_address][
                                "total_volume"
                            ],
                            "total_liquidity": current_token_metrics[token1_address][
                                "total_liquidity"
                            ],
                            "is_derived": True,
                        }
                    )
                else:
                    token_metrics.append(
                        {
                            "token_address": token1_address,
                            "timestamp": timestamp,
                            "total_volume": current_token_metrics[token1_address][
                                "total_volume"
                            ],
                            "total_liquidity": current_token_metrics[token1_address][
                                "total_liquidity"
                            ],
                            "high_price": high_price_token1
                            if high_price_token1 != 0.0
                            else current_token_metrics[token1_address]["close_price"],
                            "low_price": low_price_token1
                            if low_price_token1 != 0.0
                            else current_token_metrics[token1_address]["close_price"],
                            "close_price": close_price_token1
                            if close_price_token1 != 0.0
                            else current_token_metrics[token1_address]["close_price"],
                            "is_derived": False,
                        }
                    )
                daily_pool_metrics[pool_address]["last_price"] = (
                    prices_ratio_token0[-1]
                    if prices_ratio_token0[-1] != 0.0
                    else daily_pool_metrics[pool_address]["last_price"]
                )
                daily_pool_metrics[pool_address]["total_liquidity_token0"] += (
                    liquidity_token0
                )
                daily_pool_metrics[pool_address]["total_volume_token0"] += volume_token0
                daily_pool_metrics[pool_address]["total_liquidity_token1"] += (
                    liquidity_token1
                )
                daily_pool_metrics[pool_address]["total_volume_token1"] += volume_token1
                pool_metrics.append(
                    {
                        "pool_address": pool_address,
                        "timestamp": timestamp,
                        "price": close_price_token0
                        if close_price_token0 != 0.0
                        else daily_pool_metrics[pool_address]["last_price"],
                        "volume_token0": daily_pool_metrics.get(pool_address, {}).get(
                            "total_volume_token0", 0.0
                        ),
                        "volume_token1": daily_pool_metrics.get(pool_address, {}).get(
                            "total_volume_token1", 0.0
                        ),
                        "liquidity_token0": daily_pool_metrics.get(
                            pool_address, {}
                        ).get("total_liquidity_token0", 0.0),
                        "liquidity_token1": daily_pool_metrics.get(
                            pool_address, {}
                        ).get("total_liquidity_token1", 0.0),
                    }
                )

        return pool_metrics, token_metrics, daily_pool_metrics, current_token_metrics

    def run(self):
        while True:
            time_range = self.db_manager.fetch_last_time_range()
            if time_range is None or time_range["completed"]:
                time_range = self.add_new_time_range()

            now = datetime.now().timestamp()
            if time_range["end"] <= now:
                self.process_time_range(time_range)
            else:
                adjusted_time_range = {"start": time_range["start"], "end": now}
                self.process_time_range(adjusted_time_range)

                prev = now
                while prev <= time_range["end"]:
                    time.sleep(TIME_INTERVAL)

                    now = datetime.now()
                    if now > time_range["end"]:
                        now = time_range["end"]

                    new_time_range = {"start": prev, "end": now}
                    self.process_time_range(new_time_range)

                    prev = now
            self.get_incomplete_token_pairs(time_range["start"], time_range["end"])


if __name__ == "__main__":
    pool_data_fetcher = PoolDataFetcher()
    pool_data_fetcher.run()
