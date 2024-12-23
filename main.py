from uniswap_fetcher_rs import UniswapFetcher
from concurrent.futures import ThreadPoolExecutor
from db.db_manager import DBManager
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
import time
import pandas as pd
from typing import Dict, List, Union
from utils.utils import *
from utils.log import Logger

TIME_INTERVAL = 10 * 60
METRIC_INTERVAL = 5 * 60
START_TIMESTAMP = int(datetime(2021, 5, 4).replace(tzinfo=timezone.utc).timestamp())
DAY_SECONDS = 86400
ESP = 1e-9

class PoolDataFetcher:
    def __init__(self) -> None:
        self.db_manager = DBManager()
        self.uniswap_fetcher = UniswapFetcher(os.getenv("ETHEREUM_RPC_NODE_URL"))
        self.logger = Logger("PoolDataFetcher")

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
        self.logger.log_info(f"Adding new timetable entry between {start} and {end}")
        self.db_manager.add_timetable_entry(start, end)

        self.logger.log_info(f"Fetching token pairs between {start} and {end}")

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
        return token_pairs[:25]

    def save_pool_and_metrics_data(
        self, prob: dict, onchain_data: dict, metrics: tuple
    ) -> None:
        token_pairs = prob.get("token_pairs", None)
        onchain_data = onchain_data.get("data", None)
        pool_metrics = metrics
        self.db_manager.add_pool_event_and_metrics_data(onchain_data, pool_metrics)
        self.db_manager.mark_token_pairs_as_complete(token_pairs)

    def get_next_query(self, time_range: dict) -> dict:
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
                "has_stablecoin": token_pair["has_stablecoin"],
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
    
    
    def generate_metrics(
        self,
        onchain_data: dict,
        pool_addresses: list[str],
        pool_infos: dict,
        start_timestamp: int,
        end_timestamp: int,
        interval: int = 300,
    ) -> List[Dict]:
        """Generate metrics from the pool data."""
        pool_metrics = []
        pools_data = self.organize_pool_data(onchain_data.get("data", []))
        for pool_address in pool_addresses:
            pool_metrics.extend(
                self.generate_pool_metrics(
                    pools_data[pool_address], pool_address, pool_infos, start_timestamp, end_timestamp, interval
                )
            )
        return pool_metrics
    
    def organize_pool_data(self, pool_data: list[dict]) -> dict:
        """Organize pool data by pool address."""
        pools_data = defaultdict(list)
        for event in pool_data:
            pool_address = event.get("pool_address")
            pools_data[pool_address].append(event)
        return pools_data
    
    def generate_pool_metrics(self, pool_data: list, pool_address: str, pool_infos: dict, start_timestamp: int, end_timestamp: int, interval: int = 300) -> List[Dict]:
        """Generate pool metrics from pool data."""
        
        last_pool_metric = self.get_last_pool_metric(pool_address, start_timestamp)
        if last_pool_metric is None:
            last_pool_metric = {
                "price": 0.0,
                "liquidity_token0": 0.0,
                "liquidity_token1": 0.0,
                "volume_token0": 0.0,
                "volume_token1": 0.0,
            }
        pool_metrics = []
        aggregated_data = defaultdict(lambda: defaultdict(list))
        datetime_series = pd.date_range(
            start=datetime.fromtimestamp(start_timestamp + interval, tz=timezone.utc),
            end=datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
            freq=f"{interval}s",
        )
        for date_time in datetime_series:
            timestamp = int(date_time.timestamp())
            aggregated_data[timestamp]["total_liquidity"] = []
            aggregated_data[timestamp]["liquidity_token0"] = []
            aggregated_data[timestamp]["liquidity_token1"] = []
            aggregated_data[timestamp]["amount0"] = []
            aggregated_data[timestamp]["amount1"] = []
            aggregated_data[timestamp]["prices"] = []
        
        token0_decimals = pool_infos[pool_address].get("token0_decimals")
        token1_decimals = pool_infos[pool_address].get("token1_decimals")
        for event in pool_data:
            round_timestamp = event.get("timestamp") // interval * interval + interval
            if event.get("event").get("type") == "swap":
                amount0 = signed_hex_to_int(event.get("event").get("data").get("amount0"))
                amount1 = signed_hex_to_int(event.get("event").get("data").get("amount1"))
                liquidity = unsigned_hex_to_int(event.get("event").get("data").get("liquidity"))
                sqrt_price_x96 = event.get("event").get("data").get("sqrt_price_x96")
                price = calc_price_sqrt_base(sqrt_price_x96, token0_decimals, token1_decimals) if liquidity > 0 else 0
                # aggregated_data[round_timestamp]["prices"].append(calc_price(amount0, amount1, token0_decimals, token1_decimals))
                aggregated_data[round_timestamp]["prices"].append(price)
                aggregated_data[round_timestamp]["amount0"].append(abs(amount0))
                aggregated_data[round_timestamp]["amount1"].append(abs(amount1))
                # aggregated_data[round_timestamp]["liquidity_token0"].append(amount0)
                # aggregated_data[round_timestamp]["liquidity_token1"].append(amount1)
            else:
                amount0 = unsigned_hex_to_int(event.get("event").get("data").get("amount0", "0x0"))
                amount1 = unsigned_hex_to_int(event.get("event").get("data").get("amount1", "0x0"))
                event_type = event.get("event").get("type")
                if event_type == "mint":
                    aggregated_data[round_timestamp]["liquidity_token0"].append(amount0)
                    aggregated_data[round_timestamp]["liquidity_token1"].append(amount1)
                elif event_type == "burn":
                    aggregated_data[round_timestamp]["liquidity_token0"].append(-amount0)
                    aggregated_data[round_timestamp]["liquidity_token1"].append(-amount1)
        for timestamp, value in aggregated_data.items():
            volume_token0 = normalize_with_deciamls(
                sum(apply_abs_to_list(value["amount0"])), token0_decimals
            )
            volume_token1 = normalize_with_deciamls(
                sum(apply_abs_to_list(value["amount1"])), token1_decimals
            )
            liquidity_token0 = normalize_with_deciamls(
                sum(value["liquidity_token0"]), token0_decimals
            )
            liquidity_token1 = normalize_with_deciamls(
                sum(value["liquidity_token1"]), token1_decimals
            )
            
            prices_ratio_token0 = value["prices"]
            last_pool_metric.update(
                {
                    "price": prices_ratio_token0[-1] if len(prices_ratio_token0) > 0 else last_pool_metric["price"],
                    "liquidity_token0": liquidity_token0 + last_pool_metric["liquidity_token0"],
                    "liquidity_token1": liquidity_token1 + last_pool_metric["liquidity_token1"],
                    "volume_token0": volume_token0 + last_pool_metric["volume_token0"],
                    "volume_token1": volume_token1 + last_pool_metric["volume_token1"],
                }
            )
            pool_metrics.append(
                {
                    "pool_address": pool_address,
                    "timestamp": timestamp,
                    "price": last_pool_metric["price"],
                    "volume_token0": last_pool_metric["volume_token0"],
                    "volume_token1": last_pool_metric["volume_token1"],
                    "liquidity_token0": last_pool_metric["liquidity_token0"],
                    "liquidity_token1": last_pool_metric["liquidity_token1"],
                }
            )
        return pool_metrics
            
    def generate_token_metrics(self, pool_metrics: List[Dict], token0_address: str, token1_address: str, start_timestamp: int, end_timestamp: int, interval: int, visited: dict) -> List[Dict]:
        """Generate token metrics from pool metrics by one pool."""
        token_metrics = []
        last_token0_metric: dict = self.get_last_token_metric(token0_address, start_timestamp)
        last_token1_metric: dict = self.get_last_token_metric(token1_address, start_timestamp)
        if not last_token0_metric:
            last_token0_metric = {
                "total_volume": 0.0,
                "total_liquidity": 0.0
            }
        if not last_token1_metric:
            last_token1_metric = {
                "total_volume": 0.0,
                "total_liquidity": 0.0
            }
        
        for pool_metric in pool_metrics:
            timestamp = pool_metric["timestamp"]
            last_token0_metric.update(
                {
                    "total_volume": pool_metric["volume_token0"] + last_token0_metric["total_volume"],
                    "total_liquidity": pool_metric["liquidity_token0"] + last_token0_metric["total_liquidity"],
                }
            )
            last_token1_metric.update(
                {
                    "total_volume": pool_metric["volume_token1"] + last_token1_metric["total_volume"],
                    "total_liquidity": pool_metric["liquidity_token1"] + last_token1_metric["total_liquidity"],
                }
            )
            token_metrics.append(
                {
                    "token_address": token0_address,
                    "timestamp": timestamp,
                    "total_volume": last_token0_metric["total_volume"],
                    "total_liquidity": last_token0_metric["total_liquidity"],
                }
            )
            token_metrics.append(
                {
                    "token_address": token1_address,
                    "timestamp": timestamp,
                    "total_volume": last_token1_metric["total_volume"],
                    "total_liquidity": last_token1_metric["total_liquidity"],
                }
            )
        return token_metrics
    
    def get_last_token_metric(
        self, token_address: str, timestamp: int
    ) -> List[Dict[str, Union[int, float, str]]]:
        """Get the token metrics."""
        return self.db_manager.fetch_last_token_metric(token_address, timestamp)
    
    def get_token_addresses(self, pool_address: str) -> tuple[str, str]:
        """Get the token addresses by pool address."""
        token_pairs = self.db_manager.fetch_token_pairs_by_pool_address(pool_address)
        return token_pairs.get("token0"), token_pairs.get("token1")
    
    def get_last_pool_metric(
        self, pool_address: str, timestamp: int
    ) -> Dict[str, Union[int, float]]:
        """Get the last pool metrics."""
        return self.db_manager.fetch_last_pool_metric(pool_address, timestamp)
    
    def build_graphs(self, pools: list[dict]) -> dict:
        """Build the graph from the pools."""
        graphs = defaultdict(lambda: defaultdict(dict))
        for pool in pools:
            if pool["price"] > 0.0:
                graphs[pool["timestamp"]][pool["token0_address"]][pool["token1_address"]] = (pool["price"], pool["volume_token0"],  pool["liquidity_token0"])
                graphs[pool["timestamp"]][pool["token1_address"]][pool["token0_address"]] = (1 / pool["price"] if pool["price"] > 0 else 0,  pool["volume_token1"], pool["liquidity_token1"])
        # print(graphs[1620161400]['0x2260fac5e5542a773aa44fbcfedf7c193bc2c599'])
        return graphs
    
    def get_all_pool_metrics(self, start_timestamp: int, end_timestamp: int) -> list[dict]:
        """Get the pool data."""
        return self.db_manager.fetch_all_pool_metrics(start_timestamp, end_timestamp)
    
    def get_all_token_addresses(self) -> list[str]:
        """Get the token addresses."""
        return self.db_manager.fetch_all_token_addresses()
    
    def bfs_usd_price(self, graphs:dict, token_address: str, timestamp: int) -> float:
        """BFS to get the USD price."""
        if timestamp not in graphs:
            return 0
        graph = graphs[timestamp]
        visited = set()
        queue = deque([(token_address, 1)])
        if token_address not in graph:
            return 0
        while queue:
            node, price = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in STABLECOINS:
                return price
            for neighbor, (weight, _, _) in graph[node].items():
                queue.append((neighbor, price * weight))
        return 0
    
    def get_token_volume_and_liquidity(self, graphs:dict, token_address: str, timestamp) -> tuple[float, float]:
        if timestamp not in graphs:
            return 0, 0
        graph = graphs[timestamp]
        volume, liquidity = 0, 0
        for _, (_, volume, liquidity) in graph[token_address].items():
            volume += volume
            liquidity += liquidity
        return volume, liquidity
    
    def process_token_price(self, token_address: str, graphs: dict, start_timestamp: int, end_timestamp: int) -> None:
        start_timestamp = start_timestamp // METRIC_INTERVAL * METRIC_INTERVAL + METRIC_INTERVAL
        end_timestamp = end_timestamp // METRIC_INTERVAL * METRIC_INTERVAL + METRIC_INTERVAL
        updates = []
        for timestamp in range(start_timestamp, end_timestamp, METRIC_INTERVAL):
            usd_price = self.bfs_usd_price(graphs, token_address, timestamp)
            volume, liquidity = self.get_token_volume_and_liquidity(graphs, token_address, timestamp)
            updates.append((token_address, timestamp, usd_price, volume, liquidity))
        self.db_manager.add_or_update_token_metrics(updates)
    
    def process_token_prices(self, start_timestamp: int, end_timestamp: int) -> None:
        self.logger.log_info(f"Processing token prices between {start_timestamp} and {end_timestamp} ...")
        pool_metrics = self.get_all_pool_metrics(start_timestamp, end_timestamp)
        graphs = self.build_graphs(pool_metrics)
        token_addresses = self.get_all_token_addresses()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_token_price, token_address, graphs, start_timestamp, end_timestamp) for token_address in token_addresses]
            for future in futures:
                future.result()  # Wait for all futures to complete

        self.logger.log_info("Token metrics processed successfully.")
    
    def process_time_range(self, time_range: dict):
        self.logger.log_info(f'Processing time range between {timestamp_to_date(time_range["start"])} and {timestamp_to_date(time_range["end"])}')
        next_query = self.get_next_query(time_range)
        if next_query is None:
            return None
        self.logger.log_info(f"Querying uniswap pools with next_query: {next_query['pool_addresses']}")
        onchain_data = self.uniswap_fetcher.fetch_pool_data(
            next_query["token_pairs"], next_query["start_datetime"], next_query["end_datetime"]
        )
        metrics = self.generate_metrics(onchain_data, next_query["pool_addresses"], next_query["pools_map"], time_range["start"], time_range["end"])
        self.save_pool_and_metrics_data(next_query, onchain_data, metrics)
        self.logger.log_info(f"Pool data and metrics saved successfully")
    def run(self):
        while True:
            time_range = self.db_manager.fetch_last_time_range()
            if time_range is None or time_range["completed"]:
                if time_range:
                    self.process_token_prices(time_range["start"], time_range["end"])
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
