from uniswap_fetcher_rs  import UniswapFetcher
from db.db_manager import DBManager
import os
from collections import defaultdict
from datetime import datetime, timezone
import time
import pandas as pd
from typing import Dict, Union
from utils.utils import hex_to_signed_int, tick_to_sqrt_price, calc_prices_token0_by_token1, calc_prices_token1_by_token0

TIME_INTERVAL = 10 * 60
START_TIMESTAMP = int(datetime(2021, 5, 4).replace(tzinfo=timezone.utc).timestamp())
DAY_SECONDS = 86400
STABLECOINS = [
    "0x6b175474e89094c44da98b954eedeac495271d0f", # DAI in Ethereum Mainnet
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", # USDC in Ethereum Mainnet
    "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", # USDC2
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", # USDC3
    "0xdac17f958d2ee523a2206206994597c13d831ec7", # USDT in Ethereum Mainnet
    
]


class PoolDataFetcher:
    def __init__(self) -> None:
        self.db_manager = DBManager()
        self.uniswap_fetcher = UniswapFetcher(os.getenv('ETHEREUM_RPC_NODE_URL'))
        
    def add_new_time_range(self) -> None:
        """
        Add a new timetable entry to the database.
        """
        last_time_range = self.db_manager.fetch_last_time_range()
        if last_time_range == None:
            start = START_TIMESTAMP
            end = START_TIMESTAMP + DAY_SECONDS
        else:
            start = last_time_range["end"]
            end = last_time_range["end"] + DAY_SECONDS
        print(f"Adding new timetable entry between {start} and {end}")
        self.db_manager.add_timetable_entry(start, end)
        
        print(f"Fetching token pairs between {start} and {end}")
        
        token_pairs = self.uniswap_fetcher.get_pool_created_events_between_two_timestamps(int(start), int(end))
        self.db_manager.reset_token_pairs()
        self.db_manager.add_token_pairs(token_pairs)
        
        return {'start' : start, 'end' : end}
        
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
    
    def save_pool_and_metrics_data(self, prob: dict, answer: dict, metrics: tuple) -> None:
        """
        Save the pool data to the database.
        
        Args:
            prob: The token_pair and datetime to fetch.
            answer: The fetched data from rpc node.
        """
        token_pairs = prob.get("token_pairs", None)        
        onchain_data = answer.get("data", None)
        pool_metrics, token_metrics, daily_metrics = metrics
        print(f'saving pool data to database ...')
        self.db_manager.add_or_update_daily_metrics(daily_metrics)
        self.db_manager.add_pool_event_and_metrics_data(onchain_data, pool_metrics)
        self.db_manager.add_or_update_token_metrics(token_metrics)
        self.db_manager.mark_token_pairs_as_complete(token_pairs)
        print(f'pool data saved successfully.')

    def get_next_token_pairs(self, time_range: dict) -> dict:
        """
        Get a token pair to fetch.

        Returns:
            Token pair and time range.
        """
        token_pairs = self.get_incomplete_token_pairs(time_range['start'], time_range['end'])
        if not token_pairs:
            return None
        
        req_token_pairs = []
        req_pool_addresses = []
        req_pools_map = {}
        for token_pair in token_pairs:
            req_token_pairs.append((token_pair['token0'], token_pair['token1'], token_pair['fee']))
            req_pool_addresses.append(token_pair['pool_address'])
            req_pools_map[token_pair['pool_address']] = {
                "token0": token_pair['token0'],
                "token1": token_pair['token1'],
                "is_stablecoin": token_pair["is_stablecoin"],
                "token0_decimals": token_pair["token0_decimals"],
                "token1_decimals": token_pair["token1_decimals"],
            }

        return {"token_pairs": req_token_pairs, "pool_addresses": req_pool_addresses, "pools_map": req_pools_map, "start_datetime": int(time_range['start']), "end_datetime": int(time_range['end'])}

    def process_time_range(self, time_range: dict):
        print(f'Processing time range between {time_range["start"]} and {time_range["end"]}')
        prob = self.get_next_token_pairs(time_range)
        if prob is None:
            return None
        print(f'querying uniswap_fetcher with problem: {prob}')
        answer = self.uniswap_fetcher.fetch_pool_data(prob['token_pairs'], prob['start_datetime'], prob['end_datetime'])
        print(f'received answer')
        
        metrics = self.generate_metrics(answer, prob["pools_map"], prob['pool_addresses'], time_range['start'], time_range['end'])
        print(f'saving data...')
        self.save_pool_and_metrics_data(prob, answer, metrics)
    
    def generate_metrics(self, pool_data: Dict, pools_map: dict[Dict[str, Union[str, bool, int]]], prob_pool_addresses: list, start: int, end: int, interval: int = 300) -> tuple:
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
            if pool_map.get("is_stablecoin") == False:
                db_token_metrics = self.db_manager.fetch_token_metrics(pool_map.get("token0"), start + interval, end)
                db_token_metrics.extend(self.db_manager.fetch_token_metrics(pool_map.get("token1"), start + interval, end))
                for token_metric in db_token_metrics:
                    if token_metric["token_address"] not in derived_token_metrics:
                        derived_token_metrics[token_metric["token_address"]] = {token_metric["timestamp"]: token_metric.get("close_price", 0.0)}
                    else:
                        derived_token_metrics[token_metric["token_address"]].update({token_metric["timestamp"]: token_metric.get("close_price", 0.0)})
        # raise Exception("stop")
        data = pool_data.get("data", None)
        daily_metrics = {}
        aggregated_data = defaultdict(lambda: defaultdict(list))
        datetime_series = pd.date_range(start=datetime.fromtimestamp(start + interval, tz=timezone.utc), end=datetime.fromtimestamp(end, tz=timezone.utc), freq=f'{interval}s')
        for pool_address in prob_pool_addresses:
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
        for event in data:
            round_timestamp = event.get("timestamp") // interval * interval + interval
            pool_address = event.get("pool_address")
            if pool_address not in daily_metrics:
                daily_metrics[pool_address] = {
                    "events_count": 1,
                    "volume": 0.0,
                    "liquidity": 0.0,
                    "high_price": 0.0,
                    "low_price": 0.0,
                }
            else:
                daily_metrics[pool_address]["events_count"] += 1
            key = (pool_address, round_timestamp)
            if not key in aggregated_data:
                raise Exception(f"Key {key} not found in aggregated_data")
            if event.get("event").get("type") == "swap":
                aggregated_data[key]["sqrt_price_x96"].append(event.get("event").get("data").get("sqrt_price_x96"))
            else:
                amount = hex_to_signed_int(event.get("event").get("data").get("amount", "0x0"))
                tick_lower = event.get("event").get("data").get("tick_lower")
                tick_upper = event.get("event").get("data").get("tick_upper")
                sqrt_price_lower = tick_to_sqrt_price(tick_lower)
                sqrt_price_upper = tick_to_sqrt_price(tick_upper)
                liquidity_token0 = amount * (sqrt_price_upper - sqrt_price_lower) / (sqrt_price_upper * sqrt_price_lower)
                liquidity_token1 = amount * (sqrt_price_upper - sqrt_price_lower) / sqrt_price_upper
                aggregated_data[key]["token0_liquidity"].append(liquidity_token0)
                aggregated_data[key]["token1_liquidity"].append(liquidity_token1)
                aggregated_data[key]["total_liquidity"].append(amount)
            aggregated_data[key]["amount0"].append(hex_to_signed_int(event.get("event").get("data").get("amount0")))
            aggregated_data[key]["amount1"].append(hex_to_signed_int(event.get("event").get("data").get("amount1")))
            aggregated_data[key]["type"] = event.get("event").get("type")
            
        apply_abs = lambda x: [abs(i) for i in x]
        
        pool_metrics = []
        token_metrics = []
        
        for key, value in aggregated_data.items():
            pool_address, timestamp = key
            volume_token0 = sum(apply_abs(value["amount0"]))
            volume_token1 = sum(apply_abs(value["amount1"]))
            liquidity_token0 = sum(value["token0_liquidity"])
            liquidity_token1 = sum(value["token1_liquidity"])
            
            token0_address = pools_map[pool_address].get("token0")
            token1_address = pools_map[pool_address].get("token1")
            prices_ratio_token0 = calc_prices_token0_by_token1(value["sqrt_price_x96"], pools_map[pool_address].get("token0_decimals"), pools_map[pool_address].get("token1_decimals"))
            prices_ratio_token1 = calc_prices_token1_by_token0(value["sqrt_price_x96"], pools_map[pool_address].get("token0_decimals"), pools_map[pool_address].get("token1_decimals"))
            if token0_address in STABLECOINS or token1_address in STABLECOINS:
                stable_token = token0_address if token0_address in STABLECOINS else token1_address
                non_stable_token = token0_address if token0_address not in STABLECOINS else token1_address
                
                
                stable_price = 1.0
                if len(prices_ratio_token0) == 0:
                    prices_ratio_token0 = [0.0,]
                if len(prices_ratio_token1) == 0:
                    prices_ratio_token1 = [0.0,]
                close_non_stable_price = prices_ratio_token1[-1] if token0_address in STABLECOINS else prices_ratio_token0[-1]
                high_non_stable_price = max(prices_ratio_token1) if token0_address in STABLECOINS else max(prices_ratio_token0)
                low_non_stable_price = min(prices_ratio_token1) if token0_address in STABLECOINS else min(prices_ratio_token0)
                average_non_stable_price = sum(prices_ratio_token1) / len(prices_ratio_token1) if token0_address in STABLECOINS else sum(prices_ratio_token0) / len(prices_ratio_token0)
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
                
                non_stable_token_volume = non_stable_token_volume * average_non_stable_price
                non_stable_token_liquidity = non_stable_token_liquidity * average_non_stable_price
                token_metrics.append({
                    "token_address": stable_token,
                    "timestamp": timestamp,
                    "total_volume": stable_token_volume,
                    "total_liquidity": stable_token_liquidity,
                    "high_price": stable_price,
                    "low_price": stable_price,
                    "close_price": stable_price,
                    "is_derived": False,
                })
                token_metrics.append({
                    "token_address": non_stable_token,
                    "timestamp": timestamp,
                    "total_volume": non_stable_token_volume,
                    "total_liquidity": non_stable_token_liquidity,
                    "high_price": high_non_stable_price,
                    "low_price": low_non_stable_price,
                    "close_price": close_non_stable_price,
                    "is_derived": False,
                })
                usd_volume = stable_price * stable_token_volume + average_non_stable_price * non_stable_token_volume
                usd_liquidity = stable_price * stable_token_liquidity + average_non_stable_price * non_stable_token_liquidity 
                pool_metrics.append({
                    "pool_address": pool_address,
                    "timestamp": timestamp,
                    "price": average_non_stable_price,
                    "volume": usd_volume,
                    "liquidity": usd_liquidity,
                })
            else:
                price_token0 = derived_token_metrics.get(token0_address, {}).get(timestamp, None)
                price_token1 = derived_token_metrics.get(token1_address, {}).get(timestamp, None)
                if not price_token0 or not price_token1:
                    continue
                
                if len(prices_ratio_token0) == 0:
                    prices_ratio_token0 = [0.0,]
                    prices_ratio_token1 = [0.0,]
                average_price_ratio_token0 = sum(prices_ratio_token0) / len(prices_ratio_token0)
                is_token0_derived = True
                is_token1_derived = True
                if price_token0 and not price_token1:
                    close_price_token1 = price_token0 / prices_ratio_token0[-1]
                    high_price_token1 = price_token0 / min(prices_ratio_token0)
                    low_price_token1 = price_token0 / max(prices_ratio_token0)
                    price_token1 = price_token0 / average_price_ratio_token0
                    is_token1_derived = False
                    
                elif price_token1 and not price_token0:
                    close_price_token0 = price_token1 * prices_ratio_token0[-1]
                    high_price_token0 = price_token1 * max(prices_ratio_token0)
                    low_price_token0 = price_token1 * min(prices_ratio_token0)
                    price_token0 = price_token1 * average_price_ratio_token0
                    is_token0_derived = False
                usd_volume_token0 = volume_token0 * price_token0
                usd_volume_token1 = volume_token1 * price_token1
                usd_liquidity_token0 = liquidity_token0 * price_token0
                usd_liquidity_token1 = liquidity_token1 * price_token1
                
                if is_token0_derived:
                    token_metrics.append({
                        "token_address": token0_address,
                        "timestamp": timestamp,
                        "total_volume": usd_volume_token0,
                        "total_liquidity": usd_liquidity_token0,
                        "is_derived": True,
                    })
                else:
                    token_metrics.append({
                        "token_address": token0_address,
                        "timestamp": timestamp,
                        "total_volume": usd_volume_token0,
                        "total_liquidity": usd_liquidity_token0,
                        "high_price": high_price_token0,
                        "low_price": low_price_token0,
                        "close_price": close_price_token0,
                        "is_derived": False,
                    })
                if is_token1_derived:
                    token_metrics.append({
                        "token_address": token1_address,
                        "timestamp": timestamp,
                        "total_volume": usd_volume_token1,
                        "total_liquidity": usd_liquidity_token1,
                        "is_derived": True,
                    })
                else:
                    token_metrics.append({
                        "token_address": token1_address,
                        "timestamp": timestamp,
                        "total_volume": usd_volume_token1,
                        "total_liquidity": usd_liquidity_token1,
                        "high_price": high_price_token1,
                        "low_price": low_price_token1,
                        "close_price": close_price_token1,
                        "is_derived": False,
                    })
                pool_metrics.append({
                    "pool_address": pool_address,
                    "timestamp": timestamp,
                    "price": average_price_ratio_token0,
                    "volume": usd_volume_token0 + usd_volume_token1,
                    "liquidity": usd_liquidity_token0 + usd_liquidity_token1,
                })
                
        return pool_metrics, token_metrics, daily_metrics
        
    def run(self):
        while True:
            time_range = self.db_manager.fetch_last_time_range()
            if time_range == None or time_range['completed'] == True:
                time_range = self.add_new_time_range()
                
            now = datetime.now().timestamp()
            if(time_range['end'] <= now):
                self.process_time_range(time_range)
            else:
                adjusted_time_range = {'start': time_range['start'], 'end': now}
                self.process_time_range(adjusted_time_range)
                
                prev = now
                while prev <= time_range['end']:
                    time.sleep(TIME_INTERVAL)
                    
                    now = datetime.now()
                    if now > time_range['end']:
                        now = time_range['end']
                    
                    new_time_range = {'start': prev, 'end': now}
                    self.process_time_range(new_time_range)
                    
                    prev = now
            self.get_incomplete_token_pairs(time_range['start'], time_range['end'])
    
if __name__ == '__main__':
    pool_data_fetcher = PoolDataFetcher()
    pool_data_fetcher.run()