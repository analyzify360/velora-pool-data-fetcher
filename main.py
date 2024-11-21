from uniswap_fetcher_rs  import UniswapFetcher
from db.db_manager import DBManager
import os
from collections import defaultdict
from datetime import datetime, timezone
import time
import pandas as pd
from utils.utils import hex_to_signed_int

TIME_INTERVAL = 10 * 60
START_TIMESTAMP = int(datetime(2021, 5, 4).replace(tzinfo=timezone.utc).timestamp())
DAY_SECONDS = 86400


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
        return token_pairs[:10]
    
    def save_pool_and_signals_data(self, prob: dict, answer: dict, signals: tuple) -> None:
        """
        Save the pool data to the database.
        
        Args:
            prob: The token_pair and datetime to fetch.
            answer: The fetched data from rpc node.
        """
        token_pairs = prob.get("token_pairs", None)        
        miner_data = answer.get("data", None)
        metrics, daily_metrics = signals
        print(f'saving pool data to database ...')
        self.db_manager.add_or_update_daily_metrics(daily_metrics)
        self.db_manager.add_pool_and_signals_data(miner_data, metrics)
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
        for token_pair in token_pairs:
            req_token_pairs.append((token_pair['token0'], token_pair['token1'], token_pair['fee']))
            req_pool_addresses.append(token_pair['pool_address'])

        return {"token_pairs": req_token_pairs, "pool_addresses": req_pool_addresses, "start_datetime": int(time_range['start']), "end_datetime": int(time_range['end'])}

    def process_time_range(self, time_range: dict):
        print(f'Processing time range between {time_range["start"]} and {time_range["end"]}')
        prob = self.get_next_token_pairs(time_range)
        if prob is None:
            return None
        print(f'querying uniswap_fetcher with problem: {prob}')
        answer = self.uniswap_fetcher.fetch_pool_data(prob['token_pairs'], prob['start_datetime'], prob['end_datetime'])
        print(f'received answer')
        signals = self.generate_signals(answer, prob['pool_addresses'], time_range['start'], time_range['end'])
        print(f'saving data...')
        self.save_pool_and_signals_data(prob, answer, signals)
    
    def generate_signals(self, pool_data: dict, prob_pool_addresses: list, start: int, end: int, interval: int = 300) -> tuple:
        """
        Generate signals from the pool data.

        Args:
            pool_data: The pool data to generate signals from.
            start: The start datetime for aggregation.
            end: The end datetime for aggregation.
            interval: The interval for aggregation.
        """
        data = pool_data.get("data", None)
        if not data:
            return None
        aggregated_data = defaultdict(lambda: defaultdict(list))
        datetime_series = pd.date_range(start=datetime.fromtimestamp(start + interval, tz=timezone.utc), end=datetime.fromtimestamp(end, tz=timezone.utc), freq=f'{interval}s')
        for pool_address in prob_pool_addresses:
            for date_time in datetime_series:
                timestamp = int(date_time.timestamp())
                key = (pool_address, timestamp)
                aggregated_data[key]["amount"] = []
                aggregated_data[key]["amount0"] = []
                aggregated_data[key]["amount1"] = []
                aggregated_data[key]["sqrt_price_x96"] = []
                aggregated_data[key]["type"] = None
        for event in data:
            # print(f"timestamp: {event.get('timestamp')}")
            round_timestamp = event.get("timestamp") // interval * interval
            key = (event.get("pool_address"), round_timestamp)
            if not key in aggregated_data:
                raise Exception(f"Key {key} not found in aggregated_data")
            if event.get("event").get("type") == "swap":
                aggregated_data[key]["sqrt_price_x96"].append(event.get("event").get("data").get("sqrt_price_x96"))
            else:
                aggregated_data[key]["amount"].append(hex_to_signed_int(event.get("event").get("data").get("amount", "0x0")))
            aggregated_data[key]["amount0"].append(hex_to_signed_int(event.get("event").get("data").get("amount0")))
            aggregated_data[key]["amount1"].append(hex_to_signed_int(event.get("event").get("data").get("amount1")))
            aggregated_data[key]["type"] = event.get("event").get("type")
            
        # print(f'printing aggregated data: {aggregated_data}')
        apply_abs = lambda x: [abs(i) for i in x]
        calc_price = lambda x: [float((hex_to_signed_int(i)) / (2**96)) ** 2 for i in x]
        
        metrics = []
        daily_metrics = {}
        for key, value in aggregated_data.items():
            pool_address, timestamp = key
            volume = sum(apply_abs(value["amount0"])) + sum(value["amount1"])
            liquidity = sum(value["amount"])
            if len(value["sqrt_price_x96"]) == 0.0:
                price = 0.0
            else:
                price = sum(calc_price(value["sqrt_price_x96"])) / len(value["sqrt_price_x96"])
            
            if pool_address not in daily_metrics:
                daily_metrics[pool_address] = {
                    "volume": volume,
                    "liquidity": liquidity,
                    "price_high": price,
                    "price_low": price,
                }
            else:
                daily_metrics[pool_address]["volume"] += volume
                daily_metrics[pool_address]["liquidity"] += liquidity
                daily_metrics[pool_address]["price_high"] = max(daily_metrics[pool_address]["price_high"], price)
                daily_metrics[pool_address]["price_low"] = min(daily_metrics[pool_address]["price_low"], price)
            
            metrics.append({
                "pool_address": pool_address,
                "timestamp": timestamp,
                "volume": volume,
                "liquidity": liquidity,
                "price": price,
            })
            
        return metrics, daily_metrics
        
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