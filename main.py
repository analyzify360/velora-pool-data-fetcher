from uniswap_fetcher_rs  import UniswapFetcher
from db.db_manager import DBManager
import os
from datetime import datetime, timedelta
import time
import pandas as pd

TIME_INTERVAL = 10 * 60
START_DATE = datetime(2021, 5, 4)

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
            start = START_DATE
            end = START_DATE + timedelta(days=1)
        else:
            start = last_time_range["end"]
            end = last_time_range["end"] + timedelta(days=1)
        
        self.db_manager.add_timetable_entry(start, end)
        
        print(f"Fetching token pairs between {start} {int(start.timestamp())} and {end} {int(end.timestamp())}")
        
        token_pairs = self.uniswap_fetcher.get_pool_created_events_between_two_timestamps(int(start.timestamp()), int(end.timestamp()))
        self.db_manager.reset_token_pairs()
        self.db_manager.add_token_pairs(token_pairs)
        
        return {'start' : start, 'end' : end}
        
    def get_incomplete_token_pairs(self, start: datetime, end: datetime) -> list[dict[str, str]]:
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
    
    def save_pool_data(self, prob: dict, answer: dict) -> None:
        """
        Save the pool data to the database.
        
        Args:
            prob: The token_pair and datetime to fetch.
            answer: The fetched data from rpc node.
        """
        token_pairs = prob.get("token_pairs", None)        
        miner_data = answer.get("data", None)
        print(f'saving pool data to database ...')
        
        self.db_manager.add_pool_data(miner_data)
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
        for token_pair in token_pairs:
            req_token_pairs.append((token_pair['token0'], token_pair['token1'], token_pair['fee']))

        return {"token_pairs": req_token_pairs, "start_datetime": int(time_range['start'].timestamp()), "end_datetime": int(time_range['end'].timestamp())}

    def process_time_range(self, time_range: dict):
        print(f'Processing time range between {time_range["start"]} and {time_range["end"]}')
        prob = self.get_next_token_pairs(time_range)
        if prob is None:
            return None
            
        # print(f'querying uniswap_fetcher with problem: {prob}')
        answer = self.uniswap_fetcher.fetch_pool_data(prob['token_pairs'], prob['start_datetime'], prob['end_datetime'])
        self.generate_signals(answer, time_range['start'], time_range['end'])
        self.save_pool_data(prob, answer)
    
    def generate_signals(self, pool_data: dict, start: datetime, end: datetime, interval: str = '5min') -> None:
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
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Convert to datetime
        df['amount'] = df['event'].apply(lambda x: int(x['data']['amount'], 16) if 'amount' in x['data'] else None)
        df['amount0'] = df['event'].apply(lambda x: int(x['data']['amount0'], 16))
        df['amount1'] = df['event'].apply(lambda x: int(x['data']['amount1'], 16))
        df['sqrt_price_x96'] = df['event'].apply(lambda x: int(x['data']['sqrt_price_x96'], 16) if 'sqrt_price_x96' in x['data'] else None)
        df['type'] = df['event'].apply(lambda x: x['type'])
        # Helper function to calculate price from sqrt_price_x96
        def calculate_price(sqrt_price_x96):
            res = (sqrt_price_x96 / (2**96)) ** 2
            return res

        # Calculate price of token0 based on token1 using sqrt_price_x96
        df['price'] = df['sqrt_price_x96'].apply(
            lambda x: calculate_price(x) if x is not None else None
        )
        
        # Replace NaN values with None (NULL in database)
        
        for pool_address, group in df.groupby('pool_address'):
            metrics = self.calculate_metrics_by_interval(group[group['type'] == 'swap'], group[group['type'] == 'mint'], group[group['type'] == 'burn'], start, end, interval)
            
            signals = [
                {
                    'timestamp': row['timestamp'],
                    'price': row['price'] if pd.notna(row['price']) else None,
                    'volume': row['volume'] if pd.notna(row['volume']) else None,
                    'liquidity': row['net_liquidity'] if pd.notna(row['net_liquidity']) else None,
                    'pool_address': pool_address
                }
                for __, row in metrics.iterrows()
            ]
            self.db_manager.add_uniswap_signals(signals)

    # Define a function to calculate metrics per interval
    def calculate_metrics_by_interval(self, swap_events, mint_events, burn_events, start, end, interval):
        # Create a date range for the custom timestamp
        date_range = pd.date_range(start=start, end=end, freq=interval)
        
        swap_events = swap_events.groupby('timestamp').agg({'amount0': 'sum', 'amount1': 'sum', 'price': 'mean'}).reset_index()
        mint_events = mint_events.groupby('timestamp').agg({'amount': 'sum'}).reset_index()
        burn_events = burn_events.groupby('timestamp').agg({'amount': 'sum'}).reset_index()
        
        # Volume and Price from Swap Events
        swap_resampled = swap_events.set_index('timestamp').reindex(date_range, method='nearest').resample(interval)
        
        # Calculate price using sqrt_price_x96
        price_sqrt = swap_resampled['price'].mean()
        
        # Calculate volume as the sum of absolute amounts
        volume0 = swap_resampled['amount0'].sum().abs()  # Volume calculation
        volume1 = swap_resampled['amount1'].sum().abs() # Volume calculation
        volume = volume0 + volume1
        volume.name = 'volume'
        
        # Liquidity from Mint and Burn Events
        mint_resampled = mint_events.set_index('timestamp').reindex(date_range, method='nearest').resample(interval)['amount'].sum().fillna(0)
        burn_resampled = burn_events.set_index('timestamp').reindex(date_range, method='nearest').resample(interval)['amount'].sum().fillna(0)
        
        # Net liquidity change = Mints - Burns
        net_liquidity = (mint_resampled - burn_resampled).cumsum()
        net_liquidity.name = 'net_liquidity'
        
        # Combine all metrics
        metrics = pd.concat([price_sqrt, volume, net_liquidity], axis=1)
        metrics.index.name = 'timestamp'
        metrics.reset_index(inplace=True)
        
        return metrics

    def run(self):
        while True:
            time_range = self.db_manager.fetch_last_time_range()
            if time_range == None or time_range['completed'] == True:
                time_range = self.add_new_time_range()
                
            now = datetime.now()
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