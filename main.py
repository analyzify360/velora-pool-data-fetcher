from pool_data_fetcher import BlockchainClient
from db.db_manager import DBManager
import os
from datetime import datetime, timedelta
import time

TIME_INTERVAL = 10 * 60
START_DATE = datetime(2021, 5, 4)

class PoolDataFetcher:
    def __init__(self) -> None:
        self.db_manager = DBManager()
        self.pool_data_fetcher = BlockchainClient(os.getenv('ETHEREUM_RPC_NODE_URL'))
        
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
        
        start_date_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Fetching token pairs between {start_date_str} and {end_date_str}")
        
        token_pairs = self.pool_data_fetcher.get_pool_created_events_between_two_timestamps(start_date_str, end_date_str)
        self.db_manager.reset_token_pairs()
        self.db_manager.add_token_pairs(token_pairs)
        
        return {'start' : start, 'end' : end}
    
    def get_token_pairs(self, start: datetime, end: datetime) -> list[dict[str, str]]:
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
        return token_pairs[:80]

    def save_pool_data(self, prob: dict, answer: dict) -> None:
        """
        Save the pool data to the database.
        
        Args:
            prob: The token_pair and datetime to fetch.
            answer: The fetched data from rpc node.
        """
        token_pairs = prob.get("token_pairs", None)        
        miner_data = answer.get("data", None)
        
        self.db_manager.add_pool_data(miner_data)        
        self.db_manager.mark_token_pairs_as_complete(token_pairs)

    def get_next_token_pairs(self, time_range: dict) -> dict:
        """
        Get a token pair to fetch.

        Returns:
            Token pair and time range.
        """
        token_pairs = self.get_token_pairs(time_range['start'], time_range['end'])
        
        # Implement your custom prompt generation logic here
        start_datetime=time_range[0].strftime("%Y-%m-%d %H:%M:%S")
        end_datetime=time_range[1].strftime("%Y-%m-%d %H:%M:%S")
        
        req_token_pairs = []
        for token_pair in token_pairs:
            req_token_pairs.append((token_pair['token0'], token_pair['token1'], token_pair['fee']))

        return {"token_pairs": req_token_pairs, "start_datetime": start_datetime, "end_datetime": end_datetime}

    def process_time_range(self, time_range: dict):
        print(f'Processing time range {time_range.values()}')
        prob = self.get_next_token_pairs(time_range)
        answer = self.pool_data_fetcher.fetch_pool_data(prob['token_pairs'], prob['start_datetime'], prob['end_datetime'], '1h')
        
        pool_data_fetcher.save_pool_data(prob, answer)

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
    
if __name__ == '__main__':
    pool_data_fetcher = PoolDataFetcher()
    pool_data_fetcher.run()