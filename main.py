from pool_data_fetcher import BlockchainClient
from db.db_manager import DBManager
import os
from datetime import datetime, timedelta

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
            start = datetime(2021, 5, 4)
            end = datetime(2021, 5, 5)
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
        
        return start, end
    
    def get_time_range(self) -> tuple[datetime, datetime]:
        """
        Get the time range for the miner modules.

        Returns:
            The time range for the miner modules.
        """
        incompleted_time_range = self.db_manager.fetch_incompleted_time_range()
        
        if not incompleted_time_range:
            return self.add_new_time_range()
        else:
            return incompleted_time_range[0]["start"], incompleted_time_range[0]["end"]
    
    def get_token_pairs(self, start: datetime, end: datetime) -> list[dict[str, str]]:
        """
        Get the token pairs for the miner modules.

        Args:
            start: The start datetime.
            end: The end datetime.

        Returns:
            The token pairs for the miner modules.
        """
        token_pairs = self.db_manager.fetch_incompleted_token_pairs()
        
        if not token_pairs:
            self.db_manager.mark_time_range_as_complete(start, end)
            return None
        return token_pairs[:80]

    def save_pool_data(self, miner_prompt: dict, miner_answer: dict) -> None:
        """
        Save the pool data to the database.
        
        Args:
            miner_prompt: The prompt for the miner modules.
            miner_answer: The generated answer from the miner module
        """
        token_pairs = miner_prompt.get("token_pairs", None)
        start_datetime = miner_prompt.get("start_datetime", None)
        end_datetime = miner_prompt.get("end_datetime", None)
        
        miner_data = miner_answer.get("data", None)
        
        self.db_manager.add_pool_data(miner_data)
        
        self.db_manager.mark_token_pairs_as_complete(token_pairs)
        
        token_pairs = self.db_manager.fetch_incompleted_token_pairs()
        
        if not token_pairs:
            self.db_manager.mark_time_range_as_complete(start_datetime, end_datetime)
    

if __name__ == '__main__':
    pass