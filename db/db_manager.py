from sqlalchemy import (
    create_engine,
    Column,
    Boolean,
    MetaData,
    String,
    Integer,
    Float,
    inspect,
    text,
    UniqueConstraint,
    Index,
    or_,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from sqlalchemy.dialects.postgresql import insert
from typing import Union, List, Dict
from utils.config import get_postgres_url
from utils.utils import has_stablecoin
from utils.log import Logger

from datetime import datetime, timezone

# Define the base class for your table models
Base = declarative_base()
STABLECOINS = [
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI in Ethereum Mainnet
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC in Ethereum Mainnet
    "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC2
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC3
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT in Ethereum Mainnet
]

START_TIMESTAMP = int(datetime(2021, 5, 4).replace(tzinfo=timezone.utc).timestamp())
# Define the timetable table
class Timetable(Base):
    __tablename__ = "timetable"
    start = Column(
        Integer, primary_key=True
    )  # Assuming 'start' is a unique field, hence primary key
    end = Column(Integer)
    completed = Column(Boolean)

class TokenPairTable(Base):
    __tablename__ = 'token_pairs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    token0 = Column(String, nullable=False)
    token1 = Column(String, nullable=False)
    has_stablecoin = Column(Boolean, nullable=False)
    indexed = Column(Boolean, nullable=False)
    fee = Column(Integer, nullable=False)
    pool = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
    completed = Column(Boolean, nullable=False)
    last_synced_time = Column(Integer, nullable=True)


class TokenTable(Base):
    __tablename__ = "tokens"
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    decimals = Column(Integer, nullable=False)


class CurrentPoolMetricTable(Base):
    __tablename__ = "current_pool_metrics"
    pool_address = Column(String, primary_key=True)
    price = Column(Float)
    liquidity_token0 = Column(Float)
    liquidity_token1 = Column(Float)
    volume_token0 = Column(Float)
    volume_token1 = Column(Float)


class CurrentTokenMetricTable(Base):
    __tablename__ = "current_token_metrics"
    token_address = Column(String, primary_key=True)
    price = Column(Float, default=0.0)
    total_liquidity = Column(Float, default=0.0)
    total_volume = Column(Float, default=0.0)
    is_used = Column(Boolean)


class SwapEventTable(Base):
    __tablename__ = "swap_event"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String, nullable=False)
    pool_address = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    sender = Column(String, nullable=False)
    to = Column(String, nullable=False)
    amount0 = Column(String, nullable=False)  # I256 can be stored as String
    amount1 = Column(String, nullable=False)  # I256 can be stored as String
    sqrt_price_x96 = Column(String, nullable=False)  # U256 can be stored as String
    liquidity = Column(String, nullable=False)  # U256 can be stored as String
    tick = Column(Integer, nullable=False)  # i32 can be stored as Integer


class MintEventTable(Base):
    __tablename__ = "mint_event"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String, nullable=False)
    pool_address = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    sender = Column(String, nullable=False)
    owner = Column(String, nullable=False)
    tick_lower = Column(Integer, nullable=False)  # int24 can be stored as Integer
    tick_upper = Column(Integer, nullable=False)  # int24 can be stored as Integer
    amount = Column(String, nullable=False)  # U256 can be stored as String
    amount0 = Column(String, nullable=False)  # U256 can be stored as String
    amount1 = Column(String, nullable=False)  # U256 can be stored as String


class BurnEventTable(Base):
    __tablename__ = "burn_event"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String, nullable=False)
    pool_address = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    owner = Column(String, nullable=False)
    tick_lower = Column(Integer, nullable=False)  # int24 can be stored as Integer
    tick_upper = Column(Integer, nullable=False)  # int24 can be stored as Integer
    amount = Column(String, nullable=False)  # U256 can be stored as String
    amount0 = Column(String, nullable=False)  # U256 can be stored as String
    amount1 = Column(String, nullable=False)  # U256 can be stored as String


class CollectEventTable(Base):
    __tablename__ = "collect_event"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String, nullable=False)
    pool_address = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    owner = Column(String, nullable=False)
    recipient = Column(String, nullable=False)
    tick_lower = Column(Integer, nullable=False)  # int24 can be stored as Integer
    tick_upper = Column(Integer, nullable=False)  # int24 can be stored as Integer
    amount0 = Column(String, nullable=False)  # U256 can be stored as String
    amount1 = Column(String, nullable=False)  # U256 can be stored as String


class PoolMetricTable(Base):
    __tablename__ = "pool_metrics"

    # id = Column(Integer, autoincrement=True)  # Surrogate unique ID
    timestamp = Column(Integer, primary_key=True, nullable=False)  # Time-based column for partitioning
    pool_address = Column(String, primary_key=True, nullable=False)  # Pool address
    price = Column(Float, default=0.0)
    liquidity_token0 = Column(Float, default=0.0)
    liquidity_token1 = Column(Float, default=0.0)
    volume_token0 = Column(Float, default=0.0)
    volume_token1 = Column(Float, default=0.0)

    # Define a unique index to ensure uniqueness
    __table_args__ = (
        Index("idx_unique_timestamp_pool", "timestamp", "pool_address", unique=True),
    )


class TokenMetricTable(Base):
    __tablename__ = "token_metrics"
    timestamp = Column(Integer, nullable=False, primary_key=True)
    token_address = Column(String, nullable=False, primary_key=True)

    close_price = Column(Float, default=0.0)
    high_price = Column(Float, default=0.0)
    low_price = Column(Float, default=0.0)
    total_volume = Column(Float, default=0.0)
    total_liquidity = Column(Float, default=0.0)
    
    __table_args__ = (
        Index("idx_token_timestamp", "timestamp", "token_address", unique=True),
    )


class DBManager:
    def __init__(self, url=get_postgres_url()) -> None:
        self.logger = Logger("DBManager")
        # Create the SQLAlchemy engine
        self.engine = create_engine(url)

        # Create a configured "Session" class
        self.Session = sessionmaker(bind=self.engine)

        # Create the table if it doesn't exist
        # Base.metadata.create_all(self.engine)  # This line ensures the table is created if not exists
        self.initialize_database()

        # Enable TimescaleDB and convert specific tables to hypertables
        self.create_hypertables()
        

    def initialize_database(self):
        try:
            with self.engine.connect() as conn:
                inspector = inspect(conn)
                if not self.compare_schemas(self.engine):
                    with self.engine.connect() as conn:
                        inspector = inspect(self.engine)
                        table_names = inspector.get_table_names()
                        for table_name in table_names:
                            try:
                                if table_name in Base.metadata.tables:
                                    conn.execute(
                                        text(
                                            f"DROP TABLE IF EXISTS {table_name} CASCADE"
                                        )
                                    )
                                    conn.commit()
                            except ProgrammingError as e:
                                self.logger.log_error(f"Failed to drop table {table_name}: {e}")

                Base.metadata.create_all(self.engine)
        except OperationalError as e:
            self.logger.log_error(f"Database connection error: {e}")
            raise

    def compare_schemas(self, engine):
        # Reflect the database schema
        metadata = MetaData()
        metadata.reflect(bind=engine)

        existing_tables = set(metadata.tables.keys())
        model_tables = set(Base.metadata.tables.keys())
        self.logger.log_info("compare_schemas start")
        # Compare table names
        if not model_tables <= existing_tables:
            return False
        inspector = inspect(engine)

        for table_name in existing_tables.intersection(model_tables):
            existing_columns = set(c["name"] for c in inspector.get_columns(table_name))
            model_columns = set(
                c.name for c in Base.metadata.tables[table_name].columns
            )

            # Compare columns
            if existing_columns != model_columns:
                return False

            # Add more detailed comparison logic if needed
            existing_constraints = {
                c["name"]: c for c in inspector.get_unique_constraints(table_name)
            }
            model_constraints = {
                c.name: c
                for c in Base.metadata.tables[table_name].constraints
                if isinstance(c, UniqueConstraint)
            }

            if set(existing_constraints.keys()) != set(model_constraints.keys()):
                return False

            for name in existing_constraints.keys():
                if existing_constraints[name]["column_names"] != list(
                    model_constraints[name].columns.keys()
                ):
                    return False

        return True

    def create_hypertables(self):
        """Enable TimescaleDB extension and convert tables to hypertables."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                # Check if TimescaleDB extension is already installed
                result = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb';")
                )
                if not result.fetchone():
                    conn.execute(
                        text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                    )
                    self.logger.log_info("TimescaleDB extension created successfully.")
                else:
                    self.logger.log_info("TimescaleDB extension already exists.")

                # Check if hypertable is enabled
                result = conn.execute(
                    text("SELECT * FROM timescaledb_information.hypertables;")
                ).fetchall()
                hypertables = [entry.hypertable_name for entry in result]

                tables = [
                    "token_pairs",
                    "tokens",
                    "swap_event",
                    "mint_event",
                    "burn_event",
                    "collect_event",
                ]
                for table in tables:
                    if table not in hypertables:
                        conn.execute(
                            text(
                                f"SELECT create_hypertable('{table}', 'id', if_not_exists => TRUE, migrate_data => true);"
                            )
                        )
                        self.logger.log_info(f"Hypertable '{table}' created successfully.")
                    else:
                        self.logger.log_info(f"Hypertable '{table}' already exists.")
                conn.execute(
                    text(
                        """
                    SELECT create_hypertable(
                        'pool_metrics',
                        'timestamp',
                        'pool_address',
                        if_not_exists => TRUE, 
                        migrate_data => TRUE, 
                        chunk_time_interval => 86400,
                        number_partitions => 6000
                    );
                    """
                    )
                )
                self.logger.log_info("Hypertable 'pool_metrics' created successfully.")
                conn.execute(
                    text(
                        """
                    SELECT create_hypertable(
                        'token_metrics',
                        'timestamp',
                        'token_address',
                        if_not_exists => TRUE, 
                        migrate_data => true, 
                        chunk_time_interval => 86400,
                        number_partitions => 6000
                    );
                    """
                    )
                )
                self.logger.log_info("Hypertable 'token_metrics' created successfully.")

            except SQLAlchemyError as e:
                self.logger.log_error(f"An error occurred: {e}")

    def __enter__(self):
        self.session = self.Session()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Don't forget to close the session
        self.session.close()

    def add_timetable_entry(self, start: Integer, end: Integer) -> None:
        """Add a new timetable entry to the database."""
        with self.Session() as session:
            new_entry = Timetable(start=start, end=end, completed=False)
            session.add(new_entry)
            session.commit()

    def fetch_timetable_data(self) -> List[Dict[str, Union[Integer, bool]]]:
        """Fetch all timetable data from the database."""
        with self.Session() as session:
            timetable_data = session.query(Timetable).all()
            return [
                {"start": row.start, "end": row.end, "completed": row.completed}
                for row in timetable_data
            ]

    def fetch_incompleted_time_range(self) -> List[Dict[str, Union[Integer, bool]]]:
        """Fetch all not completed time ranges from the timetable."""
        with self.Session() as session:
            not_completed_data = (
                session.query(Timetable).filter_by(completed=False).all()
            )
            return [
                {"start": row.start, "end": row.end, "completed": row.completed}
                for row in not_completed_data
            ]

    def fetch_last_time_range(self) -> Dict[str, Union[datetime, bool]]:
        """Fetch the last time range from the timetable."""
        with self.Session() as session:
            last_time_range = (
                session.query(Timetable).order_by(Timetable.start.desc()).first()
            )
            if last_time_range is not None:
                return {
                    "start": last_time_range.start,
                    "end": last_time_range.end,
                    "completed": last_time_range.completed,
                }
            else:
                return None

    def mark_time_range_as_complete(self, start: Integer, end: Integer) -> bool:
        """Mark a timetable entry as complete."""
        with self.Session() as session:
            record = session.query(Timetable).filter_by(start=start, end=end).first()
            if record:
                record.completed = True
                session.commit()
                return True
            return False

    def add_tokens(self, tokens: List[Dict[str, Union[str, Integer]]]) -> None:
        """Add tokens to the corresponding table."""
        with self.Session() as session:
            for token in tokens:
                exists = (
                    session.query(TokenTable)
                    .filter_by(address=token["address"])
                    .first()
                )
                if not exists:
                    new_token = TokenTable(
                        address=token["address"],
                        symbol=token["symbol"],
                        name=token["name"],
                        decimals=token["decimals"],
                    )
                    session.add(new_token)
            session.commit()

    def add_token_pairs(
        self, token_pairs: List[Dict[str, Union[str, Integer]]], timestamp: int = START_TIMESTAMP
    ) -> None:
        """Add token pairs to the corresponding table."""
        with self.Session() as session:
            for token_pair in token_pairs:
                existing_token = session.query(TokenPairTable).filter_by(pool = token_pair['pool_address']).first()
                if existing_token:
                    existing_token.indexed = True
                    session.commit()
                else:
                    new_token_pair = TokenPairTable(
                        token0=token_pair["token0"]["address"],
                        token1=token_pair["token1"]["address"],
                        has_stablecoin=has_stablecoin(token_pair),
                        indexed=True,
                        fee=token_pair["fee"],
                        pool=token_pair["pool_address"],
                        block_number=token_pair["block_number"],
                        completed=False,
                        last_synced_time=timestamp
                    )
                    session.add(new_token_pair)
                    self.add_tokens([token_pair["token0"], token_pair["token1"]])
                    session.commit()

    def fetch_token_pairs(self):
        """Fetch all token pairs from the corresponding table."""
        with self.Session() as session:
            token_pairs = session.query(TokenPairTable).all()
            return [
                {
                    "token0": row.token0,
                    "token1": row.token1,
                    "fee": row.fee,
                    "completed": row.completed,
                }
                for row in token_pairs
            ]

    def fetch_incompleted_token_pairs(
        self,
    ) -> List[Dict[str, Union[str, Integer, bool]]]:
        """Fetch all incompleted token pairs from the corresponding table."""
        with self.Session() as session:
            Token0 = aliased(TokenTable)
            Token1 = aliased(TokenTable)
            incompleted_token_pairs = (
                session.query(
                    TokenPairTable,
                    Token0.decimals.label("token0_decimals"),
                    Token1.decimals.label("token1_decimals"),
                )
                .join(Token0, TokenPairTable.token0 == Token0.address)
                .join(Token1, TokenPairTable.token1 == Token1.address)
                .filter(TokenPairTable.indexed == True)
                .filter(TokenPairTable.completed == False)
                .filter(TokenPairTable.has_stablecoin == True)
                .all()
            )

            if not incompleted_token_pairs:
                incompleted_token_pairs = (
                    session.query(
                        TokenPairTable,
                        Token0.decimals.label("token0_decimals"),
                        Token1.decimals.label("token1_decimals"),
                    )
                    .join(Token0, TokenPairTable.token0 == Token0.address)
                    .join(Token1, TokenPairTable.token1 == Token1.address)
                    .filter(TokenPairTable.indexed == True)
                    .filter(TokenPairTable.completed == False)
                    .filter(TokenPairTable.has_stablecoin == False)
                    .all()
                )

            return [
                {
                    "token0": row.TokenPairTable.token0,
                    "token1": row.TokenPairTable.token1,
                    "fee": row.TokenPairTable.fee,
                    "pool_address": row.TokenPairTable.pool,
                    "has_stablecoin": row.TokenPairTable.has_stablecoin,
                    "indexed": row.TokenPairTable.indexed,
                    "completed": row.TokenPairTable.completed,
                    "token0_decimals": row.token0_decimals,
                    "token1_decimals": row.token1_decimals,
                }
                for row in incompleted_token_pairs
            ]

    def mark_token_pairs_as_complete(self, token_pairs: List[tuple]) -> bool:
        """Mark a token pair as complete."""
        with self.Session() as session:
            for token_pair in token_pairs:
                record = (
                    session.query(TokenPairTable)
                    .filter_by(
                        token0=token_pair[0], token1=token_pair[1], fee=token_pair[2], indexed=True
                    )
                    .first()
                )
                if record:
                    session.query(TokenPairTable).filter_by(
                        token0=token_pair[0], token1=token_pair[1], fee=token_pair[2], indexed=True
                    ).update({TokenPairTable.completed: True})
                else:
                    return False
            session.commit()
            return True

    def reset_token_pairs(self):
        """Reset the token pairs completed state"""
        with self.Session() as session:
            session.query(TokenPairTable).update({TokenPairTable.completed: False})
            session.commit()

    def add_pool_event_and_metrics_data(
        self, pool_data: List[Dict], pool_metrics: List[Dict]
    ) -> None:
        """Add pool data to the pool data table and related event tables, and then add Uniswap signals."""

        with self.Session() as session:
            try:
                # Add the swap event data to the swap event table
                swap_event_data = [
                    SwapEventTable(
                        transaction_hash=data["transaction_hash"],
                        pool_address=data["pool_address"],
                        block_number=data["block_number"],
                        timestamp=data["timestamp"],
                        **data["event"]["data"],
                    )
                    for data in pool_data
                    if data["event"]["type"] == "swap"
                ]
                if swap_event_data:
                    session.add_all(swap_event_data)

                # Add the mint event data to the mint event table
                mint_event_data = [
                    MintEventTable(
                        transaction_hash=data["transaction_hash"],
                        pool_address=data["pool_address"],
                        block_number=data["block_number"],
                        timestamp=data["timestamp"],
                        **data["event"]["data"],
                    )
                    for data in pool_data
                    if data["event"]["type"] == "mint"
                ]
                if mint_event_data:
                    session.add_all(mint_event_data)

                # Add the burn event data to the burn event table
                burn_event_data = [
                    BurnEventTable(
                        transaction_hash=data["transaction_hash"],
                        pool_address=data["pool_address"],
                        block_number=data["block_number"],
                        timestamp=data["timestamp"],
                        **data["event"]["data"],
                    )
                    for data in pool_data
                    if data["event"]["type"] == "burn"
                ]
                if burn_event_data:
                    session.add_all(burn_event_data)

                # Add the collect event data to the collect event table
                collect_event_data = [
                    CollectEventTable(
                        transaction_hash=data["transaction_hash"],
                        pool_address=data["pool_address"],
                        block_number=data["block_number"],
                        timestamp=data["timestamp"],
                        **data["event"]["data"],
                    )
                    for data in pool_data
                    if data["event"]["type"] == "collect"
                ]
                if collect_event_data:
                    session.add_all(collect_event_data)
                
                batch_size = 100  # Increase the batch size for better efficiency

                for i in range(0, len(pool_metrics), batch_size):
                    batch = pool_metrics[i : i + batch_size]

                    # Prepare a list of upsert operations using SQLAlchemy's 'insert' with 'on_conflict_do_update'
                    upsert_data = []
                    for metric in batch:
                        upsert_data.append(
                            {
                                "timestamp": metric["timestamp"],
                                "pool_address": metric["pool_address"],
                                "price": metric["price"],
                                "liquidity_token0": metric["liquidity_token0"],
                                "liquidity_token1": metric["liquidity_token1"],
                                "volume_token0": metric["volume_token0"],
                                "volume_token1": metric["volume_token1"],
                            }
                        )

                    # Perform an upsert (insert or update on conflict)
                    stmt = insert(PoolMetricTable).values(upsert_data)

                    # Handle conflicts: Update values if the record already exists
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["timestamp", "pool_address"],
                        set_={
                            "price": stmt.excluded.price,
                            "liquidity_token0": PoolMetricTable.liquidity_token0 + stmt.excluded.liquidity_token0,
                            "liquidity_token1": PoolMetricTable.liquidity_token1 + stmt.excluded.liquidity_token1,
                            "volume_token0": PoolMetricTable.volume_token0 + stmt.excluded.volume_token0,
                            "volume_token1": PoolMetricTable.volume_token1 + stmt.excluded.volume_token1,
                        },
                    )

                    # Execute the statement as a batch
                    session.execute(stmt)
                # Commit the transaction if all operations succeed
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                self.logger.log_error(f"An error occurred: {e}")

    def fetch_token_pairs_by_pool_address(self, pool_address: str) -> Dict[str, Union[str, int]]:
        """Fetch token pairs by pool address."""
        with self.Session() as session:
            token_pair = (
                session.query(TokenPairTable)
                .filter_by(pool=pool_address)
                .first()
            )
            return {
                "token0": token_pair.token0,
                "token1": token_pair.token1,
                "fee": token_pair.fee,
                "completed": token_pair.completed,
            }
    
    def fetch_last_token_metric(self, token_address: str, timestamp: int) -> Dict[str, Union[int, float]]:
        """Fetch the last token metric."""
        with self.Session() as session:
            metric = (
                session.query(TokenMetricTable)
                .filter(TokenMetricTable.token_address == token_address)
                .filter(TokenMetricTable.timestamp <= timestamp)
                .order_by(TokenMetricTable.timestamp.desc())
                .first()
            )
            if not metric:
                return None
            return {
                "close_price": metric.close_price,
                "high_price": metric.high_price,
                "low_price": metric.low_price,
                "total_volume": metric.total_volume,
                "total_liquidity": metric.total_liquidity,
            }
    
    def fetch_last_pool_metric(self, pool_address: str, timestamp: int) -> Dict[str, Union[int, float]]:
        """Fetch the last pool metric."""
        with self.Session() as session:
            metric = (
                session.query(PoolMetricTable)
                .filter(PoolMetricTable.pool_address == pool_address)
                .filter(PoolMetricTable.timestamp <= timestamp)
                .order_by(PoolMetricTable.timestamp.desc())
                .first()
            )
            if metric:
                return {
                    "price": metric.price,
                    "liquidity_token0": metric.liquidity_token0,
                    "liquidity_token1": metric.liquidity_token1,
                    "volume_token0": metric.volume_token0,
                    "volume_token1": metric.volume_token1,
                }
            return None
    
    def fetch_all_pool_metrics(self, start_timestamp: int, end_timestamp: int) -> List[Dict[str, Union[int, float]]]:
        """Fetch all pool metric prices."""
        with self.Session() as session:
            metrics = (
                session.query(
                    TokenPairTable.token0,
                    TokenPairTable.token1,
                    PoolMetricTable.timestamp,
                    PoolMetricTable.price,
                    PoolMetricTable.volume_token0,
                    PoolMetricTable.volume_token1,
                    PoolMetricTable.liquidity_token0,
                    PoolMetricTable.liquidity_token1
                )
                .join(TokenPairTable, PoolMetricTable.pool_address == TokenPairTable.pool)
                .filter(PoolMetricTable.timestamp >= start_timestamp)
                .filter(PoolMetricTable.timestamp <= end_timestamp)
                .all()
            )
            return [
                {
                    "token0_address": row.token0,
                    "token1_address": row.token1,
                    "timestamp": row.timestamp,
                    "price": row.price,
                    "volume_token0": row.volume_token0,
                    "volume_token1": row.volume_token1,
                    "liquidity_token0": row.liquidity_token0,
                    "liquidity_token1": row.liquidity_token1,
                }
                for row in metrics
            ]
    def fetch_all_token_addresses(self) -> List[str]:
        with self.Session() as session:
            tokens = session.query(TokenTable).all()
            return [ row.address for row in tokens ]
    
    def add_or_update_token_metrics(self, prices: list[tuple[str, int, float, float, float]]) -> None:
        with self.Session() as session:
            update_data = [
            {"token_address": token_address, "timestamp": timestamp, "close_price": price, "total_volume": volume, "total_liquidity": liquidity}
            for token_address, timestamp, price, volume, liquidity in prices
            ]
            stmt = insert(TokenMetricTable).values(update_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["token_address", "timestamp"],
                set_={
                    "close_price": stmt.excluded.close_price,
                    "total_volume": TokenMetricTable.total_volume + stmt.excluded.total_volume,
                    "total_liquidity": TokenMetricTable.total_liquidity + stmt.excluded.total_liquidity,
                },
            )
            session.execute(stmt)
            session.commit()