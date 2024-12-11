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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from typing import Union, List, Dict
from utils.config import get_postgres_url
from utils.utils import has_stablecoin

from datetime import datetime

# Define the base class for your table models
Base = declarative_base()
STABLECOINS = [
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI in Ethereum Mainnet
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC in Ethereum Mainnet
    "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC2
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC3
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT in Ethereum Mainnet
]
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
    price = Column(Float)
    total_liquidity = Column(Float)
    total_volume = Column(Float)


class SwapEventTable(Base):
    __tablename__ = "swap_event"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String, nullable=False)
    pool_address = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
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
    owner = Column(String, nullable=False)
    recipient = Column(String, nullable=False)
    tick_lower = Column(Integer, nullable=False)  # int24 can be stored as Integer
    tick_upper = Column(Integer, nullable=False)  # int24 can be stored as Integer
    amount0 = Column(String, nullable=False)  # U256 can be stored as String
    amount1 = Column(String, nullable=False)  # U256 can be stored as String


class PoolMetricTable(Base):
    __tablename__ = "pool_metrics"
    timestamp = Column(Integer, nullable=False, primary_key=True)
    pool_address = Column(String, nullable=False, primary_key=True)
    price = Column(Float)
    liquidity_token0 = Column(Float)
    liquidity_token1 = Column(Float)
    volume_token0 = Column(Float)
    volume_token1 = Column(Float)


class TokenMetricTable(Base):
    __tablename__ = "token_metrics"
    timestamp = Column(Integer, nullable=False, primary_key=True)
    token_address = Column(String, nullable=False, primary_key=True)

    open_price = Column(Float)
    close_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    total_volume = Column(Float)
    total_liquidity = Column(Float)


class DBManager:
    def __init__(self, url=get_postgres_url()) -> None:
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
                                print(f"Failed to drop table {table_name}: {e}")

                Base.metadata.create_all(self.engine)
        except OperationalError as e:
            print(f"Database connection error: {e}")
            raise

    def compare_schemas(self, engine):
        # Reflect the database schema
        metadata = MetaData()
        metadata.reflect(bind=engine)

        existing_tables = set(metadata.tables.keys())
        model_tables = set(Base.metadata.tables.keys())
        print("compare_schemas start")
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
                    print("TimescaleDB extension created successfully.")
                else:
                    print("TimescaleDB extension already exists.")

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
                        print(f"Hypertable '{table}' created successfully.")
                    else:
                        print(f"Hypertable '{table}' already exists.")
                conn.execute(
                    text(
                        """
                    SELECT create_hypertable(
                        'pool_metrics',
                        'timestamp',
                        'pool_address',
                        if_not_exists => TRUE, 
                        migrate_data => true, 
                        chunk_time_interval => 86400,
                        number_partitions => 6000
                    );
                    """
                    )
                )
                print("Hypertable 'pool_metrics' created successfully.")
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
                print("Hypertable 'token_metrics' created successfully.")

            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")

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
        self, token_pairs: List[Dict[str, Union[str, Integer]]], timestamp: int = None
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
                        **data["event"]["data"],
                    )
                    for data in pool_data
                    if data["event"]["type"] == "collect"
                ]
                if collect_event_data:
                    session.add_all(collect_event_data)

                # Add Pool Metrics
                # last_metrics = session.query(DailyPoolMetricTable).filter(DailyPoolMetricTable.pool_address.in_([metric['pool_address'] for metric in pool_metrics])).all()
                # last_metrics = {metric.pool_address: metric for metric in last_metrics} if last_metrics else {} # Convert to dict for easy access
                batch_size = 10
                for i in range(0, len(pool_metrics), batch_size):
                    batch = pool_metrics[i : i + batch_size]
                    for metric in batch:
                        existing_record = (
                            session.query(PoolMetricTable)
                            .filter_by(
                                timestamp=metric["timestamp"],
                                pool_address=metric["pool_address"],
                            )
                            .first()
                        )

                        if existing_record:
                            # Update existing record
                            existing_record.price = (
                                metric["price"]
                                if metric["price"] != 0.0
                                else existing_record.price
                            )
                            existing_record.liquidity_token0 = (
                                existing_record.liquidity_token0
                                + metric["liquidity_token0"]
                                if metric["liquidity_token0"] != 0.0
                                else existing_record.liquidity_token0
                            )
                            existing_record.liquidity_token1 = (
                                existing_record.liquidity_token1
                                + metric["liquidity_token1"]
                                if metric["liquidity_token1"] != 0.0
                                else existing_record.liquidity_token1
                            )
                            existing_record.volume_token0 = (
                                existing_record.volume_token0 + metric["volume_token0"]
                                if metric["volume_token0"] != 0.0
                                else existing_record.volume_token0
                            )
                            existing_record.volume_token1 = (
                                existing_record.volume_token1 + metric["volume_token1"]
                                if metric["volume_token1"] != 0.0
                                else existing_record.volume_token1
                            )
                        else:
                            # Insert new record
                            new_metric = PoolMetricTable(
                                timestamp=metric["timestamp"],
                                pool_address=metric["pool_address"],
                                price=metric["price"],
                                liquidity_token0=metric["liquidity_token0"],
                                liquidity_token1=metric["liquidity_token1"],
                                volume_token0=metric["volume_token0"],
                                volume_token1=metric["volume_token1"],
                            )
                            session.add(new_metric)

                # Commit the transaction if all operations succeed
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                print(f"An error occurred: {e}")

    def add_token_metrics(self, metrics: List[Dict]) -> None:
        """Add or update token metrics."""
        with self.Session() as session:
            try:
                token_metrics_entries = [
                    TokenMetricTable(
                        timestamp=metric["timestamp"],
                        token_address=metric["token_address"],
                        close_price=metric["close_price"],
                        high_price=metric["high_price"],
                        low_price=metric["low_price"],
                        total_volume=metric["total_volume"],
                        total_liquidity=metric["total_liquidity"],
                    )
                    for metric in metrics
                ]
                if token_metrics_entries:
                    session.add_all(token_metrics_entries)
                    session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"An error occurred: {e}")

    def add_or_update_token_metrics(self, metrics: List[Dict]) -> None:
        """Add or update token metrics based on the is_derived field."""
        with self.Session() as session:
            try:
                for metric in metrics:
                    existing_record = (
                        session.query(TokenMetricTable)
                        .filter_by(
                            timestamp=metric["timestamp"],
                            token_address=metric["token_address"],
                        )
                        .first()
                    )

                    if metric.get("is_derived", False):
                        if not existing_record:
                            new_metric = TokenMetricTable(
                                timestamp=metric["timestamp"],
                                token_address=metric["token_address"],
                                close_price=metric["close_price"],
                                high_price=metric["high_price"],
                                low_price=metric["low_price"],
                                total_volume=metric["total_volume"],
                                total_liquidity=metric["total_liquidity"],
                            )
                            session.add(new_metric)
                        else:
                            # existing_record.close_price = existing_record.close_price + metric['close_price'] / 2
                            # existing_record.high_price = metric['high_price'] if metric['high_price'] > existing_record.high_price else existing_record.high_price
                            # existing_record.low_price = metric['low_price'] if metric['low_price'] < existing_record.low_price else existing_record.low_price
                            existing_record.total_volume += metric["total_volume"]
                            existing_record.total_liquidity += metric["total_liquidity"]
                    else:
                        if existing_record:
                            existing_record.total_volume += metric["total_volume"]
                            existing_record.total_liquidity += metric["total_liquidity"]
                        else:
                            new_metric = TokenMetricTable(
                                timestamp=metric["timestamp"],
                                token_address=metric["token_address"],
                                close_price=metric["close_price"],
                                high_price=metric["high_price"],
                                low_price=metric["low_price"],
                                total_volume=metric["total_volume"],
                                total_liquidity=metric["total_liquidity"],
                            )
                            session.add(new_metric)

                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"An error occurred: {e}")

    def add_or_update_current_pool_metrics(self, metrics: dict) -> None:
        """Add or update daily metrics."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                for pool_address, data in metrics.items():
                    conn.execute(
                        text(
                            f"""
                        INSERT INTO current_pool_metrics (pool_address, price, liquidity_token0, liquidity_token1, volume_token0, volume_token1)
                        VALUES ('{pool_address}', {data['last_price']}, {data['total_liquidity_token0']}, {data['total_liquidity_token1']}, {data['total_volume_token0']}, {data['total_volume_token1']})
                        ON CONFLICT (pool_address) DO UPDATE
                        SET price = CASE WHEN EXCLUDED.price != 0.0 THEN EXCLUDED.price ELSE current_pool_metrics.price END,
                            liquidity_token0 = EXCLUDED.liquidity_token0,
                            liquidity_token1 = EXCLUDED.liquidity_token1,
                            volume_token0 = EXCLUDED.volume_token0,
                            volume_token1 = EXCLUDED.volume_token1;
                        """
                        )
                    )
                    conn.commit()
            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")

    def fetch_current_pool_metrics(
        self, pool_addresses: List[str]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Fetch the latest pool metrics."""
        with self.Session() as session:
            metrics = (
                session.query(CurrentPoolMetricTable)
                .filter(CurrentPoolMetricTable.pool_address.in_(pool_addresses))
                .all()
            )
            return {
                row.pool_address: {
                    "price": row.price,
                    "liquidity_token0": row.liquidity_token0,
                    "liquidity_token1": row.liquidity_token1,
                    "volume_token0": row.volume_token0,
                    "volume_token1": row.volume_token1,
                }
                for row in metrics
            }

    def fetch_token_metrics(
        self, token_address: str, start_timestamp: int, end_timestamp: int
    ) -> List[Dict[str, Union[int, float, str]]]:
        """Fetch token metrics from the corresponding table."""
        with self.Session() as session:
            token_metrics = (
                session.query(TokenMetricTable)
                .filter(
                    TokenMetricTable.token_address == token_address,
                    TokenMetricTable.timestamp >= start_timestamp,
                    TokenMetricTable.timestamp <= end_timestamp,
                )
                .all()
            )
            return [
                {
                    "token_address": row.token_address,
                    "timestamp": row.timestamp,
                    "close_price": row.close_price,
                }
                for row in token_metrics
            ]

    def add_or_update_current_token_metrics(self, metrics: dict) -> None:
        """Add or update daily metrics."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                for token_address, data in metrics.items():
                    conn.execute(
                        text(
                            f"""
                        INSERT INTO current_token_metrics (token_address, price, total_liquidity, total_volume)
                        VALUES ('{token_address}', {data['close_price']}, {data['total_liquidity']}, {data['total_volume']})
                        ON CONFLICT (token_address) DO UPDATE
                        SET price = EXCLUDED.price,
                            total_liquidity = EXCLUDED.total_liquidity,
                            total_volume = EXCLUDED.total_volume;
                        """
                        )
                    )
                    conn.commit()
            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")

    def fetch_current_token_metrics(
        self, token_addresses: List[str]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Fetch the latest token metrics."""
        with self.Session() as session:
            metrics = (
                session.query(CurrentTokenMetricTable)
                .filter(CurrentTokenMetricTable.token_address.in_(token_addresses))
                .all()
            )
            return {
                row.token_address: {
                    "close_price": row.price,
                    "total_liquidity": row.total_liquidity,
                    "total_volume": row.total_volume,
                }
                for row in metrics
            }
