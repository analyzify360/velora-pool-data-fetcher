from sqlalchemy import create_engine, Column, Boolean, MetaData, Table, String, Integer, Float, Numeric, inspect, insert, text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import Union, List, Dict
from utils.config import get_postgres_url

from datetime import datetime

# Define the base class for your table models
Base = declarative_base()

# Define the timetable table
class Timetable(Base):
    __tablename__ = 'timetable'
    start = Column(Integer, primary_key=True)  # Assuming 'start' is a unique field, hence primary key
    end = Column(Integer)
    completed = Column(Boolean)

class TokenPairTable(Base):
    __tablename__ = 'token_pairs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    token0 = Column(String, nullable=False)
    token1 = Column(String, nullable=False)
    fee = Column(Integer, nullable=False)
    pool = Column(String, nullable=False)
    block_number = Column(Integer, nullable=False)
    completed = Column(Boolean, nullable=False)
    
class TokenTable(Base):
    __tablename__ = 'tokens'
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    decimals = Column(Integer, nullable=False)

class PoolTable(Base):
    __tablename__ = 'pools'
    pool_address = Column(String, primary_key=True)
    liquidity_24h = Column(Numeric)
    volume_24h = Column(Numeric)
    price_range_24h = Column(String)

class SwapEventTable(Base):
    __tablename__ = 'swap_event'
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
    __tablename__ = 'mint_event'
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
    __tablename__ = 'burn_event'
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
    __tablename__ = 'collect_event'
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

class UniswapSignalsTable(Base):
    __tablename__ = 'uniswap_signals'
    timestamp = Column(Integer, nullable=False, primary_key=True)
    pool_address = Column(String, nullable=False, primary_key=True)
    price = Column(Float)
    liquidity = Column(Numeric)
    volume = Column(Numeric)

class DBManager:

    def __init__(self, url = get_postgres_url()) -> None:
        # Create the SQLAlchemy engine
        self.engine = create_engine(url)

        # Create a configured "Session" class
        self.Session = sessionmaker(bind=self.engine)

        # Create the table if it doesn't exist
        # Base.metadata.create_all(self.engine)  # This line ensures the table is created if not exists
        self.check_and_create_tables()

        # Enable TimescaleDB and convert specific tables to hypertables
        self.create_hypertables()
        
    def check_and_create_tables(self):
        # Reflect the database schema
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        existing_tables = set(metadata.tables.keys())
        model_tables = set(Base.metadata.tables.keys())
        print("compare_schemas start")
        # Compare table names
        if not model_tables <= existing_tables:  
            return False
        inspector = inspect(self.engine)
        tables = [
            Timetable,
            TokenPairTable,
            TokenTable,
            PoolTable,
            SwapEventTable,
            MintEventTable,
            BurnEventTable,
            CollectEventTable,
            UniswapSignalsTable
        ]

        for table in tables:
            table_name = table.__tablename__
            
            if inspector.has_table(table_name):
                # Get current table columns
                existing_columns = inspector.get_columns(table_name)
                existing_column_names = {col['name']: col for col in existing_columns}

                # Define expected columns based on current schema
                current_columns = {col.name: col for col in table.__table__.columns}

                # Compare existing schema with current schema
                schema_mismatch = False
                for column_name, column in current_columns.items():
                    if column_name not in existing_column_names:
                        schema_mismatch = True
                        print(f"Column {column_name} does not exist in {table_name}.")
                        break
                    existing = existing_column_names[column_name]
                    if (existing['type'].__class__.__name__ != column.type.__class__.__name__ or
                        existing['nullable'] != column.nullable):
                        schema_mismatch = True
                        print(f"Schema mismatch detected for {column_name} in {table_name}.")
                        break
                
                if schema_mismatch:
                    print(f"Dropping and recreating table {table_name}.")
                    Base.metadata.drop_all(self.engine, [table.__table__])
                    Base.metadata.create_all(self.engine, [table.__table__])
                else:
                    print(f"Table {table_name} exists with the same schema. No action needed.")
            else:
                print(f"Table {table_name} does not exist. Creating it.")
                Base.metadata.create_all(self.engine, [table.__table__])
        print('all tables created successfully')

        for table_name in existing_tables.intersection(model_tables):
            existing_columns = set(c['name'] for c in inspector.get_columns(table_name))
            model_columns = set(c.name for c in Base.metadata.tables[table_name].columns)

            # Compare columns
            if existing_columns != model_columns:
                return False

            # Add more detailed comparison logic if needed
            existing_constraints = {c['name']: c for c in inspector.get_unique_constraints(table_name)}
            model_constraints = {c.name: c for c in Base.metadata.tables[table_name].constraints if isinstance(c, UniqueConstraint)}

            if set(existing_constraints.keys()) != set(model_constraints.keys()):
                return False

            for name in existing_constraints.keys():
                if existing_constraints[name]['column_names'] != list(model_constraints[name].columns.keys()):
                    return False

        return True

    def create_hypertables(self):
        """Enable TimescaleDB extension and convert tables to hypertables."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                # Check if TimescaleDB extension is already installed
                result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb';"))
                if not result.fetchone():
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                    print("TimescaleDB extension created successfully.")
                else:
                    print("TimescaleDB extension already exists.")

                # Check if hypertable is enabled
                result = conn.execute(text(
                    "SELECT * FROM timescaledb_information.hypertables;"
                )).fetchall()
                hypertables = [entry.hypertable_name for entry in result]
                
                tables = ['token_pairs', 'tokens', 'swap_event', 'mint_event', 'burn_event', 'collect_event']
                for table in tables:
                    if table not in hypertables:
                        conn.execute(text(
                            f"SELECT create_hypertable('{table}', 'id', if_not_exists => TRUE, migrate_data => true);"
                        ))
                        print(f"Hypertable '{table}' created successfully.")
                    else:
                        print(f"Hypertable '{table}' already exists.")
                conn.execute(text(
                    f"""
                    SELECT create_hypertable(
                        'uniswap_signals',
                        'timestamp',
                        'pool_address',
                        if_not_exists => TRUE, 
                        migrate_data => true, 
                        chunk_time_interval => 86400,
                        number_partitions => 6000
                    );
                    """
                ))
                print("Hypertable 'uniswap_signals' created successfully.")
                conn.execute(text(
                    f"""
                    CREATE OR REPLACE FUNCTION fill_missing_values()
                    RETURNS TRIGGER AS $$
                    DECLARE
                        last_price RECORD;
                        last_liquidity RECORD;
                        last_volume RECORD;
                    BEGIN
                        -- Check and retrieve the last known price for this pool_address if NEW.price is NULL
                        IF NEW.price = 0.0 THEN
                            SELECT price
                            INTO last_price
                            FROM uniswap_signals
                            WHERE pool_address = NEW.pool_address
                            AND price != 0.0
                            ORDER BY timestamp DESC
                            LIMIT 1;

                            -- Substitute 0 price with the last known price, if available
                            IF last_price.price != 0.0 THEN
                                NEW.price := last_price.price;
                            END IF;
                        END IF;

                        -- Check and retrieve the last known liquidity for this pool_address if NEW.liquidity is 0
                        IF NEW.liquidity = 0 THEN
                            SELECT liquidity
                            INTO last_liquidity
                            FROM uniswap_signals
                            WHERE pool_address = NEW.pool_address
                            AND liquidity != 0
                            ORDER BY timestamp DESC
                            LIMIT 1;

                            -- Substitute 0 liquidity with the last known liquidity, if available
                            IF last_liquidity.liquidity != 0 THEN
                                NEW.liquidity := last_liquidity.liquidity;
                            END IF;
                        END IF;

                        -- Check and retrieve the last known volume for this pool_address if NEW.volume is 0
                        IF NEW.volume = 0 THEN
                            SELECT volume
                            INTO last_volume
                            FROM uniswap_signals
                            WHERE pool_address = NEW.pool_address
                            AND volume != 0
                            ORDER BY timestamp DESC
                            LIMIT 1;

                            -- Substitute 0 volume with the last known volume, if available
                            IF last_volume.volume != 0 THEN
                                NEW.volume := last_volume.volume;
                            END IF;
                        END IF;

                        -- Return the potentially modified NEW record
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;

                    """
                ))
                print("Function 'fill_missing_values' created successfully.")
                conn.execute(text(
                    f"""
                    DO $$
                    BEGIN
                        -- Check if the trigger already exists
                        IF NOT EXISTS (
                            SELECT 1 
                            FROM pg_trigger 
                            WHERE tgname = 'fill_missing_values_trigger'
                            AND tgrelid = 'uniswap_signals'::regclass
                        ) THEN
                            -- Only create the trigger if it doesn't exist
                            CREATE TRIGGER fill_missing_values_trigger
                            BEFORE INSERT ON uniswap_signals
                            FOR EACH ROW
                            EXECUTE FUNCTION fill_missing_values();
                        END IF;
                    END $$;
                    """
                ))

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
            return [{"start": row.start, "end": row.end, "completed": row.completed} for row in timetable_data]

    def fetch_incompleted_time_range(self) -> List[Dict[str, Union[Integer, bool]]]:
        """Fetch all not completed time ranges from the timetable."""
        with self.Session() as session:
            not_completed_data = session.query(Timetable).filter_by(completed=False).all()
            return [{"start": row.start, "end": row.end, "completed": row.completed} for row in not_completed_data]
    
    def fetch_last_time_range(self) -> Dict[str, Union[datetime, bool]]:
        """Fetch the last time range from the timetable."""
        with self.Session() as session:
            last_time_range = session.query(Timetable).order_by(Timetable.start.desc()).first()
            if last_time_range is not None:
                return {"start": last_time_range.start, 
                        "end": last_time_range.end, 
                        "completed": last_time_range.completed}
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
                exists = session.query(TokenTable).filter_by(address=token['address']).first()
                if not exists:
                    new_token = TokenTable(
                        address=token['address'],
                        symbol=token['symbol'],
                        name=token['name'],
                        decimals=token['decimals']
                    )
                    session.add(new_token)
            session.commit()

    def add_token_pairs(self, token_pairs: List[Dict[str, Union[str, Integer]]]) -> None:
        """Add token pairs to the corresponding table."""
        
        insert_values = [
            TokenPairTable(
                token0 = token_pair['token0']["address"],
                token1 = token_pair['token1']["address"],
                fee = token_pair['fee'],
                pool = token_pair['pool_address'],
                block_number = token_pair['block_number'],
                completed = False)
            for token_pair in token_pairs
        ]
        
        self.add_tokens([token for token_pair in token_pairs for token in [token_pair['token0'], token_pair['token1']]])
        
        with self.Session() as session:
            session.add_all(insert_values)
            session.commit()
    
    def fetch_token_pairs(self):
        """Fetch all token pairs from the corresponding table."""
        with self.Session() as session:
            token_pairs = session.query(TokenPairTable).all()
            return [{"token0": row.token0, "token1": row.token1, "fee": row.fee, "completed": row.completed} for row in token_pairs]

    def fetch_incompleted_token_pairs(self) -> List[Dict[str, Union[str, Integer, bool]]]:
        """Fetch all incompleted token pairs from the corresponding table."""
        with self.Session() as session:
            incompleted_token_pairs = session.query(TokenPairTable).filter_by(completed=False).all()
            return [{"token0": row.token0, "token1": row.token1, "fee": row.fee, "pool_address": row.pool, "completed": row.completed} for row in incompleted_token_pairs]

    def mark_token_pairs_as_complete(self, token_pairs: List[tuple]) -> bool:
        """Mark a token pair as complete."""
        with self.Session() as session:
            for token_pair in token_pairs:
                record = session.query(TokenPairTable).filter_by(token0=token_pair[0], token1=token_pair[1], fee=token_pair[2]).first()
                if record:
                    session.query(TokenPairTable).filter_by(token0=token_pair[0], token1=token_pair[1], fee=token_pair[2]).update({TokenPairTable.completed: True})
                else:
                    return False
            session.commit()
            return True
    def reset_token_pairs(self):
        """Reset the token pairs completed state"""
        with self.Session() as session:
            session.query(TokenPairTable).update({TokenPairTable.completed: False})
            session.commit()

    def add_pool_data(self, pool_data: List[Dict]) -> None:
        """Add pool data to the pool data table and related event tables."""

        # Add the swap event data to the swap event table
        swap_event_data = [
            SwapEventTable(transaction_hash=data['transaction_hash'], pool_address = data['pool_address'], block_number=data['block_number'], **data['event']['data'])
            for data in pool_data if data['event']['type'] == 'swap'
        ]
        if swap_event_data:
            with self.Session() as session:
                session.add_all(swap_event_data)
                session.commit()

        # Add the mint event data to the mint event table
        mint_event_data = [
            MintEventTable(transaction_hash=data['transaction_hash'], pool_address = data['pool_address'], block_number=data['block_number'], **data['event']['data'])
            for data in pool_data if data['event']['type'] == 'mint'
        ]
        if mint_event_data:
            with self.Session() as session:
                session.add_all(mint_event_data)
                session.commit()

        # Add the burn event data to the burn event table
        burn_event_data = [
            BurnEventTable(transaction_hash=data['transaction_hash'], pool_address = data['pool_address'], block_number=data['block_number'], **data['event']['data'])
            for data in pool_data if data['event']['type'] == 'burn'
        ]
        if burn_event_data:
            with self.Session() as session:
                session.add_all(burn_event_data)
                session.commit()

        # Add the collect event data to the collect event table
        collect_event_data = [
            CollectEventTable(transaction_hash=data['transaction_hash'], pool_address = data['pool_address'], block_number=data['block_number'], **data['event']['data'])
            for data in pool_data if data['event']['type'] == 'collect'
        ]
        if collect_event_data:
            with self.Session() as session:
                session.add_all(collect_event_data)
                session.commit()
    
    def add_uniswap_signals(self, signals: List[Dict]) -> None:
        """Add Uniswap signals to the corresponding table."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                values = []
                for signal in signals:
                    price = 0.0 if signal['price'] is None else signal['price']
                    liquidity = 0 if signal['liquidity'] is None else signal['liquidity']
                    volume = 0 if signal['volume'] is None else signal['volume']
                    values.append({
                        'timestamp': signal['timestamp'],
                        'pool_address': signal['pool_address'],
                        'price': price,
                        'liquidity': liquidity,
                        'volume': volume
                    })
                
                # Use bulk insert
                conn.execute(
                    text("""
                        INSERT INTO uniswap_signals (timestamp, pool_address, price, liquidity, volume)
                        VALUES (:timestamp, :pool_address, :price, :liquidity, :volume)
                    """), values
                )
                conn.commit()
            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")
    
    def add_or_update_daily_metrics(self, metrics: dict) -> None:
        """Add or update daily metrics."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                for pool_address, data in metrics.items():
                    conn.execute(text(
                        f"""
                        INSERT INTO pools (pool_address, liquidity_24h, volume_24h, price_range_24h)
                        VALUES ('{pool_address}', {data['liquidity']}, {data['volume']}, '{data['price_low']}-{data['price_high']}')
                        ON CONFLICT (pool_address) DO UPDATE
                        SET price_range_24h = EXCLUDED.price_range_24h,
                            liquidity_24h = EXCLUDED.liquidity_24h,
                            volume_24h = EXCLUDED.volume_24h;
                        """
                    ))
                    conn.commit()
            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")
