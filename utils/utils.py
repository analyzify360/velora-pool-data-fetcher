from datetime import datetime, timezone

STABLECOINS = [
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI in Ethereum Mainnet
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC in Ethereum Mainnet
    "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC2
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC3
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT in Ethereum Mainnet
]

def signed_hex_to_int(hex_str: str) -> int:
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    bit_length = len(hex_str) * 4

    value = int(hex_str, 16)

    # Check if the value is negative
    if value >= 2 ** (bit_length - 1):
        value -= 2**bit_length

    return value


def unsigned_hex_to_int(hex_str: str) -> int:
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    return int(hex_str, 16)


def tick_to_sqrt_price(tick: int) -> float:
    """Convert a tick to the square root price"""
    return 1.0001 ** (tick / 2)

def calc_price(token0_amount: int, token1_amount: int, token0_decimals: int, token1_decimals: int) -> float:
    """Calculate the price of token0 in token1"""
    return float(token1_amount) / float(token0_amount) * 10 ** (token0_decimals - token1_decimals)
def calc_price_sqrt_base(
    sqrt_price: str, token0_decimals: int, token1_decimals: int
) -> list[float]:
    """Calculate the price of token0 in token1"""
    if sqrt_price.startswith("0x"):
        price = (float(unsigned_hex_to_int(sqrt_price)) / 2**96) ** 2 * 10 ** (
            token0_decimals - token1_decimals
        )
    else:
        price = (float(sqrt_price) / 2**96) ** 2 * 10 ** (
            token0_decimals - token1_decimals
        )
    return price

def calc_prices_token0_by_token1(
    sqrt_prices: list[str], token0_decimals: int, token1_decimals: int
) -> list[float]:
    """Calculate the price of token0 in token1"""
    prices = []
    for sqrt_price in sqrt_prices:
        if sqrt_price.startswith("0x"):
            price = (float(unsigned_hex_to_int(sqrt_price)) / 2**96) ** 2 * 10 ** (
                token0_decimals - token1_decimals
            )
        else:
            price = (float(sqrt_price) / 2**96) ** 2 * 10 ** (
                token0_decimals - token1_decimals
            )
        prices.append(price)
    return prices


def calc_prices_token1_by_token0(
    sqrt_prices: list[str], token0_decimals: int, token1_decimals: int
) -> list[float]:
    """Calculate the price of token1 in token0"""
    prices = []
    for sqrt_price in sqrt_prices:
        if sqrt_price.startswith("0x"):
            price = (float(unsigned_hex_to_int(sqrt_price)) / 2**96) ** 2 * 10 ** (
                token0_decimals - token1_decimals
            )
        else:
            price = (float(sqrt_price) / 2**96) ** 2 * 10 ** (
                token0_decimals - token1_decimals
            )
        price = 1 / price
        prices.append(price)
    return prices
def has_stablecoin(token_pair: dict, stablecoins: list = STABLECOINS) -> bool:
    """Check if the pool has a stablecoin"""
    return token_pair["token0"]["address"] in stablecoins or token_pair["token1"]["address"] in stablecoins

def apply_abs_to_list(values: list[int]) -> list[int]:
    """Apply the abs function to a list of values"""
    return [abs(value) for value in values]

def normalize_with_deciamls(value: int, token_decimals: int) -> float:
    """Calculate the value with removing the decimals"""
    return float(value) / 10**token_decimals

def timestamp_to_date(timestamp: int, date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert a timestamp to a date"""
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime(date_format)

if __name__ == "__main__":
    print((signed_hex_to_int("0xb1a2bc2ec50000")) / signed_hex_to_int("0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffff64c3368") * 10 ** (6 - 18))
    print((unsigned_hex_to_int("0xfffd8963efd1fc6a506488495d951d5263988d25") / 2**96) ** 2 * 10 ** -18)
    print(calc_price_sqrt_base("0xfffd8963efd1fc6a506488495d951d5263988d25", 18, 18))
    print(calc_prices_token1_by_token0(["0x3ea65739993c6d86c200d"], 6, 6))
    print(unsigned_hex_to_int("0xfffd8963efd1fc6a506488495d951d5263988d25"))