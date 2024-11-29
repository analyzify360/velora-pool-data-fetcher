def hex_to_signed_int(hex_str: str) -> int:
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    
    bit_length = len(hex_str) * 4
    
    value = int(hex_str, 16)
    
    # Check if the value is negative
    if value >= 2**(bit_length - 1):
        value -= 2**bit_length
    
    return value

def tick_to_sqrt_price(tick: int) -> float:
    """Convert a tick to the square root price"""
    return 1.0001 ** (tick / 2)

def calc_prices_token0_by_token1(sqrt_prices: list[str], token0_decimals: int, token1_decimals: int) -> list[float]:
    """Calculate the price of token0 in token1"""
    prices = []
    for sqrt_price in sqrt_prices:
        if sqrt_price.startswith("0x"):
            price = (float(hex_to_signed_int(sqrt_price)) / 2 ** 96) ** 2 * 10 ** (token0_decimals - token1_decimals)
        else:
            price = (float(sqrt_price) / 2 ** 96) ** 2 * 10 ** (token0_decimals - token1_decimals)
        prices.append(price)
    return prices

def calc_prices_token1_by_token0(sqrt_prices: list[str], token0_decimals: int, token1_decimals: int) -> list[float]:
    """Calculate the price of token1 in token0"""
    prices = []
    for sqrt_price in sqrt_prices:
        if sqrt_price.startswith("0x"):
            price = (float(hex_to_signed_int(sqrt_price)) / 2 ** 96) ** 2 * 10 ** (token0_decimals - token1_decimals)
        else:
            price = (float(sqrt_price) / 2 ** 96) ** 2 * 10 ** (token0_decimals - token1_decimals)
        price = 1 / price
        prices.append(price)
    return prices

if __name__ == "__main__":
    print(calc_prices_token0_by_token1(["0x3ea65739993c6d86c200d"], 6, 6))
    print(calc_prices_token1_by_token0(["0x3ea65739993c6d86c200d"], 6, 6))