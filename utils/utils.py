def hex_to_signed_int(hex_str):
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    
    bit_length = len(hex_str) * 4
    
    value = int(hex_str, 16)
    
    # Check if the value is negative
    if value >= 2**(bit_length - 1):
        value -= 2**bit_length
    
    return value

def tick_to_sqrt_price(tick):
    """Convert a tick to the square root price"""
    return 1.0001 ** (tick / 2)