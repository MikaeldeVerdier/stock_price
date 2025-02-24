from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange
import eth_account
from eth_account.signers.local import LocalAccount

info = Info(constants.MAINNET_API_URL, skip_ws=True)
public_key = "" #enter public key
private_key = " " # enter private key
user_state = info.user_state(public_key)
print(user_state)
account: LocalAccount = eth_account.Account.from_key(private_key)
exchange = Exchange(account, constants.MAINNET_API_URL)

def buy(symbol: str, size: float):
    exchange.market_open(coin=symbol, is_buy=True, sz=size)

def close_position(symbol: str):
    exchange.market_close(coin=symbol)
