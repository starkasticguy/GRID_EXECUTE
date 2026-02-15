import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_data(symbol='BTC/USDT', timeframe='15m', start_date='2023-01-01', end_date=None, limit=1000, exchange_id='binance'):
    """
    Fetch OHLCV data from CCXT with local Parquet caching.
    """
    safe_symbol = symbol.replace('/', '')
    cache_dir = os.path.join('data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else int(time.time() * 1000)
    
    filename = f"{safe_symbol}_{timeframe}_{start_ts}_{end_ts}.parquet"
    filepath = os.path.join(cache_dir, filename)
    
    # Check cache
    if os.path.exists(filepath):
        print(f"📦 Loading cached data for {symbol}...")
        return pd.read_parquet(filepath)
    
    print(f"📥 Fetching {symbol} from {exchange_id}...")
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    all_ohlcv = []
    since = start_ts
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts == since:
                break
            since = last_ts + 1
            
            # Progress
            dt = datetime.fromtimestamp(last_ts / 1000)
            print(f"   Fetched until {dt}", end='\r')
            
        except Exception as e:
            print(f"Error fetching: {e}")
            time.sleep(5)
            continue
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df[df['timestamp'] <= end_ts] # Trim content after end date
    
    # Save cache
    df.to_parquet(filepath)
    print(f"\n✅ Data saved to {filepath}")
    
    return df
