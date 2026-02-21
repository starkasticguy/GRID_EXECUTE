import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_historical_funding(exchange, symbol: str, start_ts: int, end_ts: int, limit: int = 1000) -> list:
    """Fetch all history of funding rates from ccxt."""
    print(f"📥 Fetching funding rates for {symbol}...")
    all_funding = []
    since = start_ts
    
    # Check if exchange supports it
    if not exchange.has.get('fetchFundingRateHistory'):
        print("⚠️ Exchange does not support fetchFundingRateHistory, returning empty funding data.")
        return []

    while since < end_ts:
        try:
            funding = exchange.fetch_funding_rate_history(symbol, since, limit)
            if not funding:
                break
            
            all_funding.extend(funding)
            # ccxt fetch_funding_rate_history returns dicts with 'timestamp'
            last_ts = funding[-1]['timestamp']
            
            if last_ts == since:
                # Prevent infinite loop if API returns same data
                since = last_ts + 1
            else:
                since = last_ts + 1
                
        except Exception as e:
            print(f"Error fetching funding: {e}")
            time.sleep(5)
            continue
            
    return all_funding

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
            
    df_ohlcv = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_ohlcv = df_ohlcv[df_ohlcv['timestamp'] <= end_ts] # Trim content after end date
    
    # Format and merge Funding data if available
    funding_data = fetch_historical_funding(exchange, symbol, start_ts, end_ts, limit)
    
    if funding_data:
        # Convert to DF and extract the raw float fundingRate
        df_funding = pd.DataFrame(funding_data)
        # Keep only timestamp and fundingRate
        if 'fundingRate' in df_funding.columns and 'timestamp' in df_funding.columns:
            df_funding = df_funding[['timestamp', 'fundingRate']].copy()
            df_funding['timestamp'] = pd.to_numeric(df_funding['timestamp'])
            df_funding['fundingRate'] = pd.to_numeric(df_funding['fundingRate'])
            df_funding = df_funding.sort_values('timestamp').drop_duplicates('timestamp')
            
            # Use merge_asof to forward-fill funding rates into the 15m OHLCV timeline
            # 'backward' direction means for a 08:15 candle, find the closest funding timestamp <= 08:15.
            df_ohlcv = df_ohlcv.sort_values('timestamp')
            df = pd.merge_asof(df_ohlcv, df_funding, on='timestamp', direction='backward')
            
            # Fill any NaN at the very beginning with the first known funding rate, or 0
            # A typical funding rate is 0.01% (0.0001) per 8h
            df['fundingRate'] = df['fundingRate'].bfill().fillna(0.0001)
        else:
            df = df_ohlcv
            df['fundingRate'] = 0.0001
    else:
        df = df_ohlcv
        df['fundingRate'] = 0.0001
        
    # Save cache
    df.to_parquet(filepath)
    print(f"\n✅ Data saved to {filepath}")
    
    return df
