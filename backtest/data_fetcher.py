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

def _fetch_ohlcv_and_funding(exchange, symbol, timeframe, start_ts, end_ts, limit=1000):
    """Helper to fetch OHLCV and funding for a specific timestamp range and merge them."""
    print(f"📥 Fetching {symbol} from {start_ts} to {end_ts}...")
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
            
    if not all_ohlcv:
        return pd.DataFrame()
        
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
            df_ohlcv = df_ohlcv.sort_values('timestamp')
            df = pd.merge_asof(df_ohlcv, df_funding, on='timestamp', direction='backward')
            
            # Fill any NaN at the very beginning with the first known funding rate, or 0.0001
            df['fundingRate'] = df['fundingRate'].bfill().fillna(0.0001)
        else:
            df = df_ohlcv
            df['fundingRate'] = 0.0001
    else:
        df = df_ohlcv
        df['fundingRate'] = 0.0001
        
    return df

def fetch_data(symbol='BTC/USDT', timeframe='15m', start_date='2023-01-01', end_date=None, limit=1000, exchange_id='binance'):
    """
    Fetch OHLCV data from CCXT with local Parquet caching.
    Merges data into a single file per symbol/timeframe to avoid downloading existing data.
    """
    safe_symbol = symbol.replace('/', '')
    cache_dir = os.path.join('data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else int(time.time() * 1000)
    
    filename = f"{safe_symbol}_{timeframe}.parquet"
    filepath = os.path.join(cache_dir, filename)
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    df_existing = pd.DataFrame()
    
    # Load existing cache if available
    if os.path.exists(filepath):
        print(f"📦 Loading cached data for {symbol} ({filename})...")
        df_existing = pd.read_parquet(filepath)
        df_existing = df_existing.sort_values('timestamp').drop_duplicates('timestamp')
        
    if not df_existing.empty:
        cache_start_ts = int(df_existing['timestamp'].iloc[0])
        cache_end_ts = int(df_existing['timestamp'].iloc[-1])
        
        dfs_to_concat = [df_existing]
        needs_save = False
        
        # Check if we need data before the cache starts
        if start_ts < cache_start_ts:
            print(f"🔄 Need earlier data (from {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(cache_start_ts/1000)})")
            df_before = _fetch_ohlcv_and_funding(exchange, symbol, timeframe, start_ts, cache_start_ts, limit)
            if not df_before.empty:
                dfs_to_concat.insert(0, df_before)
                needs_save = True
                
        # Check if we need data after the cache ends
        if end_ts > cache_end_ts:
            print(f"🔄 Need newer data (from {datetime.fromtimestamp(cache_end_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)})")
            df_after = _fetch_ohlcv_and_funding(exchange, symbol, timeframe, cache_end_ts + 1, end_ts, limit)
            if not df_after.empty:
                dfs_to_concat.append(df_after)
                needs_save = True
                
        if needs_save:
            df = pd.concat(dfs_to_concat, ignore_index=True)
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            df.to_parquet(filepath)
            print(f"\n✅ Merged and saved updated data to {filepath}")
        else:
            df = df_existing
            print(f"✅ Requested data range is fully within existing cache.")
    else:
        # No cache exists, fetch everything requested
        df = _fetch_ohlcv_and_funding(exchange, symbol, timeframe, start_ts, end_ts, limit)
        if not df.empty:
            df.to_parquet(filepath)
            print(f"\n✅ Data saved to {filepath}")
        
    # Finally, return only the requested slice
    if not df.empty:
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)
        
    return df
