import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from core.kama import REGIME_NOISE, REGIME_UPTREND, REGIME_DOWNTREND

def extract_ml_features(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, window: int = 24) -> np.ndarray:
    """
    Extract statistical features for unsupervised clustering.
    We use rolling return (direction/velocity) and normalized true range (volatility).
    """
    n = len(close)
    features = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(1, n):
        # Feature 1: Returns (smoothed)
        if i >= window:
            ret = (close[i] - close[i - window]) / close[i - window]
        else:
            ret = (close[i] - close[0]) / close[0]
            
        # Feature 2: Volatility (True Range / Close)
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        vol = tr / close[i]
        
        features[i, 0] = ret
        features[i, 1] = vol
        
    return features

def classify_regimes_gmm(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, 
                         window: int = 24, n_components: int = 3, random_state: int = 42) -> np.ndarray:
    """
    Fit a Gaussian Mixture Model to classify the market into discrete states.
    Uses a ROLLING TRAINING WINDOW to prevent lookahead bias.
    
    Extracts features (returns, volatility) over 'window' bars (e.g., 24 bars = 6 hours).
    Fits the GMM model on the previous 30 days of data (2880 bars) to define the clusters.
    Predicts the current bar's regime step-by-step.
    """
    n = len(close)
    mapped_states = np.full(n, REGIME_NOISE, dtype=np.int32)
    
    if n < window * 2:
        return mapped_states
        
    features = extract_ml_features(close, high, low, volume, window)
    
    # 30 days of 15m bars = 2880 bars. 
    # Must wait until we have at least 'train_window_bars' before we start predicting.
    train_window_bars = 2880  
    
    # If the dataset is smaller than 30 days, we'll just train on whatever we have 
    # (min 500 bars to get enough statistical density for 3 clusters)
    actual_train_window = min(train_window_bars, n // 2)
    if actual_train_window < 500:
        actual_train_window = 500
        
    if n <= actual_train_window:
        return mapped_states

    # Step-forward rolling window (to optimize speed, we re-fit the GMM once a day = every 96 bars)
    refit_interval = 96 
    
    gmm = None
    down_state, noise_state, up_state = 0, 1, 2 # Fallback
    
    for i in range(actual_train_window, n):
        # Every 'refit_interval' bars, we retrain the GMM on the immediately preceding window
        if i % refit_interval == 0 or gmm is None:
            train_start = max(window, i - actual_train_window)
            train_end = i
            
            # Extract the training slice features
            train_features = features[train_start:train_end]
            
            # Re-fit the model
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
            try:
                gmm.fit(train_features)
                
                # Re-map the states for the new model
                cluster_means = gmm.means_
                returns_by_state = {k: cluster_means[k][0] for k in range(n_components)}
                sorted_states = sorted(returns_by_state.keys(), key=lambda k: returns_by_state[k])
                
                down_state = sorted_states[0]    # Most negative return
                noise_state = sorted_states[1]   # Middle return (closest to 0)
                up_state = sorted_states[2]      # Most positive return
            except ValueError:
                # If fitting fails (e.g., zero variance in a test subset), keep the old GMM map
                pass

        # Predict the exact current bar using the live GMM model
        if gmm is not None and hasattr(gmm, 'means_'):
            current_feature = features[i].reshape(1, -1)
            s = gmm.predict(current_feature)[0]
            
            if s == down_state:
                mapped_states[i] = REGIME_DOWNTREND
            elif s == up_state:
                mapped_states[i] = REGIME_UPTREND
            else:
                mapped_states[i] = REGIME_NOISE

    return mapped_states
