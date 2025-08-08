import numpy as np

def detect_outliers(df, asset_name):
    # Drop NaN returns
    returns = df['Daily Return'].dropna()
    
    # Define thresholds
    lower_threshold = returns.quantile(0.025)
    upper_threshold = returns.quantile(0.975)
    
    # Filter extreme days
    outliers = df[(df['Daily Return'] <= lower_threshold) | 
                  (df['Daily Return'] >= upper_threshold)]
    
    print(f"\nOutliers for {asset_name}:")
    print(outliers[['Daily Return']].sort_values('Daily Return'))
    return outliers
