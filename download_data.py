import yfinance as yf
import pandas as pd
import os
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define a diverse set of tickers
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC",
    "WFC", "GS", "MS", "JNJ", "PFE", "UNH", "MRK", "ABBV", "WMT", "COST",
    "PG", "NKE", "MCD", "XOM", "CVX", "CAT", "BA", "GE"
]

START_DATE = "2020-01-01"
END_DATE = pd.Timestamp('today').strftime('%Y-%m-%d')

DATA_DIR = "stock_data_processed"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTION TO ENSURE 1D NUMPY ARRAY
# ============================================================================
def ensure_1d_numpy_array(data_input, context_msg=""):
    """
    Ensures the input is a 1D NumPy array.
    Handles Python lists, scalars, Pandas Series values, single-column DataFrame values,
    and NumPy arrays (0D, 1D, or 2D column vectors).
    """
    if isinstance(data_input, pd.Series):
        data_values = data_input.values
    elif isinstance(data_input, pd.DataFrame):
        if data_input.shape[1] == 1: # Single column DataFrame
            data_values = data_input.iloc[:, 0].values
        else:
            raise ValueError(f"Input DataFrame for {context_msg} has more than one column: shape {data_input.shape}")
    elif not isinstance(data_input, np.ndarray):
        try:
            # Attempt to convert common list-like or scalar types
            data_values = np.array(data_input)
        except Exception as e:
            raise ValueError(f"Cannot convert input for {context_msg} to NumPy array: {e}")
    else: # It's already a NumPy array
        data_values = data_input

    # Now data_values is a NumPy array, ensure it's 1D
    if data_values.ndim == 2 and data_values.shape[1] == 1: # Column vector (N,1)
        return data_values.reshape(-1)
    elif data_values.ndim == 1: # Already 1D (N,)
        return data_values
    elif data_values.ndim == 0: # Scalar numpy array (e.g. from np.array(5) or .item())
        return np.array([data_values.item()]) # Convert to 1-element 1D array
    else: # Other shapes
        raise ValueError(f"Data for {context_msg} is not convertible to a 1D array, current shape: {data_values.shape}")


print(f"Downloading and processing data for {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")

for ticker in TICKERS:
    try:
        print(f"Processing {ticker}...")
        # 1. Download data
        data_raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

        if data_raw.empty:
            print(f"  No data found for {ticker} (yf.download). Skipping.")
            continue

        # 2. Store original 'Close' correctly as a Series
        if 'Close' not in data_raw.columns:
            print(f"  'Close' column not found in raw data for {ticker}. Skipping.")
            continue
        
        close_column_data = data_raw['Close']
        if isinstance(close_column_data, pd.DataFrame):
            if close_column_data.shape[1] == 1:
                close_column_data = close_column_data.squeeze()
            else:
                print(f"  Error: 'Close' data for {ticker} is a multi-column DataFrame ({close_column_data.shape}). Skipping.")
                continue
        
        original_close_unaligned = pd.Series(close_column_data, 
                                             index=data_raw.index, 
                                             name='Original_Close_Unaligned').copy()

        # 3. Add Technical Indicators
        data_for_ta = data_raw.copy()
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        can_add_ta = True

        if not all(col in data_for_ta.columns for col in ohlcv_cols):
            print(f"  Warning: Missing one or more required OHLCV columns for {ticker} for TA. Skipping TA addition.")
            can_add_ta = False
        else:
            for col_name in ohlcv_cols:
                try:
                    column_data = data_for_ta[col_name]
                    if isinstance(column_data, pd.DataFrame):
                        if column_data.shape[1] == 1:
                            data_for_ta[col_name] = column_data.squeeze()
                        else:
                            print(f"  Error: TA input column '{col_name}' for {ticker} is multi-column DataFrame. Skipping TA.")
                            can_add_ta = False; break
                    elif not isinstance(column_data, pd.Series):
                        print(f"  Error: TA input column '{col_name}' for {ticker} is not Series/DataFrame. Skipping TA.")
                        can_add_ta = False; break
                    data_for_ta[col_name] = pd.to_numeric(data_for_ta[col_name], errors='coerce')
                    if data_for_ta[col_name].isnull().all():
                        print(f"  Error: Column '{col_name}' for {ticker} became all NaNs after to_numeric. Skipping TA.")
                        can_add_ta = False; break
                except Exception as e_col_prep:
                    print(f"  Error preparing column '{col_name}' for TA for {ticker}: {e_col_prep}. Skipping TA.")
                    can_add_ta = False; break
            
            if can_add_ta:
                try:
                    print(f"  Adding TA features for {ticker}...")
                    data_for_ta.dropna(subset=ohlcv_cols, inplace=True)
                    if data_for_ta.empty:
                        print(f"  Data for TA became empty after dropping NaNs in OHLCV for {ticker}. Skipping TA.")
                        can_add_ta = False
                    else:
                        data_for_ta = ta.add_all_ta_features(
                            data_for_ta, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
                        )
                except Exception as e_ta:
                    print(f"  Error during ta.add_all_ta_features for {ticker}: {e_ta}. Proceeding without TA.")
                    can_add_ta = False
            
            if not can_add_ta:
                 data_for_ta = data_raw.copy()


        # 4. Handle Potential Issues (NaNs, infs)
        data_for_ta.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_for_ta.dropna(inplace=True)

        if data_for_ta.empty:
            print(f"  Not enough data after TA and NaN removal for {ticker}. Skipping.")
            continue

        # 5. Feature Selection & Normalization
        features_to_normalize_list = [
            'Close', 'Volume', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
            'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 
            'trend_sma_fast', 'trend_sma_slow', 'momentum_rsi'
        ]
        available_features = [f for f in features_to_normalize_list if f in data_for_ta.columns]
        
        if not ('Close' in available_features and 'Volume' in available_features):
             print(f"  Essential 'Close' or 'Volume' not in available features for normalization for {ticker}. Skipping.")
             continue
        if not available_features:
            print(f"  No features selected for normalization for {ticker}. Skipping.")
            continue

        data_subset_for_norm = data_for_ta[available_features].copy()

        for col in list(data_subset_for_norm.columns):
            try:
                data_subset_for_norm[col] = pd.to_numeric(data_subset_for_norm[col], errors='coerce')
            except Exception as e_convert:
                print(f"  Warning: Exception during to_numeric for column '{col}': {e_convert}. Removing feature.")
                data_subset_for_norm.drop(columns=[col], inplace=True)
                if col in available_features: available_features.remove(col)
        
        data_subset_for_norm.dropna(inplace=True)
        available_features = [col for col in available_features if col in data_subset_for_norm.columns]

        if data_subset_for_norm.empty or not available_features:
            print(f"  No usable numeric features left for normalization for {ticker}. Skipping.")
            continue
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_values_array = scaler.fit_transform(data_subset_for_norm)
        normalized_features_df = pd.DataFrame(
            normalized_values_array, columns=available_features, index=data_subset_for_norm.index
        )

        # 6. Align Original 'Close' and Construct Final DataFrame
        if normalized_features_df.empty:
            print(f"  Normalized features DataFrame is empty for {ticker}. Skipping.")
            continue
            
        aligned_original_close_series = original_close_unaligned.reindex(normalized_features_df.index)
        if aligned_original_close_series.isnull().any():
            aligned_original_close_series.ffill(inplace=True)
            aligned_original_close_series.bfill(inplace=True)
            if aligned_original_close_series.isnull().any():
                print(f"  Error: Still NaNs in aligned_original_close_series for {ticker} after fill. Skipping.")
                continue

        final_df_to_save = pd.DataFrame(index=normalized_features_df.index)
        final_df_to_save['Date'] = normalized_features_df.index.strftime('%Y-%m-%d')
        
        # Use the helper function here
        oc_values_1d = ensure_1d_numpy_array(aligned_original_close_series, f"{ticker} aligned 'Original_Close'")
        if len(oc_values_1d) == len(final_df_to_save.index):
            final_df_to_save['Original_Close'] = oc_values_1d
        else:
            print(f"  Error: Length mismatch for Original_Close for {ticker}. Skipping.")
            continue
            
        for col_name_norm_df in normalized_features_df.columns:
            output_col_name = f"Norm_{col_name_norm_df}" if col_name_norm_df.lower() == 'close' else col_name_norm_df
            # Use the helper function here
            feature_values_1d = ensure_1d_numpy_array(normalized_features_df[col_name_norm_df], f"{ticker} feature '{col_name_norm_df}'")
            if len(feature_values_1d) == len(final_df_to_save.index):
                final_df_to_save[output_col_name] = feature_values_1d
            else:
                print(f"  Error: Length mismatch for feature {output_col_name} for {ticker}. Skipping.")
                final_df_to_save = None; break
        
        if final_df_to_save is None: continue

        cols_order = ['Date', 'Original_Close']
        remaining_cols = sorted([col for col in final_df_to_save.columns if col not in cols_order])
        final_cols_order = cols_order + remaining_cols
        final_df_to_save = final_df_to_save[final_cols_order]

        # --- 7. Save Processed Data ---
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        final_df_to_save.to_csv(file_path, index=False)
        print(f"  Saved processed {ticker} data ({final_df_to_save.shape[0]} rows, {final_df_to_save.shape[1]} columns) to {file_path}")

    except Exception as e:
        print(f"  --- Failed to download or process {ticker}: {e} (Outer try-except) ---")
        import traceback
        traceback.print_exc()

print("Data processing finished.")