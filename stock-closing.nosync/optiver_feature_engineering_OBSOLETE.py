# Bronte Sihan Li, Cole Crescas Dec 2023
# CS7180

"""
Setup:
The script checks for GPU availability, sets up global constants, and utilizes various libraries, such as Pandas, Numba, and Torch

This script, facilitates the preprocessing and feature engineering of stock trading data. 
1. The script reads a CSV file containing stock data
2. drops specific features
3. addresses missing values by filling them with zeros, means, or medians based on user preference. 

Added additional features:
1. including imbalance features 
2. time-stock-related features 

The script also incorporates global feature mapping and applies memory optimization techniques during data loading and cleaning. 
The processed data is saved to a new CSV file, and two dataframes are combined by appending dropped columns. 
The final dataset, suitable for model training, is then exported to a CSV file. 

"""


import pandas as pd
import numpy as np
from typing import Literal
import gc  # Garbage collection for memory management
import warnings  # Handling warnings
from itertools import combinations  # For creating combinations of elements
from warnings import simplefilter  # Simplifying warning handling
import torch
from numba import njit, prange

# Disable warnings to keep the code clean
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def reduce_mem_usage(df, verbose=1):
    """
    Takes in a dataframe to convert memory types of columns in the dataset
    Args:
        df: Dataframe input to convert
        verbose: flag of how much to output
    Returns:
        df: dataframe with changed types
    """
    # Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    # Iterate through each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column's data type is not 'object' (i.e., numeric)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            # Check if the column's data type is an integer
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float32)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    # ℹProvide memory optimization information if 'verbose' is True
    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {decrease:.2f}%")

    # Return the DataFrame with optimized memory usage
    return df


def sizesum_and_pricestd(df):
    """
    Takes in a dataframe to add rolling features for sum and std deviation of the
    global variables PRICE_FTRS & SIZE_FTRS signifying certain columns that fall
    into those buckets.  Another possible addition would be to add a different
    window lookback.
    Args:
        df: Dataframe input to convert
    Returns:
        df: dataframe with added fields
    """
    rolled_size = (
        df[['stock_id'] + SIZE_FTRS]
        .groupby('stock_id')
        .rolling(window=6, min_periods=1)
        .sum()
    )
    rolled_size = rolled_size.reset_index(level=0, drop=True)
    for col in SIZE_FTRS:
        df[f'{col}_rolled_sum'] = rolled_size[col]

    rolled_price = (
        df[['stock_id'] + PRICE_FTRS]
        .groupby('stock_id')
        .rolling(window=6, min_periods=1)
        .std()
        .fillna(0)
    )
    rolled_price = rolled_price.reset_index(level=0, drop=True)
    for col in PRICE_FTRS:
        df[f'{col}_rolled_std'] = rolled_price[col]

    return df


# Function to compute triplet imbalance in parallel using Numba
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    """
    Calculate triplet imbalance features for given combinations of columns.

    Parameters:
    - df_values (numpy.ndarray): 2D array containing the values of the DataFrame.
    - comb_indices (list): List of combinations of triplet indices.

    Returns:
    - imbalance_features (numpy.ndarray): 2D array with triplet imbalance features.
    """
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    # Loop through all combinations of triplets
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]

        # Loop through rows of the DataFrame
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = (
                df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            )

            # Prevent division by zero
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


# Function to calculate triplet imbalance for given price data and a DataFrame
def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [
        (price.index(a), price.index(b), price.index(c))
        for a, b, c in combinations(price, 3)
    ]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    return pd.DataFrame(features_array, columns=columns)


# Function to generate imbalance features
def imbalance_features(df):
    """
    Calculate various imbalance features using Pandas eval function and statistical aggregations.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing stock data.

    Returns:
    - df (pd.DataFrame): DataFrame with added imbalance features.
    """
    # if IS_CUDA:
    #     import cudf
    #     df = cudf.from_pandas(df)

    # V1 features
    # Calculate various features using Pandas eval function
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("ask_price + bid_price") / 2
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("imbalance_size-matched_size") / df.eval(
        "matched_size+imbalance_size"
    )
    df["size_imbalance"] = df.eval("bid_size / ask_size")

    # Create features for pairwise price imbalances
    for c in combinations(PRICE_FTRS, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    # V2 features
    # Calculate additional features
    df["imbalance_momentum"] = (
        df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    )
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (
        df['far_price'] - df['near_price']
    )

    # Calculate various statistical aggregation features

    # V3 features
    # Calculate shifted and return features for specific columns
    for col in [
        'matched_size',
        'imbalance_size',
        'reference_price',
        'imbalance_buy_sell_flag',
    ]:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)

    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    # if IS_CUDA:
    #     df = df.to_pandas()
    # Replace infinite values with 0
    return df.replace([np.inf, -np.inf], 0)


def numba_imb_features(df):
    """
    Calculate imbalance features using Numba-optimized functions.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing stock data.

    Returns:
    - df (pd.DataFrame): DataFrame with added imbalance features.
    """

    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[PRICE_FTRS].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[SIZE_FTRS].agg(func, axis=1)

    # Calculate triplet imbalance features using the Numba-optimized function
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], SIZE_FTRS]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
    return df


# Function to generate time and stock-related features
def other_features(df):
    """
    Function to generate time and stock-related features
    """
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    df["minute"] = df["seconds_in_bucket"] // 60  # Minutes

    # Adding per stock features grouped my stock
    g_ask_size_median = df.groupby("stock_id")["ask_size"].median()
    g_ask_size_std = df.groupby("stock_id")["ask_size"].std()
    g_ask_size_min = df.groupby("stock_id")["ask_size"].min()
    g_ask_size_max = df.groupby("stock_id")["ask_size"].max()
    g_bid_size_median = df.groupby("stock_id")["bid_size"].median()
    g_bid_size_std = df.groupby("stock_id")["bid_size"].std()
    g_bid_size_min = df.groupby("stock_id")["bid_size"].min()
    g_bid_size_max = df.groupby("stock_id")["bid_size"].max()
    # Aggregate from all features
    global_stock_id_feats = {
        "median_size": g_bid_size_median + g_ask_size_median,
        "std_size": g_bid_size_std + g_ask_size_std,
        "ptp_size": g_bid_size_max - g_bid_size_min,
        "median_price": g_bid_size_median + g_ask_size_median,
        "std_price": g_bid_size_std + g_ask_size_std,
        "ptp_price": g_bid_size_max - g_ask_size_min,
    }
    print("Build Online Train Feats Finished.")

    # Map global features to the DataFrame
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())
    return df


def generate_all_features(df):
    """
    Function to generate all features by combining imbalance and other features
    """
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]

    # Generate imbalance features
    df = imbalance_features(df)
    df = numba_imb_features(df)
    # Generate time and stock-related features
    df = other_features(df)
    gc.collect()  # Perform garbage collection to free up memory

    # Select and return the generated features
    feature_name = [
        i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]
    ]
    return df[feature_name]


def load_and_clean_data(
    data_filepath: str,
    fillna: Literal['zero', 'mean', 'median'] = 'median',
    add_features_flag: bool = True,
) -> pd.DataFrame:
    """
    Load and clean data from csv file.
    Args:
        data_filepath (string): Path to the csv file with stock data.
        fillna (string): How to fill NaN values. Default: 'median'.
        add_features_flag (bool): Boolean to call additional methods. Default: True
    Returns:
        data (DataFrame): Cleaned data with added features
    """
    # Load data from csv file
    data = pd.read_csv(data_filepath)
    # Drop features
    data = data.drop(columns=DROP_FEATURES)

    data['far_price'].fillna(0, inplace=True)
    data['near_price'].fillna(1, inplace=True)

    if fillna == 'zero':
        # Replace all NaN values with 0
        data = data.fillna(0)
    elif fillna == 'mean':
        # Replace all NaN values in far_price and near_price with column mean
        data = data.fillna(data.mean())
    elif fillna == 'median':
        cols_group_by = ['stock_id']
        train_grouped_median = data.groupby(cols_group_by).transform('median')
        data.fillna(train_grouped_median, inplace=True)
    else:
        raise ValueError(f"fillna must be 'zero' or 'mean', not {fillna}.")

    print("Null values: ", data.isnull().sum().sum())

    data = reduce_mem_usage(data)

    # this will call a series of methods that will add features
    if add_features_flag:
        data = sizesum_and_pricestd(data)
        data_with_ftrs = generate_all_features(data)

    return data_with_ftrs


# ---------------------------------- GLOBALS ------------------------------------

IS_CUDA = torch.cuda.is_available()
NB_CARDS = torch.cuda.device_count()
print(f"{IS_CUDA=} with {NB_CARDS=}")

DATA_FILE_DIR = '/notebooks/advanced-perception/stock-closing.nosync/data/train.csv'
DROP_FEATURES = [
    # 'far_price',
    # 'near_price',
    'row_id',
]
MAX_SECONDS = 55  # Maximum number of seconds * 10 in a window

NO_FEATURE_COLS = ['date_id', 'time_id', 'target']

PRICE_FTRS = [
    "reference_price",
    "far_price",
    "near_price",
    "ask_price",
    "bid_price",
    "wap",
]  # sum
SIZE_FTRS = ["matched_size", "bid_size", "ask_size", "imbalance_size"]  # std

# MAY NEED TO RUN THIS for cudf:
# https://colab.research.google.com/drive/1TAAi_szMfWqRfHVfjGSqnGVLr_ztzUM9#scrollTo=bvcxjJOCX1tL

# --------------------------- END SETUP -----------------------------------------
# this will reduce memory usage before features, add all features and then call
# all methods above.
df_train = load_and_clean_data(DATA_FILE_DIR)

# should reduce memory usage by around 25%, taken from here: https://www.kaggle.com/code/jirkaborovec/optiver-features-torch-dnn-infer-tabular
df_train_feats = reduce_mem_usage(df_train)

df_train_feats.to_csv(
    '/notebooks/advanced-perception/stock-closing.nosync/data/train_added_features.csv',
    index=False,
)

# Below has been commented out since it is not needed

# df_train_target = data['target'].astype(np.float16)
# We need to use the date_id from df_train to split the data
# df_train_date_ids = df_train['date_id'].values
# Free up memory by deleting
# del df_train
# gc.collect()

# # Assuming df_train_feats and df_train are already defined and df_train contains the 'date_id' column
# FEATURE_NAMES = list(df_train_feats.columns)
# print(f"Feature length = {len(FEATURE_NAMES)} as {sorted(FEATURE_NAMES)}")

# df_train_feats.head()


# This block is done after the data is already created from methods and code blocks below

data = pd.read_csv(
    '/notebooks/advanced-perception/stock-closing.nosync/data/train_added_features.csv'
)

FEATURE_NAMES = list(df_train_feats.columns)
print(f"Feature length = {len(FEATURE_NAMES)} as {sorted(FEATURE_NAMES)}")

df_train_feats.head()

# Load two data frames and append the columns dropped during feature creation
data = pd.read_csv(
    '/notebooks/advanced-perception/stock-closing.nosync/data/train_added_features.csv'
)

df2 = pd.read_csv('/notebooks/advanced-perception/stock-closing.nosync/data/train.csv')

columns_to_append = df2[['target', 'time_id', 'date_id']]

data = pd.concat([data, columns_to_append], axis=1)

# save final dataframe to be used in model training
data.to_csv(
    '/notebooks/advanced-perception/stock-closing.nosync/data/train_added_features.csv',
    index=False,
)
print(data.shape)
