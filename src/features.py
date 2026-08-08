import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = BASE_DIR

TRAIN_PATH = os.path.join(DATA_DIR, "orders_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "orders_test.csv")
HUB_PATH = os.path.join(DATA_DIR, "hub_metadata.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "outputs", "submission.csv")

TARGET = "OrderVolume"
ID_COL = "Id"
DATE_COL = "Date"
HUB_COL = "HubID"

CATEGORICAL_FEATURES = [
    "HubID", "Weekday", "HubFormat", "AssortmentTier", 
    "RegionalHoliday", "LoyaltyProgramInterval"
]

FEATURES_TO_DROP = [
    TARGET, ID_COL, DATE_COL, "AppSessions", "IsOpen"
]

VALIDATION_DAYS = 42
RANDOM_STATE = 42

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def load_data():
    print("Loading datasets...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    hub = pd.read_csv(HUB_PATH)
    
    # Convert dates
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    return train, test, hub

def merge_and_clean(df, hub):
    # Merge hub metadata
    df = pd.merge(df, hub, on='HubID', how='left')
    
    # Impute missing CompetitorDistance with median
    median_dist = df['CompetitorDistance'].median()
    df['CompetitorDistance'] = df['CompetitorDistance'].fillna(median_dist)
    
    # Impute missing dates with 0 (indicating no competitor / no program)
    df['CompetitorOpenSinceMonth'] = df['CompetitorOpenSinceMonth'].fillna(0)
    df['CompetitorOpenSinceYear'] = df['CompetitorOpenSinceYear'].fillna(0)
    df['LoyaltyProgramSinceWeek'] = df['LoyaltyProgramSinceWeek'].fillna(0)
    df['LoyaltyProgramSinceYear'] = df['LoyaltyProgramSinceYear'].fillna(0)
    
    # LoyaltyProgramInterval might be NaN
    df['LoyaltyProgramInterval'] = df['LoyaltyProgramInterval'].fillna('None')
    
    # Cast RegionalHoliday to string just in case it has mixed types (e.g., 'a', 'b', 'c', 0)
    df['RegionalHoliday'] = df['RegionalHoliday'].astype(str)
    
    return df

def preprocess_pipeline():
    train, test, hub = load_data()
    
    # The models shouldn't train on closed stores as they naturally have 0 OrderVolume.
    # We will hardcode test predictions to 0 where IsOpen == 0.
    print(f"Train shape before IsOpen filtering: {train.shape}")
    train = train[train['IsOpen'] == 1].copy()
    print(f"Train shape after IsOpen filtering: {train.shape}")
    
    print("Merging and cleaning train...")
    train = merge_and_clean(train, hub)
    
    print("Merging and cleaning test...")
    test = merge_and_clean(test, hub)
    
    # Sort train chronologically
    train.sort_values(by=['Date', 'HubID'], inplace=True)
    train.reset_index(drop=True, inplace=True)
    
    return train, test

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def get_date_features(df):
    print("Generating Date & Cyclical Features...")
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['MonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['MonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['QuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
    df['QuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
    
    # Cyclical Features
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['WeekSin'] = np.sin(2 * np.pi * df['Week'] / 52.0)
    df['WeekCos'] = np.cos(2 * np.pi * df['Week'] / 52.0)
    df['DayOfWeekSin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['DayOfWeekCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    
    return df

def get_lag_and_rolling_features(df):
    print("Generating Comprehensive Lag, Rolling, Quantile, and Volatility Features (Horizon Shift = 42)...")
    df['LogTarget'] = np.log1p(df[TARGET])
    df = df.sort_values(['HubID', 'Date']).reset_index(drop=True)
    grouped = df.groupby('HubID')['LogTarget']
    
    # Test period is 42 days where OrderVolume is missing.
    # Base shift must be 42 days so all test rows have valid historical target lags.
    HORIZON = 42
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 14, 21, 28, 35, 42, 56]
    for lag in lags:
        df[f'Lag{lag}'] = grouped.shift(HORIZON + lag - 1)
        
    # Same-Weekday Lags
    df['LagSameWeekday1'] = df['Lag7']
    df['LagSameWeekday2'] = df['Lag14']
    df['LagSameWeekday3'] = df['Lag21']
    df['LagSameWeekday4'] = df['Lag28']
        
    roll_base = df.groupby('HubID')['Lag1']
    eps = 1e-8
    
    for window in [3, 7, 14, 21, 28, 42, 56]:
        df[f'RollingMean{window}'] = roll_base.transform(lambda x: x.rolling(window, min_periods=1).mean())
        if window in [7, 14, 28]:
            df[f'RollingStd{window}'] = roll_base.transform(lambda x: x.rolling(window, min_periods=1).std())
            df[f'EWMA{window}'] = roll_base.transform(lambda x: x.ewm(span=window, min_periods=1).mean())
        
    # Quantiles, Median, Max, Min
    df['RollingMedian7'] = roll_base.transform(lambda x: x.rolling(7, min_periods=1).median())
    df['RollingMedian14'] = roll_base.transform(lambda x: x.rolling(14, min_periods=1).median())
    df['RollingMedian28'] = roll_base.transform(lambda x: x.rolling(28, min_periods=1).median())
    
    df['RollingQuantile25'] = roll_base.transform(lambda x: x.rolling(28, min_periods=1).quantile(0.25))
    df['RollingQuantile75'] = roll_base.transform(lambda x: x.rolling(28, min_periods=1).quantile(0.75))
    
    df['RollingMax7'] = roll_base.transform(lambda x: x.rolling(7, min_periods=1).max())
    df['RollingMin7'] = roll_base.transform(lambda x: x.rolling(7, min_periods=1).min())
    
    # Volatility / Range & CV
    df['RollingRange7'] = df['RollingMax7'] - df['RollingMin7']
    df['RollingRange28'] = roll_base.transform(lambda x: x.rolling(28, min_periods=1).max()) - roll_base.transform(lambda x: x.rolling(28, min_periods=1).min())
    
    df['CoefficientOfVariation7'] = df['RollingStd7'] / (df['RollingMean7'] + eps)
    df['CoefficientOfVariation28'] = df['RollingStd28'] / (df['RollingMean28'] + eps)
    
    # Trend Features
    print("Generating Trend and Ratio Features...")
    df['Lag1-Lag7'] = df['Lag1'] - df['Lag7']
    df['Lag7-Lag14'] = df['Lag7'] - df['Lag14']
    df['Lag14-Lag28'] = df['Lag14'] - df['Lag28']
    df['Lag28-Lag56'] = df['Lag28'] - df['Lag56']
    
    df['RollingMean7-RollingMean14'] = df['RollingMean7'] - df['RollingMean14']
    df['RollingMean7-RollingMean28'] = df['RollingMean7'] - df['RollingMean28']
    df['RollingMean14-RollingMean28'] = df['RollingMean14'] - df['RollingMean28']
    
    df['Lag1/Lag7'] = df['Lag1'] / (df['Lag7'] + eps)
    df['Lag7/Lag14'] = df['Lag7'] / (df['Lag14'] + eps)
    df['Lag14/Lag28'] = df['Lag14'] / (df['Lag28'] + eps)
    df['Lag7/Lag28'] = df['Lag7'] / (df['Lag28'] + eps)
    
    # Momentum Features
    df['DemandGrowth'] = (df['Lag1'] - df['Lag2']) / (df['Lag2'] + eps)
    df['WeeklyGrowth'] = (df['Lag1'] - df['Lag7']) / (df['Lag7'] + eps)
    df['MonthlyGrowth'] = (df['Lag1'] - df['Lag28']) / (df['Lag28'] + eps)
    
    df.drop('LogTarget', axis=1, inplace=True)
    return df

def get_event_features(df):
    print("Generating Advanced Promotion and Holiday Features...")
    df = df.sort_values(['HubID', 'Date']).reset_index(drop=True)
    
    grouped_promo = df.groupby('HubID')['PromoActive']
    df['PromoCountLast3'] = grouped_promo.transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['PromoCountLast7'] = grouped_promo.transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['PromoCountLast14'] = grouped_promo.transform(lambda x: x.rolling(14, min_periods=1).sum())
    df['PromoCountLast30'] = grouped_promo.transform(lambda x: x.rolling(30, min_periods=1).sum())
    
    grouped_holiday = df.groupby('HubID')['RegionalHoliday']
    df['HolidayYesterday'] = grouped_holiday.shift(1).fillna(0).astype(int)
    df['HolidayTomorrow'] = grouped_holiday.shift(-1).fillna(0).astype(int)
    
    # Holiday Counts
    df['HolidayCountLast7'] = grouped_holiday.transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['HolidayCountNext7'] = grouped_holiday.transform(lambda x: x.iloc[::-1].rolling(7, min_periods=1).sum().iloc[::-1])
    
    # Lag Holiday Flags (Horizon Shift = 42)
    HORIZON = 42
    df['Lag7WasHoliday'] = grouped_holiday.shift(HORIZON + 7 - 1).fillna(0).astype(int)
    df['Lag14WasHoliday'] = grouped_holiday.shift(HORIZON + 14 - 1).fillna(0).astype(int)
    df['Lag28WasHoliday'] = grouped_holiday.shift(HORIZON + 28 - 1).fillna(0).astype(int)
    
    for col, prefix in zip(['PromoActive', 'RegionalHoliday'], ['Promo', 'Holiday']):
        df[col] = df[col].astype(float)
        
        df[f'Last{prefix}Date'] = df['Date'].where(df[col] == 1)
        df[f'Last{prefix}Date'] = df.groupby('HubID')[f'Last{prefix}Date'].ffill()
        df[f'DaysSince{prefix}'] = (df['Date'] - df[f'Last{prefix}Date']).dt.days.fillna(999)
        
        df[f'Next{prefix}Date'] = df['Date'].where(df[col] == 1)
        df[f'Next{prefix}Date'] = df.groupby('HubID')[f'Next{prefix}Date'].bfill()
        df[f'DaysUntilNext{prefix}'] = (df[f'Next{prefix}Date'] - df['Date']).dt.days.fillna(999)
        
        df.drop([f'Last{prefix}Date', f'Next{prefix}Date'], axis=1, inplace=True)

    df['IsPromoStart'] = ((df['PromoActive'] == 1) & (df.groupby('HubID')['PromoActive'].shift(1) == 0)).astype(int)
    df['IsPromoEnd'] = ((df['PromoActive'] == 1) & (df.groupby('HubID')['PromoActive'].shift(-1) == 0)).astype(int)
    df['PromoDuration'] = df.groupby('HubID')['PromoActive'].transform(lambda x: x.groupby((x == 0).cumsum()).cumsum())
    
    df['HolidayWeek'] = df.groupby(['HubID', 'Year', 'WeekOfYear'])['RegionalHoliday'].transform('max').astype(int)
    df['WeekendHoliday'] = (df['Weekend'].astype(bool) & df['RegionalHoliday'].astype(bool)).astype(int)
    
    return df

def get_metadata_and_interactions(df):
    print("Generating Hub Metadata, Non-Leaking Hub Stats, and Interactions...")
    df['CompetitorExists'] = (df['CompetitorDistance'].notna() & (df['CompetitorDistance'] > 0)).astype(int)
    df['LogCompetitorDistance'] = np.log1p(df['CompetitorDistance'])
    
    df['CompetitorAge'] = 12 * (df['Year'] - df['CompetitorOpenSinceYear']) + (df['Month'] - df['CompetitorOpenSinceMonth'])
    df.loc[df['CompetitorOpenSinceYear'] == 0, 'CompetitorAge'] = 0
    df['CompetitorAge'] = df['CompetitorAge'].clip(lower=0)
    
    df['LoyaltyAge'] = 12 * (df['Year'] - df['LoyaltyProgramSinceYear']) + (df['WeekOfYear'] - df['LoyaltyProgramSinceWeek']) / 4.0
    df.loc[df['LoyaltyProgramSinceYear'] == 0, 'LoyaltyAge'] = 0
    df['LoyaltyAge'] = df['LoyaltyAge'].clip(lower=0)
    
    first_obs = df.groupby('HubID')['Date'].transform('min')
    df['HubAge'] = (df['Date'] - first_obs).dt.days // 30
    
    df['AppSessions'] = df.groupby('HubID')['AppSessions'].ffill().fillna(0)
    
    # Hub Historical Demand Stats based on Lag1 (Horizon Shift = 42, zero leakage)
    df['HubDemandMean'] = df.groupby('HubID')['Lag1'].transform('mean')
    df['HubDemandMedian'] = df.groupby('HubID')['Lag1'].transform('median')
    df['HubDemandStd'] = df.groupby('HubID')['Lag1'].transform('std')
    
    # Interactions
    df['Promo_x_Weekend'] = df['PromoActive'] * df['Weekend']
    df['Promo_x_Holiday'] = df['PromoActive'] * df['RegionalHoliday'].astype(float)
    df['Promo_x_Loyalty'] = df['PromoActive'] * df['LoyaltyProgram'].astype(float)
    df['Promo_x_HubFormat'] = df['PromoActive'] * df['HubFormat'].astype(float)
    df['Promo_x_Weekday'] = df['PromoActive'] * df['DayOfWeek']
    df['Holiday_x_Weekday'] = df['RegionalHoliday'].astype(float) * df['DayOfWeek']
    df['Sessions_x_Promo'] = df['AppSessions'] * df['PromoActive']
    df['Sessions_x_IsOpen'] = df['AppSessions'] * df['IsOpen'].astype(float)
    
    return df

def get_oof_target_encoding(df):
    print("Generating Expanding Mean Target Encoding (Horizon Shift = 42)...")
    train_mask = df[TARGET].notna()
    if not train_mask.any():
        return df
        
    df['LogTarget'] = np.log1p(df[TARGET])
    
    HORIZON = 42
    df['OOF_Hub_Target'] = df.groupby('HubID')['LogTarget'].transform(
        lambda x: x.shift(HORIZON).expanding().mean()
    )
    
    global_mean = df.loc[train_mask, 'LogTarget'].mean()
    df['OOF_Hub_Target'] = df['OOF_Hub_Target'].fillna(global_mean)
    
    df.drop('LogTarget', axis=1, inplace=True)
    return df

def feature_engineering_pipeline(train, test):
    print("Concatenating Train and Test for complex historical features...")
    train['is_test'] = 0
    test['is_test'] = 1
    
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df = df.sort_values(by=['HubID', 'Date']).reset_index(drop=True)
    
    df = get_date_features(df)
    df = get_lag_and_rolling_features(df)
    df = get_event_features(df)
    df = get_metadata_and_interactions(df)
    df = get_oof_target_encoding(df)
    
    print("Encoding categorical features...")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    df = reduce_mem_usage(df)
            
    print("Splitting Train and Test...")
    train_out = df[df['is_test'] == 0].drop('is_test', axis=1).reset_index(drop=True)
    test_out = df[df['is_test'] == 1].drop('is_test', axis=1).reset_index(drop=True)
    
    return train_out, test_out
