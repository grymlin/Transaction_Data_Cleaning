"""
Data Cleaning Functions for Cafe Sales Dataset
==============================================

This module contains functions for cleaning and standardizing cafe sales data.
It handles missing values, data type conversions, and standardization of categorical variables.

Author: Rym Otsmane
Date: 06/27/2025
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Configuration constants
INVALID_PAYMENT_METHODS = [
    'UNKNOWN', '', 'ERROR', 'nan_count_Price_Per_Unit', 'N/A', 
    'nan', 'None', 'null', 'NaN', 'NA', 'Unknown'
]

INVALID_LOCATION_VALUES = [
    'UNKNOWN', '', 'ERROR', 'nan_count_Price_Per_Unit', 'N/A', 
    'nan', 'None', 'null', 'NaN', 'NA', 'Unknown'
]

INVALID_ITEM_VALUES = ['', 'UNKNOWN', 'Void', 'ERROR']


def load_data(file_path):
    """
    Load cafe sales data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df


def create_clean_copy(df):
    """
    Create a clean copy and convert numeric columns.
    
    Args:
        df (pd.DataFrame): Original dataframe
        
    Returns:
        pd.DataFrame: Clean copy with numeric conversions
    """
    df_clean = df.copy()
    
    # Convert relevant columns to numeric, coercing errors to NaN
    df_clean['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df_clean['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
    df_clean['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')
    
    return df_clean


def check_missing_values(df):
    """
    Check for missing values in key numeric columns.
    
    Args:
        df (pd.DataFrame): Dataframe to check
        
    Returns:
        dict: Dictionary with column names and missing value counts
    """
    nan_count_quantity = df["Quantity"].isna().sum()
    nan_count_price = df["Price Per Unit"].isna().sum()
    nan_count_total = df["Total Spent"].isna().sum()
    
    print(f"Number of NaN values in 'Quantity': {nan_count_quantity}")
    print(f"Number of NaN values in 'Price Per Unit': {nan_count_price}")
    print(f"Number of NaN values in 'Total Spent': {nan_count_total}")
    
    return {
        'Quantity': nan_count_quantity,
        'Price Per Unit': nan_count_price,
        'Total Spent': nan_count_total
    }


def fill_missing_numeric_values(df):
    """
    Fill missing values in numeric columns using relationships between columns.
    
    Args:
        df (pd.DataFrame): Dataframe with missing values
        
    Returns:
        pd.DataFrame: Dataframe with filled missing values
    """
    # Fill missing Total Spent
    df.loc[df["Total Spent"].isna() & ~df["Price Per Unit"].isna() & ~df["Quantity"].isna(), 
           "Total Spent"] = df["Price Per Unit"] * df["Quantity"]

    # Fill missing Price Per Unit
    df.loc[df["Price Per Unit"].isna() & ~df["Total Spent"].isna() & ~df["Quantity"].isna(), 
           "Price Per Unit"] = df["Total Spent"] / df["Quantity"]

    # Fill missing Quantity
    df.loc[df["Quantity"].isna() & ~df["Total Spent"].isna() & ~df["Price Per Unit"].isna(), 
           "Quantity"] = df["Total Spent"] / df["Price Per Unit"]
    
    return df


def check_calculation_consistency(df):
    """
    Check if 'Total Spent' matches 'Quantity' * 'Price Per Unit'.
    
    Args:
        df (pd.DataFrame): Dataframe to check
        
    Returns:
        int: Number of inconsistent rows
    """
    mismatch = ~np.isclose(
        df['Total Spent'],
        df['Quantity'] * df['Price Per Unit'],
        equal_nan=True
    )
    num_mismatch = mismatch.sum()
    print(f"Rows where 'Total Spent' does NOT match 'Quantity' * 'Price Per Unit': {num_mismatch}")
    return num_mismatch


def fix_calculation_inconsistencies(df):
    """
    Fix inconsistent 'Total Spent' values by recalculating.
    
    Args:
        df (pd.DataFrame): Dataframe with inconsistencies
        
    Returns:
        pd.DataFrame: Dataframe with fixed calculations
    """
    # Identify mismatches
    mismatch = ~np.isclose(
        df['Total Spent'],
        df['Quantity'] * df['Price Per Unit'],
        equal_nan=True
    )
    
    # Fix mismatches
    df.loc[mismatch, 'Total Spent'] = df['Quantity'] * df['Price Per Unit']
    
    # Verify fix
    mismatch_after = ~np.isclose(
        df['Total Spent'],
        df['Quantity'] * df['Price Per Unit'],
        equal_nan=True
    )
    print(f"Rows remaining with mismatch after fix: {mismatch_after.sum()}")
    
    return df


def clean_transaction_dates(df):
    """
    Clean and validate transaction dates.
    
    Args:
        df (pd.DataFrame): Dataframe with date column
        
    Returns:
        tuple: (cleaned dataframe, number of invalid dates converted)
    """
    # Count invalid date entries before cleaning
    invalid_dates_before = pd.to_datetime(df['Transaction Date'], errors='coerce').isna().sum()
    print(f"Number of invalid 'Transaction Date' entries before cleaning: {invalid_dates_before}")
    
    # Find entries that are not null but will become NaT after conversion
    invalid_date_mask = (pd.to_datetime(df['Transaction Date'], errors='coerce').isna() & 
                        df['Transaction Date'].notna())
    invalid_dates = df.loc[invalid_date_mask, 'Transaction Date']
    
    print(f"Number of invalid 'Transaction Date' entries before cleaning: {invalid_dates.shape[0]}")
    if len(invalid_dates) > 0:
        print("Sample invalid 'Transaction Date' entries before cleaning:")
        print(invalid_dates.head())
    
    # Convert 'Transaction Date' to datetime, invalid entries become NaT
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    
    converted_to_nat = invalid_dates.shape[0]
    print(f"Number of entries converted to NaT after conversion: {converted_to_nat}")
    
    return df, converted_to_nat


def check_duplicate_transaction_ids(df):
    """
    Check for duplicate Transaction IDs.
    
    Args:
        df (pd.DataFrame): Dataframe to check
        
    Returns:
        int: Number of duplicate transaction IDs
    """
    duplicates = df['Transaction ID'].duplicated().sum()
    print(f"Duplicate Transaction IDs: {duplicates}")
    return duplicates


def standardize_payment_methods(df):
    """
    Standardize Payment Method values.
    
    Args:
        df (pd.DataFrame): Dataframe to standardize
        
    Returns:
        pd.DataFrame: Dataframe with standardized payment methods
    """
    print("Unique Payment Methods before standardization:", df['Payment Method'].unique())
    
    # Count how many will be replaced
    num_payment_standardized = (df['Payment Method'].isin(INVALID_PAYMENT_METHODS).sum() + 
                               df['Payment Method'].isna().sum())
    
    # Replace with 'Unknown'
    df.loc[df['Payment Method'].isin(INVALID_PAYMENT_METHODS), 'Payment Method'] = 'Unknown'
    df.loc[df['Payment Method'].isna(), 'Payment Method'] = 'Unknown'
    
    print(f"Standardized {num_payment_standardized} 'Payment Method' values to 'Unknown'.")
    print("Unique Payment Methods after standardization:", df['Payment Method'].unique())
    
    return df


def standardize_locations(df):
    """
    Standardize Location values.
    
    Args:
        df (pd.DataFrame): Dataframe to standardize
        
    Returns:
        pd.DataFrame: Dataframe with standardized locations
    """
    print("Unique Locations before standardization:", df['Location'].unique())
    
    # Count how many will be replaced
    num_location_standardized = (df['Location'].isin(INVALID_LOCATION_VALUES).sum() + 
                                df['Location'].isna().sum())
    
    # Replace with 'Unknown'
    df.loc[df['Location'].isin(INVALID_LOCATION_VALUES), 'Location'] = 'Unknown'
    df.loc[df['Location'].isna(), 'Location'] = 'Unknown'
    
    print(f"Standardized {num_location_standardized} 'Location' values to 'Unknown'.")
    print("Unique Locations after standardization:", df['Location'].unique())
    
    return df


def standardize_items(df):
    """
    Standardize Item values.
    
    Args:
        df (pd.DataFrame): Dataframe to standardize
        
    Returns:
        pd.DataFrame: Dataframe with standardized items
    """
    print("Unique Items before standardization:", df['Item'].unique())
    
    # Check missing or invalid 'Item' values
    item_missing = df['Item'].isnull() | df['Item'].isin(INVALID_ITEM_VALUES)
    num_item_standardized = item_missing.sum()
    
    print(f"Missing or invalid 'Item' values before standardization: {num_item_standardized}")
    
    # Standardize missing or invalid 'Item' values
    if num_item_standardized > 0:
        df.loc[item_missing, 'Item'] = 'Unknown'
    
    print(f"Standardized {num_item_standardized} 'Item' values to 'Unknown'.")
    print("Unique Items after standardization:", df['Item'].unique())
    
    return df


def check_future_dates(df):
    """
    Check for future transaction dates.
    
    Args:
        df (pd.DataFrame): Dataframe to check
        
    Returns:
        int: Number of future dates
    """
    now = pd.Timestamp(datetime.now().date())
    future_dates = (df['Transaction Date'] > now).sum()
    print(f"Future Transaction Dates: {future_dates}")
    return future_dates


def clean_cafe_data(file_path, output_path=None):
    """
    Complete data cleaning pipeline for cafe sales data.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path to save cleaned data
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Starting data cleaning pipeline...")
    print("=" * 50)
    
    # Load data
    df = load_data(file_path)
    
    # Create clean copy and convert numeric columns
    df_clean = create_clean_copy(df)
    
    # Check missing values
    print("\n1. Checking missing values:")
    check_missing_values(df_clean)
    
    # Fill missing numeric values
    print("\n2. Filling missing numeric values:")
    df_clean = fill_missing_numeric_values(df_clean)
    
    # Check and fix calculation consistency
    print("\n3. Checking calculation consistency:")
    check_calculation_consistency(df_clean)
    df_clean = fix_calculation_inconsistencies(df_clean)
    
    # Clean transaction dates
    print("\n4. Cleaning transaction dates:")
    df_clean, _ = clean_transaction_dates(df_clean)
    
    # Check duplicates
    print("\n5. Checking duplicate Transaction IDs:")
    check_duplicate_transaction_ids(df_clean)
    
    # Standardize categorical variables
    print("\n6. Standardizing Payment Methods:")
    df_clean = standardize_payment_methods(df_clean)
    
    print("\n7. Standardizing Locations:")
    df_clean = standardize_locations(df_clean)
    
    print("\n8. Standardizing Items:")
    df_clean = standardize_items(df_clean)
    
    # Check future dates
    print("\n9. Checking for future dates:")
    check_future_dates(df_clean)
    
    # Save cleaned data
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
    
    print("\n" + "=" * 50)
    print("Data cleaning pipeline completed!")
    
    return df_clean


if __name__ == "__main__":
    # Set working directory to the project root (parent of 'src')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_root)
    print("Working directory set to:", os.getcwd())

    input_file = "data/dirty_cafe_sales.csv"
    output_file = "data/cleaned_cafe_sales.csv"

    cleaned_data = clean_cafe_data(input_file, output_file)
    print(f"\nFinal cleaned dataset shape: {cleaned_data.shape}")