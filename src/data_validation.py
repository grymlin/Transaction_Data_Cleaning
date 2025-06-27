"""
Data Validation Functions for Cafe Sales Dataset - Enhanced Version
================================================================

This module contains functions for validating cleaned data by comparing
raw and processed datasets. It includes checks for data types, missing values,
duplicates, and data quality metrics with a unified dashboard display.

Author: Rym Otsmane
Date: 06/27/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from typing import Dict, Tuple, Optional


def load_datasets(raw_path: str, clean_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw and cleaned datasets for comparison.
    
    Args:
        raw_path (str): Path to raw dataset
        clean_path (str): Path to cleaned dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Raw and cleaned dataframes
    """
    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(clean_path)
    
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Cleaned dataset shape: {clean_df.shape}")
    
    return raw_df, clean_df


def compare_data_types(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """
    Compare data types between raw and cleaned datasets.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
    """
    print("Raw data types:\n", raw_df.dtypes)
    print("\nCleaned data types:\n", clean_df.dtypes)


def check_duplicate_records(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for duplicate records in both datasets.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Dict[str, int]: Duplicate counts for each dataset
    """
    dup_raw = raw_df.duplicated().sum()
    dup_clean = clean_df.duplicated().sum()
    
    print(f"Duplicate rows in raw data: {dup_raw}")
    print(f"Duplicate rows in cleaned data: {dup_clean}")
    
    return {'raw': dup_raw, 'cleaned': dup_clean}


def compare_missing_values(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare missing values between raw and cleaned datasets.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
        
    Returns:
        pd.DataFrame: Comparison of missing values
    """
    # Get missing value counts for numeric columns
    numeric_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    
    nan_counts_before = []
    nan_counts_after = []
    
    for col in numeric_cols:
        # Convert to numeric for raw data to count properly
        raw_numeric = pd.to_numeric(raw_df[col], errors='coerce')
        nan_counts_before.append(raw_numeric.isna().sum())
        nan_counts_after.append(clean_df[col].isna().sum())
    
    comparison_df = pd.DataFrame({
        'Column': numeric_cols,
        'Before Cleaning': nan_counts_before,
        'After Cleaning': nan_counts_after
    })
    
    print("Missing Values Comparison:")
    print(comparison_df)
    
    return comparison_df


def check_calculation_consistency_comparison(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, int]:
    """
    Compare calculation consistency between raw and cleaned datasets.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Dict[str, int]: Inconsistency counts for each dataset
    """
    # Calculate inconsistent rows in raw data
    raw_total = pd.to_numeric(raw_df['Total Spent'], errors='coerce')
    raw_quantity = pd.to_numeric(raw_df['Quantity'], errors='coerce')
    raw_price = pd.to_numeric(raw_df['Price Per Unit'], errors='coerce')
    
    raw_inconsistent = ~np.isclose(
        raw_total,
        raw_quantity * raw_price,
        equal_nan=True
    )
    num_raw_inconsistent = raw_inconsistent.sum()
    
    # Calculate inconsistent rows in cleaned data
    clean_inconsistent = ~np.isclose(
        clean_df['Total Spent'],
        clean_df['Quantity'] * clean_df['Price Per Unit'],
        equal_nan=True
    )
    num_clean_inconsistent = clean_inconsistent.sum()
    
    print(f"Inconsistent calculations in raw data: {num_raw_inconsistent}")
    print(f"Inconsistent calculations in cleaned data: {num_clean_inconsistent}")
    
    return {'raw': num_raw_inconsistent, 'cleaned': num_clean_inconsistent}


def check_outliers(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Check for outliers in numeric columns for both datasets.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Dict[str, Dict[str, float]]: Outlier statistics for each dataset
    """
    numeric_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    outlier_stats = {}
    
    for col in numeric_cols:
        # Convert raw data to numeric for proper comparison
        raw_numeric = pd.to_numeric(raw_df[col], errors='coerce')
        
        # Calculate 99th percentiles
        q99_raw = raw_numeric.quantile(0.99)
        q99_clean = clean_df[col].quantile(0.99)
        
        # Count outliers (values > 99th percentile)
        outliers_raw = (raw_numeric > q99_raw).sum()
        outliers_clean = (clean_df[col] > q99_clean).sum()
        
        print(f"99th percentile of {col} in raw data: {q99_raw}")
        print(f"99th percentile of {col} in cleaned data: {q99_clean}")
        print(f"Number of {col} outliers in raw data: {outliers_raw}")
        print(f"Number of {col} outliers in cleaned data: {outliers_clean}")
        print("-" * 50)
        
        outlier_stats[col] = {
            'raw_q99': q99_raw,
            'clean_q99': q99_clean,
            'raw_outliers': outliers_raw,
            'clean_outliers': outliers_clean
        }
    
    return outlier_stats


def create_validation_dashboard(raw_df: pd.DataFrame, clean_df: pd.DataFrame, 
                              missing_comparison: pd.DataFrame, 
                              consistency_counts: Dict[str, int],
                              save_dir: str = "validation_plots") -> None:
    """
    Create a unified dashboard with all validation plots and individual save options.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Data Validation Dashboard - Raw vs Cleaned Data', fontsize=16, fontweight='bold')

    # 1. Missing Values Comparison (left)
    ax1 = plt.subplot(2, 2, 1)
    comparison_melted = missing_comparison.melt(id_vars='Column', var_name='Stage', value_name='NaN Count')
    sns.barplot(data=comparison_melted, x='Column', y='NaN Count', hue='Stage', ax=ax1)
    ax1.set_title('Missing Values: Before vs After Cleaning', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Unique Values Before/After (right)
    ax2 = plt.subplot(2, 2, 2)
    labels = ['Payment Method', 'Location', 'Item']
    raw_uniques = [raw_df[c].nunique() for c in labels]
    clean_uniques = [clean_df[c].nunique() for c in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, raw_uniques, width, label='Raw')
    ax2.bar(x + width/2, clean_uniques, width, label='Cleaned')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Unique Values')
    ax2.set_title('Unique Values: Raw vs Cleaned')
    ax2.legend()

    # 3. Calculation Consistency (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(['Raw Data', 'Cleaned Data'], 
            [consistency_counts['raw'], consistency_counts['cleaned']], 
            color=['#d9534f', '#5cb85c'])
    ax3.set_ylabel("Number of Inconsistent Rows")
    ax3.set_title("Calculation Consistency:\nTotal ≠ Quantity × Price", fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Data Quality Summary & Most Improved Column (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    raw_rows, raw_cols = raw_df.shape
    clean_rows, clean_cols = clean_df.shape
    rows_removed = raw_rows - clean_rows
    duplicates_removed = check_duplicate_records(raw_df, clean_df)['raw'] - check_duplicate_records(raw_df, clean_df)['cleaned']

    # Most improved column (missing values)
    missing_improvements = (missing_comparison['Before Cleaning'] - missing_comparison['After Cleaning'])
    most_improved_idx = missing_improvements.idxmax()
    most_improved_col = missing_comparison.loc[most_improved_idx, 'Column']
    most_improved_val = missing_improvements.max()

    # Data quality score
    completeness = 1 - (missing_comparison['After Cleaning'].sum() / max(1, missing_comparison['Before Cleaning'].sum()))
    consistency = 1 - (consistency_counts['cleaned'] / max(1, consistency_counts['raw']))
    uniqueness = 1 - ((sum(clean_uniques) / max(1, sum(raw_uniques))))
    data_quality_score = np.mean([completeness, consistency, uniqueness])

    summary_text = f"""
    DATA QUALITY SUMMARY

    Original Dataset:
    • Rows: {raw_rows:,}
    • Columns: {raw_cols}

    Cleaned Dataset:
    • Rows: {clean_rows:,}
    • Columns: {clean_cols}

    Improvements:
    • Rows removed: {rows_removed:,}
    • Duplicates removed: {duplicates_removed}
    • Inconsistencies fixed: {consistency_counts['raw'] - consistency_counts['cleaned']}
    • Most improved column (missing values): {most_improved_col} ({most_improved_val} filled)
    • Data Quality Score (0-100): {data_quality_score * 100:.1f}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    dashboard_path = os.path.join(save_dir, f"validation_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_individual_plots(raw_df: pd.DataFrame, clean_df: pd.DataFrame, 
                         missing_comparison: pd.DataFrame, 
                         consistency_counts: Dict[str, int],
                         save_dir: str) -> None:
    """
    Save individual plots as separate files.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
        missing_comparison (pd.DataFrame): Missing values comparison data
        consistency_counts (Dict[str, int]): Calculation consistency counts
        save_dir (str): Directory to save plots
    """
    
    # 1. Missing Values Plot
    plt.figure(figsize=(10, 6))
    comparison_melted = missing_comparison.melt(id_vars='Column', var_name='Stage', value_name='NaN Count')
    sns.barplot(data=comparison_melted, x='Column', y='NaN Count', hue='Stage')
    plt.title('Missing Values: Before vs After Cleaning')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "missing_values_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Location Distribution
    plt.figure(figsize=(12, 6))
    raw_location_counts = raw_df['Location'].value_counts()
    clean_location_counts = clean_df['Location'].value_counts()
    
    all_locations = set(raw_location_counts.index) | set(clean_location_counts.index)
    location_comparison = pd.DataFrame({
        'Location': list(all_locations),
        'Raw': [raw_location_counts.get(loc, 0) for loc in all_locations],
        'Cleaned': [clean_location_counts.get(loc, 0) for loc in all_locations]
    })
    
    x_pos = np.arange(len(all_locations))
    width = 0.35
    plt.bar(x_pos - width/2, location_comparison['Raw'], width, label='Raw', alpha=0.8)
    plt.bar(x_pos + width/2, location_comparison['Cleaned'], width, label='Cleaned', alpha=0.8)
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.title('Location Distribution: Raw vs Cleaned Data')
    plt.xticks(x_pos, all_locations, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "location_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Payment Methods Donut
    plt.figure(figsize=(8, 8))
    payment_counts_clean = clean_df['Payment Method'].value_counts()
    wedges, texts, autotexts = plt.pie(
        payment_counts_clean, labels=payment_counts_clean.index, 
        autopct='%1.1f%%', startangle=140
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Payment Methods Distribution (Cleaned Data)')
    plt.savefig(os.path.join(save_dir, "payment_methods_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Calculation Consistency
    plt.figure(figsize=(8, 6))
    plt.bar(['Raw Data', 'Cleaned Data'], 
            [consistency_counts['raw'], consistency_counts['cleaned']], 
            color=['#d9534f', '#5cb85c'])
    plt.ylabel("Number of Inconsistent Rows")
    plt.title("Calculation Consistency: Total ≠ Quantity × Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calculation_consistency.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Top Items
    plt.figure(figsize=(10, 8))
    top_items_clean = clean_df['Item'].value_counts().head(10)
    plt.barh(range(len(top_items_clean)), top_items_clean.values)
    plt.yticks(range(len(top_items_clean)), top_items_clean.index)
    plt.xlabel('Count')
    plt.title('Top 10 Items (Cleaned Data)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_items.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Save data quality summary as text file
    raw_rows, raw_cols = raw_df.shape
    clean_rows, clean_cols = clean_df.shape
    rows_removed = raw_rows - clean_rows
    duplicates_removed = check_duplicate_records(raw_df, clean_df)['raw'] - check_duplicate_records(raw_df, clean_df)['cleaned']
    
    summary_content = f"""DATA VALIDATION SUMMARY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
================
Original Dataset:
- Rows: {raw_rows:,}
- Columns: {raw_cols}

Cleaned Dataset:
- Rows: {clean_rows:,}
- Columns: {clean_cols}

DATA QUALITY IMPROVEMENTS:
=========================
- Rows removed: {rows_removed:,}
- Duplicates removed: {duplicates_removed}
- Calculation inconsistencies fixed: {consistency_counts['raw'] - consistency_counts['cleaned']}

CLEANED DATA CHARACTERISTICS:
===========================
- Unique payment methods: {len(clean_df['Payment Method'].unique())}
- Unique locations: {len(clean_df['Location'].unique())}
- Unique items: {len(clean_df['Item'].unique())}

MISSING VALUES SUMMARY:
======================
{missing_comparison.to_string(index=False)}

TOP LOCATIONS (Cleaned Data):
============================
{clean_df['Location'].value_counts().to_string()}

TOP PAYMENT METHODS (Cleaned Data):
==================================
{clean_df['Payment Method'].value_counts().to_string()}

TOP ITEMS (Cleaned Data):
========================
{clean_df['Item'].value_counts().head(10).to_string()}
"""
    
    with open(os.path.join(save_dir, "data_quality_summary.txt"), 'w') as f:
        f.write(summary_content)


def generate_data_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data quality report.
    
    Args:
        raw_df (pd.DataFrame): Raw dataset
        clean_df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Dict: Data quality metrics
    """
    report = {}
    
    # Basic statistics
    report['dataset_shapes'] = {
        'raw': raw_df.shape,
        'cleaned': clean_df.shape
    }
    
    # Missing values
    missing_comparison = compare_missing_values(raw_df, clean_df)
    report['missing_values'] = missing_comparison.to_dict()
    
    # Duplicates
    report['duplicates'] = check_duplicate_records(raw_df, clean_df)
    
    # Calculation consistency
    report['calculation_consistency'] = check_calculation_consistency_comparison(raw_df, clean_df)
    
    # Outliers
    report['outliers'] = check_outliers(raw_df, clean_df)
    
    # Categorical distributions
    report['unique_payment_methods'] = {
        'raw': len(raw_df['Payment Method'].unique()),
        'cleaned': len(clean_df['Payment Method'].unique())
    }
    
    report['unique_locations'] = {
        'raw': len(raw_df['Location'].unique()),
        'cleaned': len(clean_df['Location'].unique())
    }
    
    report['unique_items'] = {
        'raw': len(raw_df['Item'].unique()),
        'cleaned': len(clean_df['Item'].unique())
    }
    
    return report


def validate_cleaned_data(raw_path: str, clean_path: str, 
                         generate_dashboard: bool = True,
                         save_dir: str = "validation_plots") -> Dict:
    """
    Complete validation pipeline for cleaned cafe sales data with unified dashboard.
    
    Args:
        raw_path (str): Path to raw dataset
        clean_path (str): Path to cleaned dataset
        generate_dashboard (bool): Whether to generate the unified dashboard
        save_dir (str): Directory to save individual plots
        
    Returns:
        Dict: Comprehensive validation report
    """
    print("Starting enhanced data validation pipeline...")
    print("=" * 60)
    
    # Load datasets
    raw_df, clean_df = load_datasets(raw_path, clean_path)
    
    # Compare data types
    print("\n1. Data Types Comparison:")
    compare_data_types(raw_df, clean_df)
    
    # Check duplicates
    print("\n2. Duplicate Records Check:")
    duplicate_counts = check_duplicate_records(raw_df, clean_df)
    
    # Compare missing values
    print("\n3. Missing Values Comparison:")
    missing_comparison = compare_missing_values(raw_df, clean_df)
    
    # Check calculation consistency
    print("\n4. Calculation Consistency Check:")
    consistency_counts = check_calculation_consistency_comparison(raw_df, clean_df)
    
    # Check outliers
    print("\n5. Outlier Analysis:")
    outlier_stats = check_outliers(raw_df, clean_df)
    
    # Generate unified dashboard if requested
    if generate_dashboard:
        print("\n6. Generating Unified Validation Dashboard:")
        create_validation_dashboard(raw_df, clean_df, missing_comparison, 
                                   consistency_counts, save_dir)
    
    # Generate comprehensive report
    print("\n7. Generating Data Quality Report:")
    report = generate_data_quality_report(raw_df, clean_df)
    
    print("\n" + "=" * 60)
    print("Enhanced data validation pipeline completed!")
    
    return report


if __name__ == "__main__":
    # Set working directory to the project root (parent of 'src')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_root)
    print("Working directory set to:", os.getcwd())

    # Example usage with enhanced dashboard
    raw_file = "data/dirty_cafe_sales.csv"
    clean_file = "data/cleaned_cafe_sales.csv"
    
    validation_report = validate_cleaned_data(
        raw_file, clean_file, 
        generate_dashboard=True,
        save_dir="validation_plots"
    )
    
print("\n" + "=" * 60)
print("VALIDATION SUMMARY:")
print("=" * 60)
print(f"Raw dataset shape: {validation_report['dataset_shapes']['raw']}")
print(f"Cleaned dataset shape: {validation_report['dataset_shapes']['cleaned']}")
print(f"Duplicates removed: {validation_report['duplicates']['raw'] - validation_report['duplicates']['cleaned']}")
print(f"Calculation inconsistencies fixed: {validation_report['calculation_consistency']['raw'] - validation_report['calculation_consistency']['cleaned']}")
print(f"Outliers removed: {sum(v['raw_outliers'] for v in validation_report['outliers'].values()) - sum(v['clean_outliers'] for v in validation_report['outliers'].values())}")

# Unique values before/after for each key column
for col, label in zip(
    ['unique_payment_methods', 'unique_locations', 'unique_items'],
    ['Payment Methods', 'Locations', 'Items']
):
    print(f"Unique {label}: {validation_report[col]['raw']} → {validation_report[col]['cleaned']}")

# Most improved column (missing values)
missing_vals = validation_report['missing_values']
improvements = [missing_vals['Before Cleaning'][k] - missing_vals['After Cleaning'][k] for k in range(len(missing_vals['Column']))]
most_improved_idx = improvements.index(max(improvements))
most_improved_col = missing_vals['Column'][most_improved_idx]
most_improved_val = max(improvements)
print(f"Most improved column (missing values): {most_improved_col} ({most_improved_val} filled)")

# Data quality score
raw_uniques = [validation_report['unique_payment_methods']['raw'],
               validation_report['unique_locations']['raw'],
               validation_report['unique_items']['raw']]
clean_uniques = [validation_report['unique_payment_methods']['cleaned'],
                 validation_report['unique_locations']['cleaned'],
                 validation_report['unique_items']['cleaned']]
completeness = 1 - (sum(missing_vals['After Cleaning']) / max(1, sum(missing_vals['Before Cleaning'])))
consistency = 1 - (validation_report['calculation_consistency']['cleaned'] / max(1, validation_report['calculation_consistency']['raw']))
uniqueness = 1 - (sum(clean_uniques) / max(1, sum(raw_uniques)))
data_quality_score = (completeness + consistency + uniqueness) / 3
print(f"Data Quality Score (0-100): {data_quality_score * 100:.1f}")
print("=" * 60)