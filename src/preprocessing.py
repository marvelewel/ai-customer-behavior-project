"""
Preprocessing Module
====================
Fungsi-fungsi untuk cleaning dan preprocessing data pelanggan e-commerce.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from typing import Tuple, List, Optional
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV.
    
    Parameters:
        filepath: Path ke file CSV
        
    Returns:
        DataFrame dengan data mentah
    """
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Mendapatkan informasi dasar tentang dataset.
    
    Parameters:
        df: DataFrame input
        
    Returns:
        Dictionary berisi informasi dataset
    """
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return info


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengecek missing values di setiap kolom.
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame berisi ringkasan missing values
    """
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).values
    })
    missing = missing[missing['missing_count'] > 0].sort_values('missing_count', ascending=False)
    return missing


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mode') -> pd.DataFrame:
    """
    Menangani missing values dalam dataset.
    
    Parameters:
        df: DataFrame input
        strategy: 'mode' untuk kategorikal, 'mean'/'median' untuk numerik
        
    Returns:
        DataFrame tanpa missing values
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                # Untuk kolom kategorikal, gunakan mode
                mode_value = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_value)
                print(f"  → {col}: filled with mode '{mode_value}'")
            else:
                # Untuk kolom numerik
                if strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif strategy == 'median':
                    fill_value = df_clean[col].median()
                else:
                    fill_value = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(fill_value)
                # Format output based on value type
                if isinstance(fill_value, (int, float)):
                    print(f"  → {col}: filled with {strategy} = {fill_value:.2f}")
                else:
                    print(f"  → {col}: filled with {strategy} = {fill_value}")
    
    print(f"✅ Missing values handled")
    return df_clean


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memperbaiki tipe data kolom.
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame dengan tipe data yang benar
    """
    df_fixed = df.copy()
    
    # Convert boolean strings to actual boolean
    if 'Discount Applied' in df_fixed.columns:
        df_fixed['Discount Applied'] = df_fixed['Discount Applied'].map({
            'TRUE': True, 'FALSE': False, True: True, False: False
        })
    
    # Ensure numeric columns are numeric
    numeric_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    for col in numeric_cols:
        if col in df_fixed.columns:
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
    
    print(f"✅ Data types fixed")
    return df_fixed


def encode_categorical(df: pd.DataFrame, columns: List[str], 
                       method: str = 'label') -> Tuple[pd.DataFrame, dict]:
    """
    Encoding kolom kategorikal.
    
    Parameters:
        df: DataFrame input
        columns: List kolom yang akan di-encode
        method: 'label' untuk LabelEncoder, 'onehot' untuk OneHotEncoder
        
    Returns:
        Tuple (DataFrame hasil encoding, dictionary encoder)
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        if col not in df_encoded.columns:
            continue
            
        if method == 'label':
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            print(f"  → {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
        elif method == 'onehot':
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            encoders[col] = list(dummies.columns)
            print(f"  → {col}: created {len(dummies.columns)} dummy columns")
    
    print(f"✅ Categorical encoding completed ({method})")
    return df_encoded, encoders


def scale_features(df: pd.DataFrame, columns: List[str], 
                   method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scaling fitur numerik.
    
    Parameters:
        df: DataFrame input
        columns: List kolom yang akan di-scale
        method: 'standard' untuk StandardScaler, 'minmax' untuk MinMaxScaler
        
    Returns:
        Tuple (DataFrame dengan fitur ter-scale, scaler object)
    """
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Filter only existing columns
    valid_columns = [col for col in columns if col in df_scaled.columns]
    
    if valid_columns:
        df_scaled[[f'{col}_scaled' for col in valid_columns]] = scaler.fit_transform(df_scaled[valid_columns])
        print(f"✅ Features scaled ({method}): {valid_columns}")
    
    return df_scaled, scaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline lengkap untuk data cleaning.
    
    Parameters:
        df: DataFrame mentah
        
    Returns:
        DataFrame bersih siap untuk analisis
    """
    print("\n🔧 Starting Data Cleaning Pipeline...")
    print("=" * 50)
    
    # Step 1: Fix data types
    df_clean = fix_data_types(df)
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df_clean)
    
    # Step 3: Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed = initial_rows - len(df_clean)
    if removed > 0:
        print(f"  → Removed {removed} duplicate rows")
    
    print("=" * 50)
    print(f"✅ Cleaning complete: {len(df_clean)} rows retained")
    
    return df_clean


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Menyimpan data yang sudah diproses.
    
    Parameters:
        df: DataFrame yang akan disimpan
        filepath: Path tujuan file CSV
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Data saved to: {filepath}")
