"""
Feature Engineering Module
==========================
Fungsi-fungsi untuk membuat fitur baru dari data pelanggan.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_customer_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat Customer Value Score berdasarkan spending dan aktivitas.
    
    Formula: CVS = (Normalized Spend * 0.4) + (Normalized Items * 0.3) + (Normalized Rating * 0.3)
    
    Parameters:
        df: DataFrame dengan kolom Total Spend, Items Purchased, Average Rating
        
    Returns:
        DataFrame dengan kolom Customer_Value_Score
    """
    df_feat = df.copy()
    
    # Normalize each component (0-1 scale)
    spend_norm = (df_feat['Total Spend'] - df_feat['Total Spend'].min()) / \
                 (df_feat['Total Spend'].max() - df_feat['Total Spend'].min())
    
    items_norm = (df_feat['Items Purchased'] - df_feat['Items Purchased'].min()) / \
                 (df_feat['Items Purchased'].max() - df_feat['Items Purchased'].min())
    
    rating_norm = (df_feat['Average Rating'] - df_feat['Average Rating'].min()) / \
                  (df_feat['Average Rating'].max() - df_feat['Average Rating'].min())
    
    # Calculate weighted score
    df_feat['Customer_Value_Score'] = (spend_norm * 0.4) + (items_norm * 0.3) + (rating_norm * 0.3)
    
    print(f"✅ Created Customer_Value_Score (mean: {df_feat['Customer_Value_Score'].mean():.3f})")
    return df_feat


def create_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat metrik engagement pelanggan.
    
    Metrics:
    - Recency_Category: Recent (<20 days), Medium (20-40), Inactive (>40)
    - Spend_Per_Item: Average spending per item
    - Is_Active: Boolean based on recency
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame dengan engagement metrics
    """
    df_feat = df.copy()
    
    # Recency Category
    def categorize_recency(days):
        if days < 20:
            return 'Recent'
        elif days <= 40:
            return 'Medium'
        else:
            return 'Inactive'
    
    df_feat['Recency_Category'] = df_feat['Days Since Last Purchase'].apply(categorize_recency)
    
    # Spend per item
    df_feat['Spend_Per_Item'] = df_feat['Total Spend'] / df_feat['Items Purchased']
    
    # Is Active (purchased within 30 days)
    df_feat['Is_Active'] = df_feat['Days Since Last Purchase'] <= 30
    
    print(f"✅ Created engagement metrics:")
    print(f"   → Recency distribution: {df_feat['Recency_Category'].value_counts().to_dict()}")
    print(f"   → Active customers: {df_feat['Is_Active'].sum()} ({df_feat['Is_Active'].mean()*100:.1f}%)")
    
    return df_feat


def create_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat indikator risiko churn pelanggan.
    
    Indicators:
    - Churn_Risk_Score: 0-1 score indicating churn probability
    - Risk_Level: Low, Medium, High
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame dengan risk indicators
    """
    df_feat = df.copy()
    
    # Churn risk factors (normalized 0-1)
    # Higher days since purchase = higher risk
    recency_risk = (df_feat['Days Since Last Purchase'] - df_feat['Days Since Last Purchase'].min()) / \
                   (df_feat['Days Since Last Purchase'].max() - df_feat['Days Since Last Purchase'].min())
    
    # Lower rating = higher risk
    rating_risk = 1 - (df_feat['Average Rating'] - df_feat['Average Rating'].min()) / \
                      (df_feat['Average Rating'].max() - df_feat['Average Rating'].min())
    
    # Lower spend = higher risk (inverse)
    spend_risk = 1 - (df_feat['Total Spend'] - df_feat['Total Spend'].min()) / \
                     (df_feat['Total Spend'].max() - df_feat['Total Spend'].min())
    
    # Satisfaction risk mapping
    if 'Satisfaction Level' in df_feat.columns:
        satisfaction_map = {'Satisfied': 0, 'Neutral': 0.5, 'Unsatisfied': 1}
        satisfaction_risk = df_feat['Satisfaction Level'].map(satisfaction_map).fillna(0.5)
    else:
        satisfaction_risk = 0.5
    
    # Weighted churn risk score
    df_feat['Churn_Risk_Score'] = (
        recency_risk * 0.35 + 
        rating_risk * 0.25 + 
        spend_risk * 0.15 + 
        satisfaction_risk * 0.25
    )
    
    # Risk level categories
    def categorize_risk(score):
        if score < 0.4:
            return 'Low'
        elif score < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    df_feat['Risk_Level'] = df_feat['Churn_Risk_Score'].apply(categorize_risk)
    
    print(f"✅ Created risk indicators:")
    print(f"   → Risk distribution: {df_feat['Risk_Level'].value_counts().to_dict()}")
    print(f"   → Avg churn risk score: {df_feat['Churn_Risk_Score'].mean():.3f}")
    
    return df_feat


def create_membership_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur berbasis membership.
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame dengan membership features
    """
    df_feat = df.copy()
    
    # Membership value (numeric)
    membership_value = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df_feat['Membership_Value'] = df_feat['Membership Type'].map(membership_value)
    
    # Is Premium (Gold membership)
    df_feat['Is_Premium'] = df_feat['Membership Type'] == 'Gold'
    
    # Spending efficiency (spend relative to membership tier)
    tier_avg_spend = df_feat.groupby('Membership Type')['Total Spend'].transform('mean')
    df_feat['Spend_Efficiency'] = df_feat['Total Spend'] / tier_avg_spend
    
    print(f"✅ Created membership features")
    return df_feat


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur demografis.
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame dengan demographic features
    """
    df_feat = df.copy()
    
    # Age groups
    def categorize_age(age):
        if age < 30:
            return 'Young'
        elif age < 40:
            return 'Middle'
        else:
            return 'Senior'
    
    df_feat['Age_Group'] = df_feat['Age'].apply(categorize_age)
    
    # City tier (based on typical market size)
    city_tier = {
        'New York': 1,
        'Los Angeles': 1,
        'Chicago': 2,
        'San Francisco': 1,
        'Houston': 2,
        'Miami': 2
    }
    df_feat['City_Tier'] = df_feat['City'].map(city_tier).fillna(3)
    
    print(f"✅ Created demographic features")
    return df_feat


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline untuk membuat semua fitur engineering.
    
    Parameters:
        df: DataFrame input (sudah cleaned)
        
    Returns:
        DataFrame dengan semua fitur baru
    """
    print("\n🛠️ Starting Feature Engineering Pipeline...")
    print("=" * 50)
    
    df_feat = df.copy()
    
    # Apply all feature functions
    df_feat = create_customer_value_score(df_feat)
    df_feat = create_engagement_metrics(df_feat)
    df_feat = create_risk_indicators(df_feat)
    df_feat = create_membership_features(df_feat)
    df_feat = create_demographic_features(df_feat)
    
    print("=" * 50)
    print(f"✅ Feature engineering complete: {len(df_feat.columns)} total columns")
    
    return df_feat


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mendapatkan ringkasan statistik untuk semua fitur numerik.
    
    Parameters:
        df: DataFrame input
        
    Returns:
        DataFrame summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe().T
    summary['missing'] = df[numeric_cols].isnull().sum()
    return summary
