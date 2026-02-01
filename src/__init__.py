"""
AI Customer Behavior Analysis - Source Package
==============================================

This package contains reusable modules for customer behavior analysis:
- preprocessing: Data cleaning and preprocessing functions
- features: Feature engineering utilities
- models: Machine learning model functions
- ai_logic: AI decision engine for recommendations

Usage:
    from src import load_data, clean_data, create_all_features
    from src.models import train_kmeans, train_classifier
    from src.ai_logic import generate_executive_summary
"""

from .preprocessing import (
    load_data,
    clean_data,
    check_missing_values,
    handle_missing_values,
    fix_data_types,
    encode_categorical,
    scale_features,
    save_processed_data,
    get_data_info
)

from .features import (
    create_all_features,
    create_customer_value_score,
    create_engagement_metrics,
    create_risk_indicators,
    create_membership_features,
    create_demographic_features,
    get_feature_summary
)

from .models import (
    split_data,
    train_classifier,
    train_kmeans,
    evaluate_model,
    find_optimal_k,
    get_cluster_profile,
    get_cluster_summary,
    compare_classifiers,
    save_model,
    load_model
)

from .ai_logic import (
    interpret_cluster,
    generate_cluster_profiles,
    generate_recommendation,
    generate_segment_strategy,
    generate_executive_summary,
    predict_customer_action
)

__version__ = "1.0.0"
__author__ = "AI Customer Behavior Analysis Project"
