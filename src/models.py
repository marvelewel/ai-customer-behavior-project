"""
Machine Learning Models Module
==============================
Fungsi-fungsi untuk training, evaluasi, dan persistensi model ML.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score
)
import joblib
import os
from typing import Tuple, Dict, List, Any


# ============================================
# CLASSIFICATION MODELS
# ============================================

def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple:
    """
    Split data menjadi training dan testing set.
    
    Parameters:
        X: Features DataFrame
        y: Target Series
        test_size: Proporsi test set (default 0.2)
        random_state: Random seed
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                     model_type: str = 'logistic',
                     **kwargs) -> Any:
    """
    Training model klasifikasi.
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        model_type: 'logistic', 'knn', 'decision_tree', 'random_forest'
        **kwargs: Additional model parameters
        
    Returns:
        Trained model object
    """
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42, **kwargs),
        'knn': KNeighborsClassifier(**kwargs),
        'decision_tree': DecisionTreeClassifier(random_state=42, **kwargs),
        'random_forest': RandomForestClassifier(random_state=42, **kwargs)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = models[model_type]
    model.fit(X_train, y_train)
    print(f"✅ {model_type.replace('_', ' ').title()} model trained")
    
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, 
                   y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluasi performa model klasifikasi.
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary berisi metrik evaluasi
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def get_classification_report(model: Any, X_test: pd.DataFrame, 
                              y_test: pd.Series) -> str:
    """
    Mendapatkan classification report lengkap.
    
    Returns:
        String classification report
    """
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)


def get_confusion_matrix(model: Any, X_test: pd.DataFrame, 
                         y_test: pd.Series) -> np.ndarray:
    """
    Mendapatkan confusion matrix.
    
    Returns:
        Confusion matrix array
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                         cv: int = 5) -> Dict[str, float]:
    """
    Melakukan cross-validation.
    
    Parameters:
        model: Model object
        X: Full features
        y: Full labels
        cv: Number of folds
        
    Returns:
        Dictionary dengan cross-validation scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'cv_scores': scores.tolist()
    }


def compare_classifiers(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Membandingkan performa berbagai classifier.
    
    Returns:
        DataFrame perbandingan performa
    """
    model_types = ['logistic', 'knn', 'decision_tree', 'random_forest']
    results = []
    
    for model_type in model_types:
        model = train_classifier(X_train, y_train, model_type)
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = model_type
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    return comparison_df


# ============================================
# CLUSTERING MODELS
# ============================================

def find_optimal_k(X: pd.DataFrame, k_range: range = range(2, 11)) -> Dict:
    """
    Mencari jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score.
    
    Parameters:
        X: Features untuk clustering
        k_range: Range nilai k yang akan dicoba
        
    Returns:
        Dictionary dengan inertia dan silhouette scores
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'best_k_silhouette': k_range[np.argmax(silhouette_scores)]
    }


def train_kmeans(X: pd.DataFrame, n_clusters: int = 3) -> Tuple[KMeans, np.ndarray]:
    """
    Training K-Means clustering model.
    
    Parameters:
        X: Features untuk clustering
        n_clusters: Jumlah cluster
        
    Returns:
        Tuple (trained KMeans model, cluster labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    print(f"✅ K-Means trained: {n_clusters} clusters, Silhouette Score: {silhouette:.3f}")
    
    return kmeans, labels


def get_cluster_profile(df: pd.DataFrame, cluster_labels: np.ndarray, 
                        features: List[str]) -> pd.DataFrame:
    """
    Membuat profil statistik untuk setiap cluster.
    
    Parameters:
        df: Original DataFrame
        cluster_labels: Array label cluster
        features: List fitur untuk profiling
        
    Returns:
        DataFrame profil cluster
    """
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    profile = df_clustered.groupby('Cluster')[features].agg(['mean', 'median', 'std'])
    
    return profile


def get_cluster_summary(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Membuat ringkasan sederhana per cluster.
    
    Returns:
        DataFrame ringkasan cluster
    """
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    summary = df_clustered.groupby('Cluster').agg({
        'Customer ID': 'count',
        'Total Spend': 'mean',
        'Items Purchased': 'mean',
        'Average Rating': 'mean',
        'Days Since Last Purchase': 'mean'
    }).round(2)
    
    summary.columns = ['Customer_Count', 'Avg_Spend', 'Avg_Items', 'Avg_Rating', 'Avg_Recency']
    
    return summary


# ============================================
# MODEL PERSISTENCE
# ============================================

def save_model(model: Any, filepath: str) -> None:
    """
    Menyimpan model ke file.
    
    Parameters:
        model: Model object
        filepath: Path file tujuan (.pkl)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load model dari file.
    
    Parameters:
        filepath: Path file model (.pkl)
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    print(f"✅ Model loaded from: {filepath}")
    return model
