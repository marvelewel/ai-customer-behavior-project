"""
AI Decision Engine Module
=========================
Fungsi-fungsi untuk interpretasi hasil ML dan menghasilkan rekomendasi berbasis AI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


# ============================================
# CLUSTER INTERPRETATION
# ============================================

def interpret_cluster(cluster_id: int, cluster_stats: Dict) -> Dict[str, Any]:
    """
    Menginterpretasikan karakteristik cluster secara otomatis.
    
    Parameters:
        cluster_id: ID cluster (0, 1, 2, ...)
        cluster_stats: Dictionary berisi statistik cluster
        
    Returns:
        Dictionary interpretasi cluster
    """
    stats = cluster_stats
    
    # Determine cluster characteristics
    interpretation = {
        'cluster_id': cluster_id,
        'size': stats.get('customer_count', 0),
        'characteristics': [],
        'persona': '',
        'priority': ''
    }
    
    # Analyze spending behavior
    avg_spend = stats.get('avg_spend', 0)
    if avg_spend > 1200:
        interpretation['characteristics'].append('High Spender')
        interpretation['priority'] = 'High Value'
    elif avg_spend > 700:
        interpretation['characteristics'].append('Medium Spender')
        interpretation['priority'] = 'Potential Value'
    else:
        interpretation['characteristics'].append('Low Spender')
        interpretation['priority'] = 'Retention Focus'
    
    # Analyze recency
    avg_recency = stats.get('avg_recency', 30)
    if avg_recency < 20:
        interpretation['characteristics'].append('Highly Active')
    elif avg_recency < 40:
        interpretation['characteristics'].append('Moderately Active')
    else:
        interpretation['characteristics'].append('At Risk - Inactive')
    
    # Analyze rating
    avg_rating = stats.get('avg_rating', 4.0)
    if avg_rating >= 4.5:
        interpretation['characteristics'].append('Very Satisfied')
    elif avg_rating >= 4.0:
        interpretation['characteristics'].append('Satisfied')
    else:
        interpretation['characteristics'].append('Needs Attention')
    
    # Generate persona name
    if 'High Spender' in interpretation['characteristics'] and 'Highly Active' in interpretation['characteristics']:
        interpretation['persona'] = 'VIP Champions'
    elif 'High Spender' in interpretation['characteristics']:
        interpretation['persona'] = 'Loyal Spenders'
    elif 'At Risk - Inactive' in interpretation['characteristics']:
        interpretation['persona'] = 'Hibernating Customers'
    elif 'Low Spender' in interpretation['characteristics']:
        interpretation['persona'] = 'Budget Conscious'
    else:
        interpretation['persona'] = 'Regular Customers'
    
    return interpretation


def generate_cluster_profiles(cluster_summary: pd.DataFrame) -> List[Dict]:
    """
    Generate profil lengkap untuk semua cluster.
    
    Parameters:
        cluster_summary: DataFrame dengan ringkasan statistik per cluster
        
    Returns:
        List of cluster profile dictionaries
    """
    profiles = []
    
    for cluster_id in cluster_summary.index:
        row = cluster_summary.loc[cluster_id]
        
        stats = {
            'customer_count': int(row.get('Customer_Count', 0)),
            'avg_spend': float(row.get('Avg_Spend', 0)),
            'avg_items': float(row.get('Avg_Items', 0)),
            'avg_rating': float(row.get('Avg_Rating', 0)),
            'avg_recency': float(row.get('Avg_Recency', 0))
        }
        
        profile = interpret_cluster(cluster_id, stats)
        profiles.append(profile)
    
    return profiles


# ============================================
# RECOMMENDATION ENGINE
# ============================================

def generate_recommendation(customer_data: Dict, 
                           cluster_info: Optional[Dict] = None,
                           prediction: Optional[str] = None) -> Dict[str, Any]:
    """
    Menghasilkan rekomendasi aksi untuk pelanggan berdasarkan data dan prediksi.
    
    Parameters:
        customer_data: Dictionary data pelanggan
        cluster_info: Optional cluster interpretation
        prediction: Optional prediction result (e.g. satisfaction level)
        
    Returns:
        Dictionary berisi rekomendasi
    """
    recommendation = {
        'customer_id': customer_data.get('Customer ID'),
        'risk_level': 'Medium',
        'recommended_actions': [],
        'priority': 'Normal',
        'expected_impact': '',
        'reasoning': []
    }
    
    # Analyze customer data
    total_spend = customer_data.get('Total Spend', 0)
    days_since_purchase = customer_data.get('Days Since Last Purchase', 30)
    avg_rating = customer_data.get('Average Rating', 4.0)
    membership = customer_data.get('Membership Type', 'Bronze')
    satisfaction = prediction or customer_data.get('Satisfaction Level', 'Neutral')
    
    # Rule-based recommendation logic
    
    # High-value customer at risk
    if total_spend > 1000 and days_since_purchase > 40:
        recommendation['risk_level'] = 'High'
        recommendation['priority'] = 'Urgent'
        recommendation['recommended_actions'].append('🚨 Immediate re-engagement campaign')
        recommendation['recommended_actions'].append('📞 Personal outreach from account manager')
        recommendation['recommended_actions'].append('🎁 Exclusive loyalty reward offer')
        recommendation['reasoning'].append('High-value customer showing signs of churn')
    
    # Unsatisfied customer
    if satisfaction == 'Unsatisfied':
        recommendation['risk_level'] = 'High'
        recommendation['recommended_actions'].append('📋 Conduct satisfaction survey')
        recommendation['recommended_actions'].append('🔧 Identify and resolve pain points')
        recommendation['recommended_actions'].append('💰 Offer compensation/discount')
        recommendation['reasoning'].append('Customer expressed dissatisfaction')
    
    # Inactive but previously engaged
    if days_since_purchase > 50:
        recommendation['recommended_actions'].append('📧 Win-back email campaign')
        recommendation['recommended_actions'].append('⏰ Limited-time return offer')
        recommendation['reasoning'].append('Customer has been inactive for extended period')
    
    # Upgrade potential (Silver with high spending)
    if membership == 'Silver' and total_spend > 800:
        recommendation['recommended_actions'].append('⬆️ Offer Gold membership upgrade')
        recommendation['recommended_actions'].append('✨ Highlight premium benefits')
        recommendation['reasoning'].append('Customer shows potential for tier upgrade')
    
    # Loyal customer maintenance
    if avg_rating >= 4.5 and satisfaction == 'Satisfied':
        recommendation['risk_level'] = 'Low'
        recommendation['priority'] = 'Maintain'
        recommendation['recommended_actions'].append('🌟 Enroll in referral program')
        recommendation['recommended_actions'].append('💎 Early access to new products')
        recommendation['reasoning'].append('Satisfied customer - focus on retention and advocacy')
    
    # New customer nurturing (low items purchased)
    if customer_data.get('Items Purchased', 10) < 10:
        recommendation['recommended_actions'].append('📚 Send product education content')
        recommendation['recommended_actions'].append('🎯 Personalized product recommendations')
        recommendation['reasoning'].append('Customer still exploring product catalog')
    
    # Set expected impact
    if recommendation['priority'] == 'Urgent':
        recommendation['expected_impact'] = 'Prevent potential churn of high-value customer'
    elif recommendation['priority'] == 'Maintain':
        recommendation['expected_impact'] = 'Strengthen loyalty and generate referrals'
    else:
        recommendation['expected_impact'] = 'Increase engagement and customer lifetime value'
    
    # Remove duplicates
    recommendation['recommended_actions'] = list(set(recommendation['recommended_actions']))
    
    return recommendation


def generate_segment_strategy(cluster_profiles: List[Dict]) -> List[Dict]:
    """
    Menghasilkan strategi bisnis untuk setiap segmen pelanggan.
    
    Parameters:
        cluster_profiles: List profil cluster dari generate_cluster_profiles()
        
    Returns:
        List strategi per segmen
    """
    strategies = []
    
    for profile in cluster_profiles:
        strategy = {
            'segment': profile['persona'],
            'cluster_id': profile['cluster_id'],
            'customer_count': profile['size'],
            'marketing_strategy': '',
            'retention_strategy': '',
            'upsell_opportunity': '',
            'resource_allocation': ''
        }
        
        persona = profile['persona']
        
        if persona == 'VIP Champions':
            strategy['marketing_strategy'] = 'Exclusive VIP experiences and early access'
            strategy['retention_strategy'] = 'Dedicated account manager, loyalty rewards'
            strategy['upsell_opportunity'] = 'Premium services, bundled packages'
            strategy['resource_allocation'] = 'High - Personalized attention'
            
        elif persona == 'Loyal Spenders':
            strategy['marketing_strategy'] = 'Personalized recommendations, loyalty program'
            strategy['retention_strategy'] = 'Regular engagement, appreciation events'
            strategy['upsell_opportunity'] = 'Tier upgrades, complementary products'
            strategy['resource_allocation'] = 'Medium-High - Nurture relationship'
            
        elif persona == 'Hibernating Customers':
            strategy['marketing_strategy'] = 'Win-back campaigns, special return offers'
            strategy['retention_strategy'] = 'Identify churn reasons, address pain points'
            strategy['upsell_opportunity'] = 'Limited - Focus on reactivation first'
            strategy['resource_allocation'] = 'Medium - Strategic intervention'
            
        elif persona == 'Budget Conscious':
            strategy['marketing_strategy'] = 'Value-focused promotions, discount alerts'
            strategy['retention_strategy'] = 'Consistent value delivery, bundle deals'
            strategy['upsell_opportunity'] = 'Cross-sell affordable add-ons'
            strategy['resource_allocation'] = 'Low-Medium - Efficient mass communication'
            
        else:  # Regular Customers
            strategy['marketing_strategy'] = 'General promotions, seasonal campaigns'
            strategy['retention_strategy'] = 'Standard loyalty program benefits'
            strategy['upsell_opportunity'] = 'Product recommendations based on history'
            strategy['resource_allocation'] = 'Medium - Balanced approach'
        
        strategies.append(strategy)
    
    return strategies


# ============================================
# INSIGHT GENERATION
# ============================================

def generate_executive_summary(df: pd.DataFrame, 
                               cluster_profiles: List[Dict],
                               model_performance: Dict) -> str:
    """
    Menghasilkan executive summary untuk stakeholders.
    
    Parameters:
        df: Full customer DataFrame
        cluster_profiles: Cluster interpretation results
        model_performance: Dictionary metrik model
        
    Returns:
        Formatted executive summary string
    """
    total_customers = len(df)
    
    # Calculate key metrics
    total_revenue = df['Total Spend'].sum()
    avg_spend = df['Total Spend'].mean()
    satisfaction_dist = df['Satisfaction Level'].value_counts(normalize=True) * 100
    
    # Find high-risk segment
    high_risk_profiles = [p for p in cluster_profiles if 'At Risk' in str(p.get('characteristics', []))]
    
    summary = f"""
═══════════════════════════════════════════════════════════════════
                    EXECUTIVE SUMMARY
            AI-Driven Customer Behavior Analysis
═══════════════════════════════════════════════════════════════════

📊 DATASET OVERVIEW
───────────────────
• Total Customers Analyzed: {total_customers:,}
• Total Revenue: ${total_revenue:,.2f}
• Average Spend per Customer: ${avg_spend:,.2f}

📈 SATISFACTION DISTRIBUTION
───────────────────────────
"""
    
    for level, pct in satisfaction_dist.items():
        emoji = '😊' if level == 'Satisfied' else ('😐' if level == 'Neutral' else '😞')
        summary += f"• {emoji} {level}: {pct:.1f}%\n"
    
    summary += f"""
🎯 CUSTOMER SEGMENTS IDENTIFIED
───────────────────────────────
"""
    
    for profile in cluster_profiles:
        summary += f"• Cluster {profile['cluster_id']}: {profile['persona']} ({profile['size']} customers)\n"
        summary += f"  Characteristics: {', '.join(profile['characteristics'])}\n"
    
    if high_risk_profiles:
        summary += f"""
⚠️ RISK ALERT
─────────────
{len(high_risk_profiles)} segment(s) identified as at-risk requiring immediate attention.
"""
    
    summary += f"""
🤖 MODEL PERFORMANCE
────────────────────
• Accuracy: {model_performance.get('accuracy', 0)*100:.1f}%
• Precision: {model_performance.get('precision', 0)*100:.1f}%
• Recall: {model_performance.get('recall', 0)*100:.1f}%
• F1-Score: {model_performance.get('f1_score', 0)*100:.1f}%

═══════════════════════════════════════════════════════════════════
                    RECOMMENDED ACTIONS
═══════════════════════════════════════════════════════════════════

1. 🚀 IMMEDIATE: Re-engage hibernating customers with win-back campaign
2. 💎 SHORT-TERM: Upgrade eligible Silver members to Gold tier
3. 🌟 ONGOING: Leverage VIP Champions for referral program
4. 📊 CONTINUOUS: Monitor satisfaction trends and adjust strategies

═══════════════════════════════════════════════════════════════════
"""
    
    return summary


def predict_customer_action(customer_id: int, df: pd.DataFrame,
                            cluster_model: Any, classifier_model: Any,
                            feature_columns: List[str]) -> Dict[str, Any]:
    """
    Prediksi aksi yang direkomendasikan untuk customer tertentu.
    
    Parameters:
        customer_id: ID pelanggan
        df: Full DataFrame
        cluster_model: Trained clustering model
        classifier_model: Trained classification model
        feature_columns: List of feature columns used in models
        
    Returns:
        Complete action recommendation dictionary
    """
    # Get customer data
    customer_row = df[df['Customer ID'] == customer_id]
    
    if customer_row.empty:
        return {'error': f'Customer ID {customer_id} not found'}
    
    customer_data = customer_row.iloc[0].to_dict()
    
    # Get cluster assignment
    customer_features = customer_row[feature_columns].values
    cluster_id = cluster_model.predict(customer_features)[0]
    
    # Get satisfaction prediction
    prediction = classifier_model.predict(customer_features)[0]
    
    # Generate recommendation
    recommendation = generate_recommendation(
        customer_data=customer_data,
        prediction=prediction
    )
    
    recommendation['predicted_satisfaction'] = prediction
    recommendation['cluster'] = int(cluster_id)
    
    return recommendation
