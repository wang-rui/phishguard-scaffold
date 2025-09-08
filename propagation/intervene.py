from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def risk_from_logits(user_ids, logits) -> Dict:
    """Enhanced user risk calculation using predicted phishing probability.
    
    Maps user risk using predicted phishing probability of their posts,
    with additional considerations for posting patterns and consistency.
    
    Args:
        user_ids: List of user IDs
        logits: Model output logits
        
    Returns:
        Dictionary mapping user_id to aggregated risk score
    """
    probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    
    # Aggregate risk scores per user (in case multiple posts per user)
    user_risks = defaultdict(list)
    for u, p in zip(user_ids, probs):
        user_risks[u].append(float(p))
    
    # Calculate aggregated risk with multiple strategies
    final_risks = {}
    for user_id, risk_scores in user_risks.items():
        if len(risk_scores) == 1:
            final_risks[user_id] = risk_scores[0]
        else:
            # For users with multiple posts, use sophisticated aggregation
            mean_risk = np.mean(risk_scores)
            max_risk = np.max(risk_scores)
            consistency = 1.0 - np.std(risk_scores)  # Higher consistency = lower std
            
            # Weighted combination favoring consistent high-risk users
            final_risks[user_id] = 0.6 * mean_risk + 0.3 * max_risk + 0.1 * consistency * mean_risk
    
    return final_risks

def compute_user_influence_metrics(G: nx.DiGraph, user_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive influence metrics for users.
    
    Args:
        G: Social network graph
        user_ids: List of user IDs to compute metrics for
        
    Returns:
        Dictionary mapping user_id to metrics dictionary
    """
    metrics = {}
    
    # Compute global centrality measures
    try:
        pagerank = nx.pagerank(G, weight='weight')
        betweenness = nx.betweenness_centrality(G, weight='weight')
        eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except Exception as e:
        logger.warning(f"Failed to compute centrality measures: {e}")
        pagerank = {node: 1.0/len(G.nodes) for node in G.nodes}
        betweenness = {node: 0.0 for node in G.nodes}
        eigenvector = {node: 1.0/len(G.nodes) for node in G.nodes}
    
    for user_id in user_ids:
        if user_id not in G.nodes:
            metrics[user_id] = {'pagerank': 0.0, 'betweenness': 0.0, 'eigenvector': 0.0,
                               'out_degree': 0.0, 'in_degree': 0.0, 'clustering': 0.0}
            continue
            
        # Basic metrics
        out_degree = G.out_degree(user_id, weight='weight')
        in_degree = G.in_degree(user_id, weight='weight')
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(G.to_undirected(), user_id, weight='weight')
        except Exception:
            clustering = 0.0
        
        metrics[user_id] = {
            'pagerank': pagerank.get(user_id, 0.0),
            'betweenness': betweenness.get(user_id, 0.0),
            'eigenvector': eigenvector.get(user_id, 0.0),
            'out_degree': out_degree,
            'in_degree': in_degree,
            'clustering': clustering
        }
    
    return metrics

def advanced_risk_assessment(user_ids: List[str], logits: torch.Tensor, 
                           G: nx.DiGraph, tweets_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Advanced risk assessment combining multiple factors.
    
    This implements a sophisticated risk calculation that considers:
    - Phishing probability from model
    - Network centrality and influence
    - User behavior patterns
    - Temporal activity patterns
    
    Args:
        user_ids: List of user IDs
        logits: Model predictions
        G: Social network graph
        tweets_df: Optional DataFrame with tweet metadata
        
    Returns:
        Dictionary mapping user_id to comprehensive risk score
    """
    # Base risk from model predictions
    base_risks = risk_from_logits(user_ids, logits)
    
    # Network influence metrics
    influence_metrics = compute_user_influence_metrics(G, list(base_risks.keys()))
    
    # Behavioral patterns from tweets (if available)
    behavioral_scores = {}
    if tweets_df is not None and len(tweets_df) > 0:
        for user_id in base_risks.keys():
            user_tweets = tweets_df[tweets_df['user_id'] == user_id]
            if len(user_tweets) == 0:
                behavioral_scores[user_id] = 0.0
                continue
                
            # URL usage patterns (phishing often involves URLs)
            url_ratio = user_tweets['url'].notna().mean() if 'url' in user_tweets.columns else 0.0
            
            # Activity frequency
            if 'timestamp' in user_tweets.columns:
                try:
                    timestamps = pd.to_datetime(user_tweets['timestamp'])
                    time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # hours
                    activity_freq = len(user_tweets) / (time_span + 1)  # tweets per hour
                except Exception:
                    activity_freq = 0.0
            else:
                activity_freq = len(user_tweets) / 24  # assume 24h window
            
            # Behavioral risk score
            behavioral_scores[user_id] = 0.5 * url_ratio + 0.3 * min(1.0, activity_freq / 5.0) + 0.2 * len(user_tweets) / 100.0
    else:
        behavioral_scores = {user_id: 0.0 for user_id in base_risks.keys()}
    
    # Combine all factors into final risk score
    final_risks = {}
    for user_id in base_risks.keys():
        base_risk = base_risks[user_id]
        metrics = influence_metrics[user_id]
        behavioral_risk = behavioral_scores[user_id]
        
        # Network influence score
        influence_score = (0.4 * metrics['pagerank'] + 
                         0.3 * metrics['betweenness'] + 
                         0.2 * metrics['eigenvector'] + 
                         0.1 * min(1.0, metrics['out_degree'] / 10.0))
        
        # Final risk combines base model prediction with network position and behavior
        final_risk = (0.5 * base_risk +                    # Model prediction weight
                     0.3 * influence_score +               # Network influence weight
                     0.2 * behavioral_risk)                # Behavioral pattern weight
        
        # Boost risk for users with high clustering (tight communities can spread faster)
        community_boost = 1.0 + 0.2 * metrics['clustering']
        final_risk *= community_boost
        
        final_risks[user_id] = min(1.0, final_risk)  # Cap at 1.0
    
    return final_risks

def pick_candidates(df_users: pd.Series, topk: int = 200, G: Optional[nx.DiGraph] = None, 
                   risk_scores: Optional[Dict[str, float]] = None) -> List:
    """Enhanced candidate selection for intervention.
    
    Selects candidate nodes for intervention based on multiple criteria:
    - Activity frequency (original approach)
    - Network centrality (if graph provided)
    - Risk scores (if provided)
    
    Args:
        df_users: Series with user IDs
        topk: Number of top candidates to return
        G: Optional social network graph
        risk_scores: Optional risk scores per user
        
    Returns:
        List of candidate user IDs for intervention
    """
    # Get activity-based candidates (most active users)
    activity_counts = df_users.value_counts()
    
    if G is None and risk_scores is None:
        # Fallback to original activity-based approach
        return list(activity_counts.head(topk).index)
    
    # Enhanced candidate selection
    candidate_scores = {}
    
    for user_id in activity_counts.index:
        score = 0.0
        
        # Activity component (normalized)
        activity_score = activity_counts[user_id] / activity_counts.iloc[0] if len(activity_counts) > 0 else 0.0
        score += 0.3 * activity_score
        
        # Network centrality component
        if G is not None and user_id in G.nodes:
            try:
                # Simple centrality approximation (degree centrality is fast)
                degree_centrality = G.degree(user_id, weight='weight') / (len(G.nodes) - 1) if len(G.nodes) > 1 else 0.0
                score += 0.4 * degree_centrality
            except Exception:
                pass
        
        # Risk score component
        if risk_scores is not None and user_id in risk_scores:
            score += 0.3 * risk_scores[user_id]
        
        candidate_scores[user_id] = score
    
    # Sort by combined score and return top candidates
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return [user_id for user_id, _ in sorted_candidates[:topk]]

def evaluate_intervention_impact(G: nx.DiGraph, intervention_nodes: List[str], 
                               risk_scores: Dict[str, float], samples: int = 100) -> Dict[str, float]:
    """Evaluate the impact of intervention on propagation control.
    
    Args:
        G: Social network graph
        intervention_nodes: List of nodes selected for intervention
        risk_scores: Risk scores for all users
        samples: Number of simulation samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    from .graph import ic_spread
    
    # High-risk users as potential seeds
    high_risk_users = [u for u, risk in risk_scores.items() if risk > 0.5 and u in G.nodes]
    
    # Baseline spread without intervention
    baseline_spread = ic_spread(G, high_risk_users, samples) if high_risk_users else 0.0
    
    # Spread with intervention (remove intervention nodes)
    G_intervened = G.copy()
    for node in intervention_nodes:
        if node in G_intervened.nodes:
            G_intervened.remove_node(node)
    
    # Remaining high-risk users after intervention
    remaining_seeds = [u for u in high_risk_users if u in G_intervened.nodes]
    intervened_spread = ic_spread(G_intervened, remaining_seeds, samples) if remaining_seeds else 0.0
    
    # Calculate impact metrics
    spread_reduction = baseline_spread - intervened_spread
    relative_reduction = spread_reduction / baseline_spread if baseline_spread > 0 else 0.0
    
    # Cost-effectiveness (spread reduction per intervention node)
    cost_effectiveness = spread_reduction / len(intervention_nodes) if len(intervention_nodes) > 0 else 0.0
    
    return {
        'baseline_spread': baseline_spread,
        'intervened_spread': intervened_spread,
        'spread_reduction': spread_reduction,
        'relative_reduction': relative_reduction,
        'cost_effectiveness': cost_effectiveness,
        'num_interventions': len(intervention_nodes)
    }
