import networkx as nx
import pandas as pd
from typing import Dict, List, Optional
import random
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def construct_social_graph(tweets_df: pd.DataFrame, edges_df: Optional[pd.DataFrame] = None, cfg: Dict = None) -> nx.DiGraph:
    """Construct social network graph following the research methodology.
    
    Creates a simplified graph where nodes represent users and edges indicate 
    phishing-related interactions within a time window to simulate real-world spread.
    
    Args:
        tweets_df: DataFrame with tweet data including user_id, timestamp
        edges_df: Optional DataFrame with explicit edge relationships
        cfg: Configuration dictionary
        
    Returns:
        Directed graph representing social network structure
    """
    G = nx.DiGraph()
    cfg = cfg or {}
    
    # Configuration parameters
    time_window_hours = cfg.get("propagation", {}).get("time_window_hours", 24)
    edge_weight_threshold = cfg.get("propagation", {}).get("edge_weight_threshold", 0.01)
    
    # Add nodes from tweet data
    user_tweet_count = tweets_df.groupby('user_id').size().to_dict()
    user_phishing_rate = tweets_df.groupby('user_id')['label'].mean().to_dict()
    
    for user_id, tweet_count in user_tweet_count.items():
        G.add_node(user_id, 
                   tweet_count=tweet_count,
                   phishing_rate=user_phishing_rate.get(user_id, 0.0))
    
    # Add edges from explicit edge data if provided
    if edges_df is not None:
        for _, row in edges_df.iterrows():
            src, dst = row["src"], row["dst"]
            weight = float(row.get("weight", 0.1))
            if weight >= edge_weight_threshold:
                G.add_edge(src, dst, weight=max(0.0, min(1.0, weight)), edge_type="explicit")
    
    # Infer edges from temporal patterns (retweets, replies, etc.)
    if 'timestamp' in tweets_df.columns and 'parent_user_id' in tweets_df.columns:
        tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
        
        # Group by time windows and find interactions
        for _, tweet in tweets_df.iterrows():
            if pd.notna(tweet.get('parent_user_id')):
                src_user = tweet['parent_user_id']
                dst_user = tweet['user_id']
                
                if src_user != dst_user and src_user in G.nodes and dst_user in G.nodes:
                    # Calculate edge weight based on interaction frequency and user characteristics
                    src_phishing_rate = G.nodes[src_user].get('phishing_rate', 0.0)
                    dst_tweet_count = G.nodes[dst_user].get('tweet_count', 1)
                    
                    # Higher weight if source user has higher phishing rate
                    base_weight = 0.1 + 0.4 * src_phishing_rate
                    # Adjust by destination user activity (more active users have higher influence)
                    activity_factor = min(1.0, dst_tweet_count / 10.0)
                    weight = base_weight * activity_factor
                    
                    if weight >= edge_weight_threshold:
                        if G.has_edge(src_user, dst_user):
                            # Strengthen existing edge
                            current_weight = G[src_user][dst_user]['weight']
                            G[src_user][dst_user]['weight'] = min(1.0, current_weight + weight * 0.5)
                        else:
                            G.add_edge(src_user, dst_user, weight=weight, edge_type="inferred")
    
    logger.info(f"Constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def load_graph(edges_csv: str) -> nx.DiGraph:
    """Load graph from CSV file (backward compatibility)."""
    df = pd.read_csv(edges_csv)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        p = float(row.get("weight", 0.1))
        G.add_edge(row["src"], row["dst"], weight=max(0.0, min(1.0, p)))
    return G

def ic_spread(G: nx.DiGraph, seeds: List, samples: int = 100, max_steps: int = 10) -> float:
    """Enhanced Independent Cascade spread estimation.
    
    Args:
        G: Social network graph
        seeds: Initial seed nodes (phishing content sources)
        samples: Number of Monte Carlo samples
        max_steps: Maximum diffusion steps
        
    Returns:
        Expected number of activated (influenced) nodes
    """
    if not seeds or not G.nodes:
        return 0.0
        
    n = 0.0
    for _ in range(samples):
        activated = set(seeds)
        frontier = list(seeds)
        
        for step in range(max_steps):
            if not frontier:
                break
                
            new_frontier = []
            for u in frontier:
                for v in G.successors(u):
                    if v in activated: 
                        continue
                    p = G[u][v].get("weight", 0.1)
                    
                    # Apply influence decay over time/steps
                    decay_factor = 0.9 ** step
                    effective_p = p * decay_factor
                    
                    if random.random() < effective_p:
                        activated.add(v)
                        new_frontier.append(v)
            frontier = new_frontier
            
        n += len(activated)
    return n / samples

def compute_influence_scores(G: nx.DiGraph, user_risk: Dict[str, float]) -> Dict[str, float]:
    """Compute influence scores for nodes in the graph.
    
    Combines network centrality measures with phishing risk scores.
    
    Args:
        G: Social network graph
        user_risk: Dictionary mapping user_id to phishing risk score
        
    Returns:
        Dictionary mapping user_id to influence score
    """
    influence_scores = {}
    
    if not G.nodes:
        return influence_scores
    
    # Compute centrality measures
    try:
        pagerank = nx.pagerank(G, weight='weight')
        betweenness = nx.betweenness_centrality(G, weight='weight')
        out_degree = dict(G.out_degree(weight='weight'))
    except Exception as e:
        logger.warning(f"Failed to compute centrality measures: {e}")
        # Fallback to simple degree centrality
        pagerank = {node: 1.0/len(G.nodes) for node in G.nodes}
        betweenness = {node: 0.0 for node in G.nodes}
        out_degree = dict(G.out_degree())
    
    # Normalize centrality measures
    max_pagerank = max(pagerank.values()) if pagerank.values() else 1.0
    max_betweenness = max(betweenness.values()) if betweenness.values() else 1.0
    max_out_degree = max(out_degree.values()) if out_degree.values() else 1.0
    
    for node in G.nodes:
        # Combine centrality measures
        norm_pagerank = pagerank.get(node, 0) / max_pagerank
        norm_betweenness = betweenness.get(node, 0) / (max_betweenness + 1e-8)
        norm_out_degree = out_degree.get(node, 0) / (max_out_degree + 1e-8)
        
        # Weighted combination of centrality measures
        network_influence = (0.5 * norm_pagerank + 
                           0.3 * norm_betweenness + 
                           0.2 * norm_out_degree)
        
        # Combine with risk score
        risk_score = user_risk.get(node, 0.0)
        
        # Final influence score (higher risk + higher centrality = higher influence)
        influence_scores[node] = network_influence * (1.0 + risk_score)
    
    return influence_scores

def compute_propagation_loss(logits: torch.Tensor, user_ids: List[str], 
                           G: nx.DiGraph, user_risk: Dict[str, float],
                           samples: int = 50) -> torch.Tensor:
    """Compute propagation control loss based on actual graph structure.
    
    This implements the propagation control component of the joint optimization
    objective as described in the research framework.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        user_ids: List of user IDs corresponding to the batch
        G: Social network graph
        user_risk: Dictionary mapping user_id to risk score
        samples: Number of samples for spread estimation
        
    Returns:
        Propagation control loss tensor
    """
    if not user_ids or not G.nodes:
        return torch.tensor(0.0, device=logits.device)
    
    # Convert logits to phishing probabilities
    probs = F.softmax(logits, dim=-1)
    phishing_probs = probs[:, 1] if logits.shape[-1] > 1 else probs[:, 0]
    
    total_propagation_risk = 0.0
    valid_users = 0
    
    for i, (user_id, prob) in enumerate(zip(user_ids, phishing_probs)):
        if user_id not in G.nodes:
            continue
            
        valid_users += 1
        
        # Estimate spread if this user posts phishing content
        try:
            expected_spread = ic_spread(G, [user_id], samples=samples)
            
            # Weight by user's influence and phishing probability
            influence_score = compute_influence_scores(G, user_risk).get(user_id, 0.0)
            
            # Propagation risk = P(phishing) * expected_spread * influence
            prop_risk = prob * expected_spread * (1.0 + influence_score)
            total_propagation_risk += prop_risk
            
        except Exception:
            # Fallback: use simple risk based on node degree
            out_degree = G.out_degree(user_id, weight='weight')
            prop_risk = prob * out_degree * 0.1  # Simple fallback
            total_propagation_risk += prop_risk
    
    if valid_users == 0:
        return torch.tensor(0.0, device=logits.device)
    
    # Average propagation risk across batch
    avg_propagation_risk = total_propagation_risk / valid_users
    
    # Convert to tensor (minimize propagation risk)
    return torch.tensor(avg_propagation_risk, device=logits.device, requires_grad=True)

def greedy_minimize_spread(G: nx.DiGraph, budget: int, risk: Dict, candidates: List, samples: int = 100) -> List:
    """Enhanced greedy algorithm to minimize expected phishing spread through targeted intervention.
    
    This implements the targeted intervention strategy described in the research
    to disrupt high-risk propagation paths.
    
    Args:
        G: Social network graph
        budget: Number of nodes to intervene on
        risk: Dictionary mapping user_id to risk score
        candidates: List of candidate nodes for intervention
        samples: Number of samples for spread estimation
        
    Returns:
        List of nodes selected for intervention
    """
    chosen = []
    if not candidates or budget <= 0:
        return chosen
    
    # Compute influence scores for better candidate ranking
    influence_scores = compute_influence_scores(G, risk)
    
    # Sort candidates by combined risk and influence
    def candidate_priority(u):
        risk_score = risk.get(u, 0.0)
        influence_score = influence_scores.get(u, 0.0)
        return risk_score * (1.0 + influence_score)
    
    cand = sorted(candidates, key=candidate_priority, reverse=True)
    
    # Estimate baseline spread without any intervention
    high_risk_seeds = [u for u in cand[:min(20, len(cand))] if risk.get(u, 0.0) > 0.5]
    base_spread = ic_spread(G, high_risk_seeds, samples) if high_risk_seeds else 0.0
    
    logger.info(f"Baseline expected spread: {base_spread:.2f}")
    
    for iteration in range(min(budget, len(cand))):
        best_gain, best_node = -1, None
        
        for u in cand:
            if u in chosen or u not in G.nodes:
                continue
            
            # Compute spread reduction if we intervene on node u
            # Intervention = removing node from graph temporarily
            G_intervened = G.copy()
            G_intervened.remove_node(u)
            
            # Estimate spread with intervention
            remaining_seeds = [s for s in high_risk_seeds if s != u and s in G_intervened.nodes]
            spread_with_intervention = ic_spread(G_intervened, remaining_seeds, samples) if remaining_seeds else 0.0
            
            # Gain = reduction in spread
            gain = base_spread - spread_with_intervention
            
            # Bias by node characteristics
            risk_bias = 1.0 + 2.0 * risk.get(u, 0.0)  # Higher weight for risky nodes
            influence_bias = 1.0 + influence_scores.get(u, 0.0)  # Higher weight for influential nodes
            
            adjusted_gain = gain * risk_bias * influence_bias
            
            if adjusted_gain > best_gain:
                best_gain, best_node = adjusted_gain, u
        
        if best_node is None:
            break
            
        chosen.append(best_node)
        logger.info(f"Selected intervention node {best_node} with gain {best_gain:.3f}")
        
        # Update base spread for next iteration (greedy approach)
        base_spread = max(0.0, base_spread - best_gain / (len(chosen) + 1))
    
    logger.info(f"Selected {len(chosen)} nodes for intervention: {chosen}")
    return chosen
