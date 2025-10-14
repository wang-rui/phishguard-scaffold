import networkx as nx
import pandas as pd
from typing import Dict, List, Optional
import random
import torch
import torch.nn.functional as F
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.constants import (
    INFLUENCE_DECAY_FACTOR,
    MIN_EDGE_WEIGHT,
    MAX_EDGE_WEIGHT,
    HIGH_RISK_SEED_LIMIT,
    HIGH_RISK_THRESHOLD,
    RISK_BIAS_MULTIPLIER,
    PAGERANK_WEIGHT,
    BETWEENNESS_WEIGHT,
    OUT_DEGREE_WEIGHT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_social_graph(
    tweets_df: pd.DataFrame, edges_df: Optional[pd.DataFrame] = None, cfg: Dict = None
) -> nx.DiGraph:
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
    edge_weight_threshold = cfg.get("propagation", {}).get(
        "edge_weight_threshold", 0.01
    )

    # Add nodes from tweet data
    user_tweet_count = tweets_df.groupby("user_id").size().to_dict()
    user_phishing_rate = tweets_df.groupby("user_id")["label"].mean().to_dict()

    for user_id, tweet_count in user_tweet_count.items():
        G.add_node(
            user_id,
            tweet_count=tweet_count,
            phishing_rate=user_phishing_rate.get(user_id, 0.0),
        )

    # Add edges from explicit edge data if provided
    if edges_df is not None:
        for _, row in edges_df.iterrows():
            src, dst = row["src"], row["dst"]
            weight = float(row.get("weight", 0.1))
            if weight >= edge_weight_threshold:
                G.add_edge(
                    src,
                    dst,
                    weight=max(MIN_EDGE_WEIGHT, min(MAX_EDGE_WEIGHT, weight)),
                    edge_type="explicit",
                )

    # Infer edges from temporal patterns (retweets, replies, etc.)
    if "timestamp" in tweets_df.columns and "parent_user_id" in tweets_df.columns:
        tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"])

        # Group by time windows and find interactions
        for _, tweet in tweets_df.iterrows():
            if pd.notna(tweet.get("parent_user_id")):
                src_user = tweet["parent_user_id"]
                dst_user = tweet["user_id"]

                if src_user != dst_user and src_user in G.nodes and dst_user in G.nodes:
                    # Calculate edge weight based on interaction frequency and user characteristics
                    src_phishing_rate = G.nodes[src_user].get("phishing_rate", 0.0)
                    dst_tweet_count = G.nodes[dst_user].get("tweet_count", 1)

                    # Higher weight if source user has higher phishing rate
                    base_weight = 0.1 + 0.4 * src_phishing_rate
                    # Adjust by destination user activity (more active users have higher influence)
                    activity_factor = min(1.0, dst_tweet_count / 10.0)
                    weight = base_weight * activity_factor

                    if weight >= edge_weight_threshold:
                        if G.has_edge(src_user, dst_user):
                            # Strengthen existing edge
                            current_weight = G[src_user][dst_user]["weight"]
                            G[src_user][dst_user]["weight"] = min(
                                1.0, current_weight + weight * 0.5
                            )
                        else:
                            G.add_edge(
                                src_user, dst_user, weight=weight, edge_type="inferred"
                            )

    logger.info(
        f"Constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    return G


def load_graph(edges_csv: str) -> nx.DiGraph:
    """Load graph from CSV file (backward compatibility)."""
    df = pd.read_csv(edges_csv)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        p = float(row.get("weight", 0.1))
        G.add_edge(
            row["src"], row["dst"], weight=max(MIN_EDGE_WEIGHT, min(MAX_EDGE_WEIGHT, p))
        )
    return G


def ic_spread(
    G: nx.DiGraph, seeds: List[str], samples: int = 100, max_steps: int = 10
) -> float:
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
                    decay_factor = INFLUENCE_DECAY_FACTOR**step
                    effective_p = p * decay_factor

                    if random.random() < effective_p:
                        activated.add(v)
                        new_frontier.append(v)
            frontier = new_frontier

        n += len(activated)
    return n / samples


def compute_influence_scores(
    G: nx.DiGraph, user_risk: Dict[str, float]
) -> Dict[str, float]:
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
        pagerank = nx.pagerank(G, weight="weight")
        betweenness = nx.betweenness_centrality(G, weight="weight")
        out_degree = dict(G.out_degree(weight="weight"))
    except Exception as e:
        logger.warning(f"Failed to compute centrality measures: {e}")
        # Fallback to simple degree centrality
        pagerank = {node: 1.0 / len(G.nodes) for node in G.nodes}
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
        network_influence = (
            PAGERANK_WEIGHT * norm_pagerank
            + BETWEENNESS_WEIGHT * norm_betweenness
            + OUT_DEGREE_WEIGHT * norm_out_degree
        )

        # Combine with risk score
        risk_score = user_risk.get(node, 0.0)

        # Final influence score (higher risk + higher centrality = higher influence)
        influence_scores[node] = network_influence * (1.0 + risk_score)

    return influence_scores


def compute_propagation_loss(
    logits: torch.Tensor,
    user_ids: List[str],
    G: nx.DiGraph,
    user_risk: Dict[str, float],
    samples: int = 50,
) -> torch.Tensor:
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
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Convert logits to phishing probabilities (maintain gradients)
    probs = F.softmax(logits, dim=-1)
    phishing_probs = probs[:, 1] if logits.shape[-1] > 1 else probs[:, 0]

    # Build propagation weights tensor (maintains gradient flow)
    propagation_weights = torch.zeros_like(phishing_probs)
    valid_users = 0

    # Pre-compute influence scores once
    influence_scores = compute_influence_scores(G, user_risk)

    for i, user_id in enumerate(user_ids):
        if user_id not in G.nodes:
            continue

        valid_users += 1

        # Estimate spread if this user posts phishing content
        try:
            expected_spread = ic_spread(G, [user_id], samples=samples)
            influence_score = influence_scores.get(user_id, 0.0)

            # Propagation weight = expected_spread * (1 + influence)
            # This will be multiplied by phishing probability
            propagation_weights[i] = expected_spread * (1.0 + influence_score)

        except Exception:
            # Fallback: use simple risk based on node degree
            out_degree = G.out_degree(user_id, weight="weight")
            propagation_weights[i] = out_degree * 0.1

    if valid_users == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Compute weighted propagation risk (maintains gradient flow)
    # Propagation risk = sum(P(phishing) * propagation_weight) / valid_users
    weighted_risk = (phishing_probs * propagation_weights).sum() / valid_users

    return weighted_risk


def greedy_minimize_spread(
    G: nx.DiGraph,
    budget: int,
    risk: Dict[str, float],
    candidates: List[str],
    samples: int = 100,
) -> List[str]:
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
    high_risk_seeds = [
        u
        for u in cand[: min(HIGH_RISK_SEED_LIMIT, len(cand))]
        if risk.get(u, 0.0) > HIGH_RISK_THRESHOLD
    ]
    base_spread = ic_spread(G, high_risk_seeds, samples) if high_risk_seeds else 0.0

    logger.info(f"Baseline expected spread: {base_spread:.2f}")

    # Create a working copy of the graph once
    G_working = G.copy()
    current_spread = base_spread

    for iteration in range(min(budget, len(cand))):
        best_gain, best_node = -1, None

        for u in cand:
            if u in chosen or u not in G_working.nodes:
                continue

            # Compute spread reduction if we intervene on node u
            # Temporarily remove node and estimate spread
            edges_to_restore = [
                (pred, u, G_working[pred][u])
                for pred in G_working.predecessors(u)
                if G_working.has_edge(pred, u)
            ]
            edges_to_restore += [
                (u, succ, G_working[u][succ])
                for succ in G_working.successors(u)
                if G_working.has_edge(u, succ)
            ]

            # Remove node temporarily
            G_working.remove_node(u)

            # Estimate spread with intervention
            remaining_seeds = [
                s for s in high_risk_seeds if s != u and s in G_working.nodes
            ]
            spread_with_intervention = (
                ic_spread(G_working, remaining_seeds, samples)
                if remaining_seeds
                else 0.0
            )

            # Restore node (more efficient than copying graph each time)
            G_working.add_node(u, **G.nodes[u])
            for src, dst, data in edges_to_restore:
                G_working.add_edge(src, dst, **data)

            # Gain = reduction in spread
            gain = current_spread - spread_with_intervention

            # Bias by node characteristics
            risk_bias = 1.0 + RISK_BIAS_MULTIPLIER * risk.get(
                u, 0.0
            )  # Higher weight for risky nodes
            influence_bias = 1.0 + influence_scores.get(
                u, 0.0
            )  # Higher weight for influential nodes

            adjusted_gain = gain * risk_bias * influence_bias

            if adjusted_gain > best_gain:
                best_gain, best_node = adjusted_gain, u

        if best_node is None:
            break

        chosen.append(best_node)
        # Remove best node permanently from working graph
        if best_node in G_working.nodes:
            G_working.remove_node(best_node)

        logger.info(f"Selected intervention node {best_node} with gain {best_gain:.3f}")

        # Update current spread for next iteration
        current_spread = max(0.0, current_spread - best_gain)

    logger.info(f"Selected {len(chosen)} nodes for intervention: {chosen}")
    return chosen
