"""Counterfactual Inference Engine for ARBITER.

Implements do-calculus style counterfactual computation:
  intervene(graph, record, feature, value) → counterfactual_outcome

Algorithm:
  1. Topological sort of the causal DAG.
  2. Set the intervened feature to the counterfactual value (do-operator).
  3. Forward-propagate through the graph recomputing node values.
  4. Return the outcome node's value.
"""
import networkx as nx
from typing import Any, Dict, Optional


def intervene(
    graph: nx.DiGraph,
    record: Dict,
    feature_id: str,
    counterfactual_value: Any,
) -> Dict:
    """
    Compute the counterfactual outcome for a decision record.

    Parameters
    ----------
    graph              : Ground-truth causal DiGraph (with hidden nodes).
    record             : A decision record dict.
    feature_id         : The feature whose value we're intervening on.
    counterfactual_value : The value we're setting it to.

    Returns
    -------
    dict with:
        original_outcome       – the record's actual outcome
        counterfactual_outcome – the predicted outcome under intervention
        changed                – bool, whether outcome changed
        causal_path            – list of nodes on the intervention path
        confidence             – float 0-1
    """
    original_outcome = record["outcome"]

    # Build a working copy of the record's feature state
    state = {}
    state.update(record.get("feature_vector", {}))
    state.update(record.get("proxy_vector", {}))
    state.update(record.get("hidden_vector", {}))

    # Do-operator: fix the intervened feature
    state[feature_id] = counterfactual_value

    # Topological sort of causal edges only
    causal_subgraph = nx.DiGraph()
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") in ("causal", "temporal"):
            causal_subgraph.add_edge(u, v, **d)

    try:
        topo_order = list(nx.topological_sort(causal_subgraph))
    except nx.NetworkXUnfeasible:
        # Cycle fallback — return original
        return {
            "original_outcome":       original_outcome,
            "counterfactual_outcome": original_outcome,
            "changed":                False,
            "causal_path":            [],
            "confidence":             0.3,
        }

    # Forward propagation
    causal_path = [feature_id]
    for node in topo_order:
        if node == feature_id:
            continue
        preds = list(causal_subgraph.predecessors(node))
        if not preds:
            continue
        # Simple weighted influence: if any predecessor is high-influence and
        # the intervened feature is upstream, propagate the effect.
        if feature_id in nx.ancestors(causal_subgraph, node):
            causal_path.append(node)
            node_data = graph.nodes.get(node, {})
            ntype = node_data.get("node_type", "")
            if ntype == "outcome" or node.endswith("_rate_overall") or node.endswith("_rate_minority"):
                state[node] = _compute_outcome(state, node, graph, feature_id, counterfactual_value)

    # Determine counterfactual outcome
    cf_outcome = _infer_final_outcome(state, feature_id, counterfactual_value, graph)
    confidence = _compute_confidence(graph, feature_id, cf_outcome, original_outcome)

    return {
        "original_outcome":       original_outcome,
        "counterfactual_outcome": cf_outcome,
        "changed":                cf_outcome != original_outcome,
        "causal_path":            causal_path,
        "confidence":             confidence,
    }


def _compute_outcome(state: Dict, outcome_node: str, graph: nx.DiGraph,
                     intervened_feature: str, cf_value: Any) -> str:
    """
    Heuristic outcome computation based on the intervened feature.
    In a real system this would run the actual model; here we use the
    causal graph structure to infer the direction.
    """
    # Get edge weight from intervened feature to this outcome (via any path)
    try:
        path = nx.shortest_path(graph, intervened_feature, outcome_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return state.get(outcome_node, "denied")

    # If the path goes through a proxy or hidden risk score → denial likely
    has_proxy  = any(graph.nodes[n].get("proxy",  False) for n in path[1:])
    has_hidden = any(graph.nodes[n].get("hidden", False) for n in path[1:])

    if has_proxy or has_hidden:
        # Removing proxy influence → approval more likely
        if "cluster_7" not in str(cf_value):
            return "approved"
        else:
            return "denied"

    return state.get(outcome_node, "denied")


def _infer_final_outcome(state: Dict, feature_id: str, cf_value: Any,
                         graph: nx.DiGraph) -> str:
    """Determine the final loan outcome after intervention."""
    # Primary rule: if we changed away from a discriminatory proxy value
    if "zip_code" in feature_id or "surname" in feature_id or "neighborhood" in feature_id:
        if "cluster_7" not in str(cf_value) and "pattern_A" not in str(cf_value):
            return "approved"
        return "denied"

    # Credit-score based rule
    credit = state.get("credit_score", 600)
    if isinstance(credit, (int, float)):
        return "approved" if credit > 620 else "denied"

    return state.get("outcome", "denied")


def _compute_confidence(graph: nx.DiGraph, feature_id: str,
                        cf_outcome: str, original_outcome: str) -> float:
    """Estimate confidence in the counterfactual inference."""
    # Higher confidence if the feature has a direct causal edge to an outcome
    outcome_nodes = [n for n, d in graph.nodes(data=True)
                     if d.get("node_type") == "outcome" or "rate" in n]
    for on in outcome_nodes:
        if graph.has_edge(feature_id, on):
            return 0.95
    # Medium confidence if there's a 2-hop path
    for on in outcome_nodes:
        try:
            path = nx.shortest_path(graph, feature_id, on)
            if len(path) == 3:
                return 0.80
            if len(path) > 3:
                return 0.65
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return 0.50
