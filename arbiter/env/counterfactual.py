"""Counterfactual Inference Engine for ARBITER.

Implements do-calculus style counterfactual computation:
  intervene(graph, record, feature, value) -> counterfactual_outcome

Algorithm:
  1. Topological sort of the causal DAG.
  2. Set the intervened feature to the counterfactual value (do-operator).
  3. Forward-propagate through the graph recomputing node values.
  4. Return the outcome node's value.

Domain context is read from graph.graph["domain_context"] when present;
loan-domain literals are used as a fallback so the existing loan path is
unchanged.
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
        original_outcome       - the record's actual outcome
        counterfactual_outcome - the predicted outcome under intervention
        changed                - bool, whether outcome changed
        causal_path            - list of nodes on the intervention path
        confidence             - float 0-1
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
        # Cycle fallback - return original
        return {
            "original_outcome":       original_outcome,
            "counterfactual_outcome": original_outcome,
            "changed":                False,
            "causal_path":            [],
            "confidence":             0.3,
        }

    # Pull domain context (None -> loan path)
    domain_context = graph.graph.get("domain_context")

    # Forward propagation
    causal_path = [feature_id]
    for node in topo_order:
        if node == feature_id:
            continue
        preds = list(causal_subgraph.predecessors(node))
        if not preds:
            continue
        if feature_id in nx.ancestors(causal_subgraph, node):
            causal_path.append(node)
            node_data = graph.nodes.get(node, {})
            ntype = node_data.get("node_type", "")
            if _is_outcome_node(node, ntype, domain_context):
                state[node] = _compute_outcome(
                    state, node, graph, feature_id, counterfactual_value, domain_context
                )

    # Determine counterfactual outcome
    cf_outcome = _infer_final_outcome(
        state, feature_id, counterfactual_value, graph, domain_context
    )
    confidence = _compute_confidence(graph, feature_id, cf_outcome, original_outcome)

    return {
        "original_outcome":       original_outcome,
        "counterfactual_outcome": cf_outcome,
        "changed":                cf_outcome != original_outcome,
        "causal_path":            causal_path,
        "confidence":             confidence,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_outcome_node(node: str, ntype: str, domain_context: Optional[Dict]) -> bool:
    """Return True if this node should be treated as an outcome node."""
    if ntype == "outcome":
        return True
    if domain_context:
        pos = domain_context.get("positive_outcome", "")
        neg = domain_context.get("negative_outcome", "")
        return (
            node.endswith("_rate_overall")
            or node.endswith("_rate_minority")
            or node.endswith("_rate_majority")
            or (pos and node.startswith(pos))
            or (neg and node.startswith(neg))
        )
    # Loan-domain fallback
    return node.endswith("_rate_overall") or node.endswith("_rate_minority")


def _compute_outcome(
    state: Dict,
    outcome_node: str,
    graph: nx.DiGraph,
    intervened_feature: str,
    cf_value: Any,
    domain_context: Optional[Dict],
) -> str:
    """
    Heuristic outcome computation based on the intervened feature.
    In a real system this would run the actual model; here we use the
    causal graph structure to infer the direction.
    """
    # Resolve outcome strings from context or fall back to loan literals
    if domain_context:
        positive = domain_context.get("positive_outcome", "approved")
        negative = domain_context.get("negative_outcome", "denied")
        discriminated_val = domain_context.get("discriminated_value", "cluster_7")
    else:
        positive = "approved"
        negative = "denied"
        discriminated_val = "cluster_7"

    # Get edge weight from intervened feature to this outcome (via any path)
    try:
        path = nx.shortest_path(graph, intervened_feature, outcome_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return state.get(outcome_node, negative)

    # If the path goes through a proxy or hidden risk score -> denial likely
    has_proxy  = any(graph.nodes[n].get("proxy",  False) for n in path[1:])
    has_hidden = any(graph.nodes[n].get("hidden", False) for n in path[1:])

    if has_proxy or has_hidden:
        # Removing proxy influence -> approval more likely
        if str(cf_value) != str(discriminated_val):
            return positive
        else:
            return negative

    return state.get(outcome_node, negative)


def _infer_final_outcome(
    state: Dict,
    feature_id: str,
    cf_value: Any,
    graph: nx.DiGraph,
    domain_context: Optional[Dict],
) -> str:
    """Determine the final outcome after intervention."""
    # Resolve domain parameters from context or use loan-domain fallbacks
    if domain_context:
        discriminated_feature = domain_context.get("discriminated_feature", "")
        discriminated_value   = domain_context.get("discriminated_value", "cluster_7")
        threshold_feature     = domain_context.get("threshold_feature", "credit_score")
        threshold_value       = domain_context.get("threshold_value", 620)
        positive_outcome      = domain_context.get("positive_outcome", "approved")
        negative_outcome      = domain_context.get("negative_outcome", "denied")
    else:
        discriminated_feature = ""
        discriminated_value   = "cluster_7"
        threshold_feature     = "credit_score"
        threshold_value       = 620
        positive_outcome      = "approved"
        negative_outcome      = "denied"

    # Primary rule: if we changed away from the discriminatory proxy value
    if feature_id == discriminated_feature or (
        # Loan-domain backwards compat: substring match on known proxy ids
        not discriminated_feature and (
            "zip_code" in feature_id
            or "surname" in feature_id
            or "neighborhood" in feature_id
        )
    ):
        if str(cf_value) != str(discriminated_value) and "pattern_A" not in str(cf_value):
            return positive_outcome
        return negative_outcome

    # Threshold-feature based rule
    score = state.get(threshold_feature, state.get("credit_score", 600))
    if isinstance(score, (int, float)):
        return positive_outcome if score > threshold_value else negative_outcome

    return state.get("outcome", negative_outcome)


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
