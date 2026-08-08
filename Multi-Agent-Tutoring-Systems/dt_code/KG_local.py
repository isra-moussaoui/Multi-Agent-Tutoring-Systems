"""
kg_local.py

Free, local, Neo4j-free replacement for dt_code/KG/KG_create.py + KG_traversal.py.

The original repo builds a knowledge graph in a Neo4j database (one graph per
propositional-logic problem, e.g. "1.1", "2.3", ...) from Data/props/prop_X.X.csv,
then runs a BFS over it to check whether a proposed next-step is derivable.

Neo4j adds zero value here: each per-problem graph has a few dozen nodes. This
module builds the exact same graph as a plain Python dict and runs the exact
same BFS in memory. No signup, no service, no cost, runs instantly on a CPU.

CSV row format (props/prop_X.X.csv), reproduced from KG_create.py:
    each row is one example derivation path through the problem, written as
    comma-separated elements of the form "expression;parents;rule", e.g.

    (A>(B*C));0;Given,(A+D);0;Given,-D;(-D*E);Simplification,...

    - rule == "Given"      -> expression is a starting premise (no parents)
    - rule == anything else -> expression was derived from the parent
      expression(s) (dot-separated if more than one) using that rule.

We aggregate every row in a file into one dict:
    node_derivations = {
        expression: [ [parent1, parent2, ..., rule], ... ]   # possible derivations
    }
A Given node maps to [] (no derivation needed, it's already known).
"""

import csv
from pathlib import Path
from collections import deque

_GRAPH_CACHE = {}


def _parse_row(row):
    """Parse one CSV row into (givens, derivations)."""
    givens = []
    derivations = []  # list of (expr, parents_list, rule)
    for element in row:
        parts = element.split(";")
        if len(parts) < 3:
            continue
        expr = parts[0].strip()
        parent_field = parts[1].strip()
        rule = parts[2].strip()
        if rule == "Given":
            givens.append(expr)
        else:
            parents = [p.strip() for p in parent_field.split(".") if p.strip()]
            derivations.append((expr, parents, rule))
    return givens, derivations


def build_graph_for_cluster(csv_path):
    """
    Build node_derivations for a single problem's CSV file.
    Mirrors KG_create.py's create_given_nodes + create_derived_nodes_and_relationships,
    but writes to a dict instead of MERGE-ing into Neo4j.
    """
    node_derivations = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            givens, derivations = _parse_row(row)
            for g in givens:
                node_derivations.setdefault(g, [])
            for expr, parents, rule in derivations:
                parent_rule_set = sorted(set(parents)) + [rule]
                node_derivations.setdefault(expr, [])
                if parent_rule_set not in node_derivations[expr]:
                    node_derivations[expr].append(parent_rule_set)
    return node_derivations


def load_graph(cluster_id, props_dir):
    """Cached loader, keyed by problem id (e.g. '1.1'). props_dir = Data/props."""
    key = str(cluster_id)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    csv_path = Path(props_dir) / f"prop_{key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No props file for problem {key}: {csv_path}")
    graph = build_graph_for_cluster(csv_path)
    _GRAPH_CACHE[key] = graph
    return graph


def forward_bfs(node_derivations, known_expressions, target_expression):
    """
    Same algorithm as dt_code/KG/KG_traversal.py::forward_bfs, just fed a local
    dict instead of a live Neo4j query.
    Returns (derived_map, discovered_set, success, depth).
    """
    discovered = set(known_expressions)
    queue = deque(known_expressions)
    derived_map = {expr: {"used_parents": [], "method": "Given"} for expr in known_expressions}
    depth_map = {expr: 0 for expr in known_expressions}

    while queue:
        current = queue.popleft()
        if current == target_expression:
            return derived_map, discovered, True, depth_map.get(target_expression, -1)

        for child_expr, derivation_list in node_derivations.items():
            if child_expr in discovered:
                continue
            for parent_set in derivation_list:
                if not parent_set:
                    continue
                potential_parents = parent_set[:-1]
                method = parent_set[-1]
                if all(p in discovered for p in potential_parents):
                    discovered.add(child_expr)
                    derived_map[child_expr] = {
                        "used_parents": list(potential_parents),
                        "method": method,
                    }
                    depth_map[child_expr] = depth_map[current] + 1
                    queue.append(child_expr)
                    break

    return derived_map, discovered, (target_expression in discovered), depth_map.get(target_expression, -1)


def reconstruct_derivation(derived_map, target_expression):
    """Same as KG_traversal.py::reconstruct_derivation."""
    if target_expression not in derived_map:
        return []
    visited = set()
    order = []

    def dfs(expr):
        if expr in visited:
            return
        visited.add(expr)
        for p in derived_map[expr]["used_parents"]:
            dfs(p)
        order.append(expr)

    dfs(target_expression)

    step_list = []
    for expr in order:
        info = derived_map[expr]
        parents = info["used_parents"]
        method = info["method"]
        if method == "Given":
            step_list.append([expr, [None]])
        elif len(parents) == 1:
            step_list.append([expr, [parents[0], method]])
        else:
            step_list.append([expr, parents + [method]])
    return step_list


def reachable_next_steps(node_derivations, known_expressions):
    """
    One BFS-hop expansion: every expression that is IMMEDIATELY derivable right
    now from known_expressions. This is the set of legitimate "valid next steps"
    a student could take -- used to tell 'valid-alternative' apart from 'incorrect'.
    """
    known = set(known_expressions)
    valid_next = set()
    for child_expr, derivation_list in node_derivations.items():
        if child_expr in known:
            continue
        for parent_set in derivation_list:
            if not parent_set:
                continue
            potential_parents = parent_set[:-1]
            if all(p in known for p in potential_parents):
                valid_next.add(child_expr)
                break
    return valid_next


def label_step(proposed_step, correct_step, node_derivations, known_expressions):
    """
    Ground-truth labeler. Mirrors dt_code/llm_response_processing/step_evaluation.py
    ::check_step(), fully local (no Neo4j, no API call).

    Returns one of: "optimal", "valid_alternative", "incorrect"
    """
    if proposed_step is None:
        return "incorrect"
    proposed_step = proposed_step.strip()
    if proposed_step == correct_step.strip():
        return "optimal"
    valid_next = reachable_next_steps(node_derivations, known_expressions)
    if proposed_step in valid_next:
        return "valid_alternative"
    return "incorrect"


if __name__ == "__main__":
    # Quick self-test against the real repo data (no API calls, no Neo4j).
    import json

    data_dir = Path(__file__).parent / "Data"
    props_dir = data_dir / "props"
    preState = data_dir / "cleaned_data" / "preState.jsonl"

    tested = 0
    with open(preState, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#") or line.startswith("/*") or line.startswith("*"):
                continue
            row = json.loads(line)
            problem = row["currentProblem"]
            givens = row["Givens"]
            intermediates = row["Intermediates"]["Expressions"]
            known = givens + intermediates
            correct = row["sAssertion"]

            graph = load_graph(problem, props_dir)

            # sAssertion should always label itself as "optimal"
            label = label_step(correct, correct, graph, known)
            assert label == "optimal", f"id {row['id']}: expected optimal, got {label}"

            # A nonsense string should always be "incorrect"
            label_bad = label_step("NOT_A_REAL_EXPRESSION", correct, graph, known)
            assert label_bad == "incorrect", f"id {row['id']}: expected incorrect, got {label_bad}"

            tested += 1

    print(f"Self-test passed on {tested} proof states across all problems. "
          f"No Neo4j, no API calls, pure local Python.")