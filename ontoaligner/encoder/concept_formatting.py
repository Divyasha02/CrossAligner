from typing import Any, Dict, List


def _join_labels(items: List[Dict[str, Any]]) -> str:
    labels = [item.get("label") or item.get("name") or item.get("iri") for item in items or []]
    labels = [label for label in labels if label]
    return "; ".join(labels) if labels else "(none)"


def _join_comments(comments: List[str]) -> str:
    cleaned_comments = [comment.strip() for comment in (comments or []) if isinstance(comment, str) and comment.strip()]
    return " ".join(cleaned_comments) if cleaned_comments else "(none)"

def _join_axioms(axioms: List[str]) -> str:
    cleaned_axioms = [
        " ".join(axiom.split()).strip()
        for axiom in (axioms or [])
        if isinstance(axiom, str) and axiom.strip()
    ]
    return "; ".join(cleaned_axioms) if cleaned_axioms else "(none)"


# Stage 1 
def format_candidate_concept_context(owl: Dict[str, Any]) -> str:
    """Context for first-stage candidate selection.

    Includes the concept label, synonyms, definition, and verbalized axioms,
    while deliberately excluding parents and children.
    """
    concept_label = owl.get("label") or owl.get("name") or owl.get("iri", "(none)")
    synonyms = _join_labels(owl.get("synonyms", []))
    definition = _join_comments(owl.get("comment", []))
    verbalized_axioms = _join_axioms(owl.get("verbalized_axioms", []))
    return (
        f"Concept: {concept_label}\n"
        f"Synonyms: {synonyms}\n"
        f"Definitions: {definition}\n"
    )

# Stage 2
def format_concept_context(owl: Dict[str, Any]) -> str:
    concept_label = owl.get("label") or owl.get("name") or owl.get("iri", "(none)")
    parents = _join_labels(owl.get("parents", []))
    childrens = _join_labels(owl.get("childrens", []))
    synonyms = _join_labels(owl.get("synonyms", []))
    definition = _join_comments(owl.get("comment", []))
    verbalized_axioms = _join_axioms(owl.get("verbalized_axioms", []))

    return (
        f"Concept: {concept_label}\n"
        f"Parents: {parents}\n"
        f"Childrens: {childrens}\n"
        f"Synonyms: {synonyms}\n"
        f"Definitions: {definition}\n"
        f"Verbalized Axioms: {verbalized_axioms}\n"
    )