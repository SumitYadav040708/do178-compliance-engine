"""
DO-178 Compliance Engine - Configuration

Central location for all configurable thresholds and parameters.
Modify these values to tune system behavior.
"""


class SimilarityConfig:
    """Similarity score thresholds for keyword matching classification."""
    
    # Strong match: high semantic alignment (>= 0.85)
    STRONG_THRESHOLD = 0.85
    
    # Weak match: partial semantic alignment (>= 0.55 and < 0.85)
    WEAK_THRESHOLD = 0.55
    
    # Missing: insufficient semantic alignment (< 0.55)
    # No explicit threshold needed


def classify_similarity(score: float) -> str:
    """
    Classify a similarity score into connection type.
    
    Args:
        score: Similarity score between 0 and 1
        
    Returns:
        Connection type: "strong", "weak", or "missing"
    """
    if score >= SimilarityConfig.STRONG_THRESHOLD:
        return "strong"
    elif score >= SimilarityConfig.WEAK_THRESHOLD:
        return "weak"
    else:
        return "missing"


class LLMConfig:
    """Configuration for LLM explanation generation."""
    
    # LLM is called only for weak alignment (0.55 <= score < 0.85)
    # Strong alignment (>= 0.85) returns deterministic text
    # Missing (< 0.55) returns deterministic text
    MIN_LLM_THRESHOLD = SimilarityConfig.WEAK_THRESHOLD
    MAX_LLM_THRESHOLD = SimilarityConfig.STRONG_THRESHOLD
    
    # Maximum tokens for explanation
    MAX_EXPLANATION_TOKENS = 100
