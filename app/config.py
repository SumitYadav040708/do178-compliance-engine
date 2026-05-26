"""
DO-178 Compliance Engine - Configuration

Central location for all configurable thresholds and parameters.
Modify these values to tune system behavior.
"""


class SimilarityConfig:
    """Similarity score thresholds for keyword matching classification."""
    
    # Strong match: high semantic alignment (>= 0.85)
    STRONG_THRESHOLD = 0.80
    
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


class KeywordBoostConfig:
    """Configuration for keyword-aware scoring boost."""
    
    # Hybrid scoring: final_score = (embedding_weight * embedding_sim) + (keyword_weight * keyword_match)
    # embedding_weight: Weight for semantic embedding similarity
    # keyword_weight: Weight for explicit keyword presence (binary: 0 or 1)
    EMBEDDING_WEIGHT = 0.8  # 80% weight on embedding similarity
    KEYWORD_WEIGHT = 0.2    # 20% weight on keyword presence
    
    # Ensure weights sum to 1.0
    _WEIGHT_SUM = EMBEDDING_WEIGHT + KEYWORD_WEIGHT
    assert _WEIGHT_SUM == 1.0, f"Weights must sum to 1.0, got {_WEIGHT_SUM}"
    
    @staticmethod
    def calculate_hybrid_score(embedding_similarity: float, keyword_match: int) -> float:
        """
        Calculate hybrid score combining embedding similarity and keyword presence.
        
        Args:
            embedding_similarity: Embedding-based similarity score (0-1)
            keyword_match: Binary keyword match (1 if present, 0 if not)
            
        Returns:
            Hybrid score (0-1) clipped to valid range
        """
        hybrid = (KeywordBoostConfig.EMBEDDING_WEIGHT * embedding_similarity +
                  KeywordBoostConfig.KEYWORD_WEIGHT * keyword_match)
        return min(max(hybrid, 0.0), 1.0)  # Clip to [0, 1]
