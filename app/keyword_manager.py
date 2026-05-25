"""
Keyword Manager Module (READ-ONLY)
Manages DO-178 compliance keywords from keywords.json.
IMPORTANT: This module reads keywords only. User manages keywords.json manually.
"""

import logging
import json
import os
from typing import List, Set, Dict

logger = logging.getLogger(__name__)


class KeywordManager:
    """
    Manages DO-178 compliance keywords (READ-ONLY from keywords.json).
    
    DESIGN PRINCIPLE: Keywords are user-controlled only.
    This module reads keywords for use in LLM context, but NEVER modifies them.
    """
    
    def __init__(self, keywords_file: str = "keywords.json"):
        """
        Initialize Keyword Manager (READ-ONLY mode).
        
        Args:
            keywords_file: Path to keywords.json file
        """
        self.keywords_file = keywords_file
        self.keywords = self._load_keywords()
        logger.info(f"KeywordManager initialized (READ-ONLY) with {len(self.keywords)} keywords")
    
    def _load_keywords(self) -> Set[str]:
        """
        Load keywords from keywords.json.
        
        Returns:
            Set of do178 keywords
        """
        if not os.path.exists(self.keywords_file):
            logger.warning(f"Keywords file not found: {self.keywords_file}")
            return set()
        
        try:
            with open(self.keywords_file, 'r') as f:
                data = json.load(f)
                keywords = set(data.get("do178_keywords", []))
                logger.info(f"Loaded {len(keywords)} keywords from {self.keywords_file}")
                return keywords
        except Exception as e:
            logger.error(f"Error loading keywords: {str(e)}")
            return set()
    
    def find_keywords_in_text(self, text: str) -> List[str]:
        """
        Find matching keywords in text (for context/reference only).
        
        Used for LLM prompt construction to provide optional context.
        DOES NOT modify keywords.json or keyword list.
        
        Args:
            text: Text to search for keywords
            
        Returns:
            List of found keywords (if any)
        """
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def get_keywords(self) -> Set[str]:
        """Get current set of keywords (READ-ONLY)."""
        return self.keywords.copy()
    
    def get_keywords_list(self) -> List[str]:
        """Get keywords as sorted list (READ-ONLY)."""
        return sorted(list(self.keywords))
