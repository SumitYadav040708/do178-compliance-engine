"""
LLM Explainer Module
Integrates Ollama for generating compliance explanations.
Uses local Phi model for deterministic, offline reasoning.
"""

import logging
import json
from typing import Dict, Optional, List
import requests
from typing import Any
from app.config import classify_similarity

logger = logging.getLogger(__name__)


class OllamaExplainer:
    """
    Generates compliance explanations using local Ollama model.
    
    Attributes:
        base_url: Ollama server URL
        model_name: Model to use (e.g., 'phi')
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "phi",
        timeout: int = 60
    ):
        """
        Initialize Ollama Explainer.
        
        Args:
            base_url: Ollama server URL (default localhost:11434)
            model_name: Model identifier (default phi)
            timeout: Request timeout seconds
            
        Raises:
            ConnectionError: If Ollama server not reachable
        """
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.api_url = f"{base_url}/api"
        
        # Check connection
        if not self.check_connection():
            logger.warning(
                f"Cannot connect to Ollama at {base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        else:
            logger.info(f"Connected to Ollama at {base_url}")
    
    def check_connection(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            True if reachable, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama connection check failed: {str(e)}")
            return False
    
    def check_model_available(self) -> bool:
        """
        Check if configured model is available locally.
        
        Returns:
            True if model available, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                is_available = any(self.model_name in name for name in model_names)
                
                if is_available:
                    logger.info(f"Model {self.model_name} is available")
                else:
                    logger.warning(
                        f"Model {self.model_name} not found. "
                        f"Available: {model_names}"
                    )
                
                return is_available
            return False
        except Exception as e:
            logger.error(f"Error checking available models: {str(e)}")
            return False
    
    def generate_explanation(
        self,
        keyword: str,
        similarity_score: float,
        connection_type: str,
        chunk: str,
        standard_chunks: Any = None
    ) -> str:
        """
        Generate compliance explanation with support for multiple reference chunks.
        
        Args:
            keyword: DO-178 keyword being analyzed
            similarity_score: Similarity score (0-1) or max_similarity if multiple chunks
            connection_type: Classification from config ("strong", "weak", "missing")
            chunk: Retrieved text from document
            standard_chunks: Single chunk string OR list of chunk dicts with similarity_score.
                            If dict list, combines all chunks and passes to LLM.
                            Backward compatible: accepts string for single chunk.
            
        Returns:
            Generated explanation string
        """
        # Strong alignment: return fixed deterministic explanation
        if connection_type == "strong":
            return f"Strong alignment detected for '{keyword}' with DO-178 requirements. Traceability and requirement linkage is evident."
        
        # Missing: brief explanation
        if connection_type == "missing":
            return f"No significant alignment for '{keyword}'. Limited relevance to DO-178 requirements."
        
        # Weak alignment: use LLM with lightweight prompt
        if not chunk or not chunk.strip():
            return f"Partial alignment for '{keyword}'. Some relevance to DO-178 requirements, but gaps exist."
        
        # Combine multiple chunks if provided as list
        combined_reference = self._combine_reference_chunks(standard_chunks)
        
        prompt = self._build_weak_alignment_prompt(
            keyword=keyword,
            project_chunk=chunk[:300],
            reference_evidence=combined_reference,
            connection_type=connection_type
        )
        
        try:
            logger.info(f"LLM: Calling for weak alignment '{keyword}' (sim={similarity_score:.3f})")
            response = self._call_ollama(prompt, max_tokens=50)
            logger.info(f"LLM: Success! Generated {len(response)} chars")
            return response
        except Exception as e:
            logger.debug(f"LLM failed for '{keyword}': {str(e)}")
            return f"Weak alignment for '{keyword}'. Limited relevance to DO-178 requirements."
    
    def _combine_reference_chunks(self, standard_chunks: Any) -> str:
        """
        Combine multiple reference chunks into a single formatted string.
        
        Args:
            standard_chunks: Can be:
                - None: return empty string
                - str: return as-is (backward compatibility)
                - list: list of chunk dicts with 'chunk_text' and optionally 'similarity_score'
                
        Returns:
            Formatted combined chunks string (trimmed to ~600 chars total)
        """
        if not standard_chunks:
            return ""
        
        # Backward compatibility: single string
        if isinstance(standard_chunks, str):
            return standard_chunks[:300]
        
        # Multiple chunks as list
        if isinstance(standard_chunks, list) and len(standard_chunks) > 0:
            combined = []
            max_total_length = 600
            current_length = 0
            
            for i, chunk_data in enumerate(standard_chunks, 1):
                # Extract chunk text (handle dict or string format)
                if isinstance(chunk_data, dict):
                    chunk_text = chunk_data.get("chunk_text", "")
                    sim_score = chunk_data.get("similarity_score", 0)
                else:
                    chunk_text = str(chunk_data)
                    sim_score = 0
                
                # Trim chunk to reasonable size
                chunk_preview = chunk_text[:150]
                
                # Format: "[Match N]: <text>"
                formatted = f"[Match {i}]: {chunk_preview}"
                
                # Check if adding this chunk exceeds limit
                new_length = current_length + len(formatted) + 2  # +2 for newline
                if new_length > max_total_length and combined:
                    break
                
                combined.append(formatted)
                current_length = new_length
            
            return "\n".join(combined) if combined else ""
        
        return ""
    
    def _build_weak_alignment_prompt(
        self,
        keyword: str,
        project_chunk: str,
        connection_type: str,
        reference_evidence: str = ""
    ) -> str:
        """
        Build prompt for alignment explanation with support for multiple reference chunks.
        
        Args:
            keyword: DO-178 keyword
            project_chunk: Project document context
            connection_type: "weak" or "missing" (already filtered strong earlier)
            reference_evidence: Combined reference chunks (can be multi-match format)
            
        Returns:
            Formatted prompt for LLM
        """
        # Build reference section if provided
        reference_section = ""
        if reference_evidence:
            reference_section = f"\nReference Evidence (from standard):\n{reference_evidence}\n"
        
        prompt = f"""You are a DO-178C compliance assistant.

Connection Type: {connection_type}

Explain briefly why the alignment is {connection_type}.

Rules:
- Do NOT assume full compliance
- Do NOT say "insufficient evidence" unless nothing can be inferred
- Do NOT start sentences with "Yes"
- Avoid repeating the exact same opening phrase every time
- Focus on what is partially present and what is missing
- Do NOT mention whether the keyword appears or not
- Do NOT add unrelated examples or external concepts
- Keep tone neutral and direct

Instructions:
- Identify 1 thing present (if any)
- Identify 1 missing requirement
- Output only one concise sentence

Input:
Keyword: {keyword}
Project Text: {project_chunk}{reference_section}
Output:
Short explanation of {connection_type} alignment."""

        return prompt
    
    def _call_ollama(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Call Ollama API for text generation.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default 50)
            
        Returns:
            Generated text
            
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "model": "phi",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "top_p": 1.0,
                "top_k": 30,
                "repeat_penalty": 1.0,
                "num_predict": max_tokens,
                "stop": ["\n\n", "Question:", "Suppose", "Consider"]
            }
            
            logger.debug(f"Calling Ollama with model: phi")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "").strip()
                
                # Remove conversational starters
                conversational_starters = ["Yes,", "Yes.", "Yes ", "No,", "No.", "No ", "As an AI", "I cannot", "I'm sorry"]
                for starter in conversational_starters:
                    if text.startswith(starter):
                        text = text[len(starter):].strip()
                        break
                
                # Replace unnecessary phrases
                text = text.replace("the alignment is weak because", "Weak alignment:")
                text = text.replace("the alignment is weak in", "Weak alignment:")
                
                # Extract complete sentences (split by "." and keep non-empty ones)
                sentences = [s.strip() for s in text.split(".") if s.strip()]
                
                # Keep first 2-3 complete sentences (no fragments)
                sentences = sentences[:3]
                
                # Reconstruct with proper punctuation
                if sentences:
                    text = ". ".join(sentences) + "."
                else:
                    text = ""
                
                # Apply soft length cap (~300 chars) at sentence boundaries only
                max_length = 300
                if len(text) > max_length:
                    truncated = ""
                    for sentence in sentences:
                        # Build incrementally, checking length before adding each sentence
                        if truncated:
                            candidate = truncated + " " + sentence + "."
                        else:
                            candidate = sentence + "."
                        
                        if len(candidate) <= max_length:
                            truncated = candidate
                        else:
                            break
                    
                    text = truncated if truncated else (sentences[0] + "." if sentences else "")
                
                logger.debug(f"Ollama response: {len(text)} characters (cleaned)")
                return text
            else:
                raise Exception(
                    f"API returned status {response.status_code}: "
                    f"{response.text[:200]}"
                )
        
        except requests.Timeout:
            logger.error("Ollama request timed out")
            raise Exception("LLM timeout")
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama")
            raise Exception("Cannot connect to Ollama server")
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from Ollama")
            raise Exception("Invalid JSON response from Ollama")
    
    def batch_generate_explanations(
        self,
        matches: List[Dict[str, Any]],
        max_tokens: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple matches (DEPRECATED - use generate_explanation directly).
        
        Args:
            matches: List of match dicts with keyword, similarity_score, chunk, standard_chunks, connection_type
            max_tokens: Max tokens per explanation
            
        Returns:
            Matches with generated explanations
        """
        explained_matches = []
        
        for i, match in enumerate(matches):
            keyword = match.get("keyword", "")
            chunk = match.get("chunk", "")
            standard_chunks = match.get("standard_chunks") or match.get("standard_chunk")
            connection_type = match.get("connection_type", "weak")
            similarity = match.get("similarity_score", 0.0)
            
            try:
                explanation = self.generate_explanation(
                    keyword=keyword,
                    similarity_score=similarity,
                    connection_type=connection_type,
                    chunk=chunk,
                    standard_chunks=standard_chunks
                )
                match["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Failed to generate explanation for match {i}: {str(e)}")
                match["explanation"] = "Unable to generate explanation"
            
            explained_matches.append(match)
        
        return explained_matches
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about configured model.
        
        Returns:
            Model information
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "connection_status": "connected" if self.check_connection() else "disconnected",
            "model_available": self.check_model_available()
        }


class LocalExplainerFallback:
    """
    Fallback explainer when Ollama is not available.
    Generates basic rule-based explanations.
    """
    
    @staticmethod
    def generate_explanation(
        keyword: str,
        similarity_score: float,
        connection_type: str,
        chunk: str,
        standard_chunks: Any = None
    ) -> str:
        """
        Generate rule-based explanation with DO-178 guidance.
        Strong alignment (>= 0.85) returns deterministic text.
        Weak alignment (< 0.85) returns deterministic guidance text.
        
        Args:
            keyword: DO-178 keyword being analyzed
            similarity_score: Similarity score (0-1) or max_similarity if multiple chunks
            connection_type: Classification from config ("strong", "weak", "missing")
            chunk: Project context
            standard_chunks: Single chunk string OR list of chunk dicts.
                            Backward compatible: accepts string for single chunk.
            
        Returns:
            Generated explanation string
        """
        # Strong alignment (>= 0.85): return fixed deterministic explanation
        if similarity_score >= 0.85:
            return f"Strong alignment detected for '{keyword}' with DO-178 requirements. Traceability and requirement linkage is evident."
        
        # Weak alignment (< 0.85): fallback deterministic text
        return f"Weak alignment for '{keyword}'. Limited relevance to DO-178 requirements."
    
    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """
        Get fallback model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            "type": "Fallback",
            "model": "LocalExplainerFallback",
            "description": "Rule-based fallback explainer (Ollama not available)",
            "status": "Active"
        }
