"""
DO-178 Compliance Analysis System - Refactored Pipeline

FINAL DESIGN - Two Modes:

1. BUILD MODE: Index reference documents
   Input: ReferencePDF/ folder (reference documents)
   Process: Extract → Chunk → Embed → Build FAISS index
   Output: indexes/reference_index.faiss + indexes/metadata.json
   Rules: keywords.json is READ-ONLY (never modified)

2. ANALYSIS MODE: Analyze documents against reference
   Input: VerifyDocumentCompliance/ folder (documents to analyze)
   Process: 
     - Load FAISS index (must exist)
     - Extract → Chunk → Embed each document
     - For each keyword: find best matches in standard
     - Generate LLM explanations (if index exists)
     - Output: keyword-per-row CSV
   Rules:
     - FAISS index NOT modified
     - keywords.json NOT modified
     - ReferencePDF data NOT affected

KEYWORD OUTPUT STRUCTURE:
- Each keyword becomes ONE ROW in CSV
- Columns: filename, keyword, connection_type, similarity_score, matched_reference, matched_page_number, llm_explanation
- connection_type: strong (≥0.75), weak (≥0.55 and <0.75), missing (<0.55)

CLASSIFICATION (configurable in app/config.py):
- ≥ 0.75 → strong
- ≥ 0.55 to < 0.75 → weak
- < 0.55 → missing

SPECIAL CASE - No FAISS Index:
- Keyword-only matching (no semantic search)
- No LLM explanation
- similarity_score: empty/0
- llm_explanation: empty
"""

import logging
import sys
import argparse
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import numpy as np

from .pdf_parser import PDFParser
from .chunker import DocumentChunker
from .embedder import Embedder
from .retriever import FAISSRetriever
from .llm_explainer import OllamaExplainer, LocalExplainerFallback
from .report_generator import ReportGenerator
from .keyword_manager import KeywordManager
from .config import classify_similarity, SimilarityConfig, LLMConfig, KeywordBoostConfig


# Configure logging
def setup_logging(verbose: bool = False):
    """Configure logging with color support."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('compliance_analysis.log', encoding='utf-8')
        ]
    )


logger = logging.getLogger(__name__)


class DO178ComplianceAnalyzer:
    """
    Refactored compliance analysis engine with two distinct modes:
    1. BUILD: Create FAISS index from ReferencePDF folder
    2. ANALYZE: Keyword-based document analysis against standard
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.pdf_parser = PDFParser()
        self.chunker = DocumentChunker(
            min_chunk_size=self.config.get("min_chunk_size", 60),
            max_chunk_size=self.config.get("max_chunk_size", 120),
            overlap=self.config.get("overlap", 25)
        )
        self.embedder = Embedder(device=self.config.get("device", "cpu"))
        self.retriever = FAISSRetriever(
            index_path=self.config.get("index_path", "indexes/reference_index.faiss"),
            metadata_path=self.config.get("metadata_path", "indexes/metadata.json")
        )
        
        # Initialize keyword manager (READ-ONLY)
        keywords_file = self.config.get("keywords_file", "keywords.json")
        self.keyword_manager = KeywordManager(keywords_file=keywords_file)
        
        # Try Ollama, fallback to rule-based
        try:
            self.explainer = OllamaExplainer(
                model_name=self.config.get("model_name", "phi")
            )
            self.use_ollama = self.explainer.check_connection()
            if not self.use_ollama:
                logger.warning("Ollama not available, using fallback explainer")
                self.explainer = LocalExplainerFallback()
        except Exception as e:
            logger.warning(f"Could not initialize Ollama: {str(e)}, using fallback")
            self.explainer = LocalExplainerFallback()
            self.use_ollama = False
        
        self.report_gen = ReportGenerator(
            output_dir=self.config.get("output_dir", "outputs")
        )
        
        logger.info("DO-178 Compliance Analyzer initialized (REFACTORED)")
        logger.info(f"Index path: {self.retriever.index_path}")
        logger.info(f"Keywords file: {keywords_file} (READ-ONLY)")
    
    # ==================== BUILD MODE ====================
    
    def build_reference_index_from_folder(self, standard_pdf_folder: str) -> bool:
        """
        BUILD MODE: Process all PDFs in ReferencePDF folder to create FAISS index.
        
        Args:
            standard_pdf_folder: Path to ReferencePDF folder
            
        Returns:
            True if successful
        """
        logger.info("=" * 70)
        logger.info("BUILD MODE: Indexing ReferencePDF Folder")
        logger.info("=" * 70)
        
        if not os.path.isdir(standard_pdf_folder):
            logger.error(f"ReferencePDF folder not found: {standard_pdf_folder}")
            return False
        
        # Find all PDFs in folder
        pdf_files = glob.glob(os.path.join(standard_pdf_folder, "*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in ReferencePDF folder: {standard_pdf_folder}")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) in ReferencePDF folder")
        
        all_chunks = []
        all_embeddings = []  # Will store numpy arrays
        
        # Process each PDF
        for pdf_file in pdf_files:
            logger.info(f"\nProcessing: {os.path.basename(pdf_file)}")
            
            if not self.pdf_parser.validate_pdf(pdf_file):
                logger.warning(f"Skipping invalid PDF: {pdf_file}")
                continue
            
            # Extract text with metadata
            logger.info("  → Extracting text...")
            extracted = self.pdf_parser.extract_text_with_metadata(pdf_file)
            
            # Chunk the PDF
            logger.info("  → Creating chunks...")
            chunks = self.chunker.chunk_pdf_pages(
                extracted["pages"],
                source_file=os.path.basename(pdf_file)
            )
            
            logger.info(f"  → Created {len(chunks)} chunks from {os.path.basename(pdf_file)}")
            if not chunks:
                logger.warning(f"No chunks created from: {pdf_file}")
                continue
            
            logger.info(f"  → Created {len(chunks)} chunks")
            
            # Generate embeddings
            logger.info("  → Generating embeddings...")
            texts = [chunk["chunk_text"] for chunk in chunks]
            embeddings = self.embedder.embed_texts(texts)  # Returns np.ndarray
            
            all_chunks.extend(chunks)
            all_embeddings.append(embeddings)  # Store each embedding array
        
        if not all_chunks:
            logger.error("No chunks created from any ReferencePDF files")
            return False
        
        logger.info(f"\nTotal chunks: {len(all_chunks)}")
        
        # Concatenate all embeddings into single numpy array
        combined_embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 384))
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self.retriever.build_index(combined_embeddings, all_chunks)
        
        # Save index
        logger.info("Saving index to disk...")
        success = self.retriever.save_index()
        
        if success:
            logger.info("✓ BUILD MODE COMPLETE")
            stats = self.retriever.get_index_stats()
            logger.info(f"  Index statistics: {stats}")
            logger.info("  NOTE: keywords.json was NOT modified (user-controlled only)")
            return True
        else:
            logger.error("Failed to save index")
            return False
    
    # ==================== ANALYSIS MODE ====================
    
    def analyze_documents_from_folder(
        self,
        check_document_folder: str,
        similarity_threshold: float = 0.75,
        top_k: int = 3
    ) -> Dict:
        """
        ANALYSIS MODE: Analyze all PDFs in VerifyDocumentCompliance folder using TOP-K retrieval.
        
        Uses TOP-K retrieval to collect multiple relevant chunks instead of single best match,
        improving accuracy especially for larger documents where context is distributed.
        
        Args:
            check_document_folder: Path to VerifyDocumentCompliance folder
            similarity_threshold: Threshold for weak matches
            top_k: Number of top results to retrieve per keyword (default 3)
            
        Returns:
            Analysis results with keyword-based structure
        """
        logger.info("=" * 70)
        logger.info("ANALYSIS MODE: Analyzing VerifyDocumentCompliance Folder")
        logger.info("=" * 70)
        
        if not os.path.isdir(check_document_folder):
            logger.error(f"VerifyDocumentCompliance folder not found: {check_document_folder}")
            return {}
        
        # Check if index exists
        index_exists = self.retriever.load_index()
        if not index_exists:
            logger.warning("⚠ FAISS index not found. Will use keyword-only matching (no LLM).")
            logger.warning(f"  Expected path: {self.retriever.index_path}")
        else:
            logger.info("✓ FAISS index loaded successfully")
            logger.info(f"  Index path: {self.retriever.index_path}")
            logger.info(f"  Index size: {self.retriever.index.ntotal} vectors")  # type: ignore
        
        # Get keywords
        keywords = self.keyword_manager.get_keywords()
        if not keywords:
            logger.warning("No keywords loaded from keywords.json")
            return {}
        
        # Sort keywords for consistent ordering
        keywords = sorted(keywords)
        
        logger.info(f"Loaded {len(keywords)} keywords from keywords.json")
        
        # Find all PDFs in folder
        pdf_files = glob.glob(os.path.join(check_document_folder, "*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in VerifyDocumentCompliance folder: {check_document_folder}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) to analyze")
        
        # Analyze each document
        all_connections = []  # keyword-per-row results
        all_missing = []      # missing keywords
        
        for pdf_file in pdf_files:
            logger.info(f"\nAnalyzing: {os.path.basename(pdf_file)}")
            filename = os.path.basename(pdf_file)
            
            if not self.pdf_parser.validate_pdf(pdf_file):
                logger.warning(f"Skipping invalid PDF: {pdf_file}")
                continue
            
            # Extract text
            logger.info("  → Extracting text...")
            extracted = self.pdf_parser.extract_text_with_metadata(pdf_file)
            
            # Chunk document
            logger.info("  → Creating chunks...")
            doc_chunks = self.chunker.chunk_pdf_pages(
                extracted["pages"],
                source_file=filename
            )
            
            if not doc_chunks:
                logger.warning(f"No chunks created from: {pdf_file}")
                continue
            
            logger.info(f"  → Created {len(doc_chunks)} chunks")
            
            # Process each keyword
            logger.info(f"  → Processing {len(keywords)} keywords...")
            for keyword in keywords:
                result = self._analyze_keyword_in_document(
                    filename,
                    keyword,
                    doc_chunks,
                    similarity_threshold,
                    top_k,
                    index_exists
                )
                
                if result.get("connection_type") == "missing":
                    all_missing.append(result)
                else:
                    all_connections.append(result)
        
        logger.info(f"\nAnalysis complete:")
        logger.info(f"  - Strong matches: {len([c for c in all_connections if c['connection_type'] == 'strong'])}")
        logger.info(f"  - Weak matches: {len([c for c in all_connections if c['connection_type'] == 'weak'])}")
        logger.info(f"  - Missing keywords: {len(all_missing)}")
        logger.info("  NOTE: FAISS index NOT modified")
        logger.info("  NOTE: keywords.json NOT modified")
        
        return {
            "connections": all_connections,
            "missing_keywords": all_missing,
            "index_exists": index_exists,
            "keywords_count": len(keywords),
            "documents_analyzed": len(pdf_files)
        }
    
    def _analyze_keyword_in_document(
        self,
        filename: str,
        keyword: str,
        doc_chunks: List[Dict],
        similarity_threshold: float,
        top_k: int,
        index_exists: bool
    ) -> Dict:
        """
        Analyze a single keyword in a document with TOP-K retrieval and keyword-aware boosting.
        
        TOP-K LOGIC:
        - Retrieves top_k matches from FAISS for each document chunk containing keyword
        - Applies keyword-aware boosting to each result:
          * Hybrid score = (0.8 * embedding_similarity) + (0.2 * keyword_presence)
          * keyword_presence = 1 if keyword in chunk text, else 0
        - Combines all matches and computes:
          - max_similarity: highest boosted score among all top_k results
          - avg_similarity: average of all boosted scores
        - Combines chunk texts with clear formatting
        - Passes all chunks to LLM for better context
        
        Args:
            filename: Document filename
            keyword: Keyword to analyze
            doc_chunks: Document chunks
            similarity_threshold: Threshold for weak matches
            top_k: Number of top results to retrieve (default 3-4)
            index_exists: Whether FAISS index exists
            
        Returns:
            Result dict for this keyword (one row in output)
        """
        logger.debug(f"Analyzing keyword '{keyword}' in {filename} (index_exists={index_exists}, top_k={top_k})")
        
        # Find chunks containing this keyword
        matching_chunks = []
        for chunk in doc_chunks:
            if keyword.lower() in chunk["chunk_text"].lower():
                matching_chunks.append(chunk)
        
        logger.info(f"  Keyword '{keyword}': Found {len(matching_chunks)} matching chunks")
        
        if not matching_chunks:
            # Keyword not found in document
            logger.info(f"    -> '{keyword}' NOT found in document")
            return {
                "filename": filename,
                "keyword": keyword,
                "connection_type": "missing",
                "similarity_score": None,
                "matched_reference": "",
                "matched_page_number": "",
                "llm_explanation": "",
                "max_similarity": None,
                "avg_similarity": None
            }
        
        # If index exists, find best matches using TOP-K strategy
        if index_exists:
            logger.info(f"    -> Using FAISS for TOP-K={top_k} semantic search")
            all_top_matches = []
            all_scores = []
            
            # For each matching chunk, search for top_k matches in standard
            for chunk_idx, chunk in enumerate(matching_chunks):
                # Generate embedding for this chunk
                embedding = self.embedder.embed_texts([chunk["chunk_text"]])[0]
                
                # Search in FAISS for top_k results
                results = self.retriever.search(embedding, k=top_k)
                logger.debug(f"      Chunk {chunk_idx+1}: FAISS returned {len(results)} results")
                
                if results:
                    # Collect all results and apply keyword-aware boosting
                    for result in results:
                        embedding_sim = result.get('similarity_score', 0)
                        
                        # Apply keyword-aware boosting to this result
                        boosted_score = self._apply_keyword_aware_boosting(
                            result,
                            keyword,
                            embedding_sim
                        )
                        
                        # Update the result with boosted score
                        result['original_similarity_score'] = embedding_sim
                        result['similarity_score'] = boosted_score
                        
                        all_scores.append(boosted_score)
                        all_top_matches.append(result)
            
            logger.info(f"    Total top_k matches collected: {len(all_top_matches)}")
            
            if all_top_matches and all_scores:
                # Calculate similarity metrics (using boosted scores)
                max_similarity = max(all_scores)
                avg_similarity = sum(all_scores) / len(all_scores)
                
                logger.info(
                    f"    Similarity stats (with keyword-aware boosting): "
                    f"max={max_similarity:.4f}, avg={avg_similarity:.4f}"
                )
                
                # Classify using max_similarity as primary score
                connection_type = classify_similarity(max_similarity)
                logger.info(f"    Classification: {connection_type} (max_score={max_similarity:.4f})")
                
                # Combine all top_k chunks into formatted reference
                combined_reference = self._format_combined_chunks(all_top_matches)
                
                # Extract all page numbers from top_k matches (deduplicated, sorted)
                all_page_numbers = self._extract_page_numbers(all_top_matches)
                
                # Generate explanation with all chunks
                llm_explanation = self.explainer.generate_explanation(
                    keyword=keyword,
                    similarity_score=max_similarity,
                    connection_type=connection_type,
                    chunk=matching_chunks[0]["chunk_text"],
                    standard_chunks=all_top_matches  # Pass all top_k chunks
                )
                
                return {
                    "filename": filename,
                    "keyword": keyword,
                    "connection_type": connection_type,
                    "similarity_score": max_similarity,
                    "matched_reference": combined_reference,
                    "matched_page_number": all_page_numbers,
                    "llm_explanation": llm_explanation,
                    "max_similarity": max_similarity,
                    "avg_similarity": avg_similarity
                }
        else:
            logger.info(f"    -> Index does not exist, no FAISS search")
        
        # No index or no match found
        return {
            "filename": filename,
            "keyword": keyword,
            "connection_type": "missing",
            "similarity_score": None,
            "matched_reference": "",
            "matched_page_number": "",
            "llm_explanation": "",
            "max_similarity": None,
            "avg_similarity": None
        }
    
    def _extract_page_numbers(self, chunks: List[Dict]) -> str:
        """
        Extract and combine page numbers from all retrieved chunks.
        
        Args:
            chunks: List of chunk dicts with 'page_number' key
            
        Returns:
            Comma-separated string of unique, sorted page numbers
            Example: "1, 2, 3"
            
        Edge cases:
            - Single match: returns single page number
            - No matches: returns empty string
            - Multiple chunks from same page: deduplicates
        """
        if not chunks:
            return ""
        
        # Extract page numbers from all chunks
        page_numbers = []
        for chunk in chunks:
            page_num = chunk.get("page_number")
            if page_num is not None and page_num != "":
                # Convert to int for sorting (if possible)
                try:
                    page_numbers.append(int(page_num))
                except (ValueError, TypeError):
                    # Keep as string if not convertible
                    page_numbers.append(str(page_num))
        
        if not page_numbers:
            return ""
        
        # Deduplicate and sort
        unique_pages = sorted(set(page_numbers))
        
        # Format as comma-separated string
        result = ", ".join(str(p) for p in unique_pages)
        
        logger.debug(
            f"Extracted page numbers from {len(chunks)} chunks: {result}"
        )
        return result
    
    def _format_combined_chunks(self, chunks: List[Dict]) -> str:
        """
        Format multiple chunks with clear formatting: [Match N]: <text>.
        
        Args:
            chunks: List of chunk dicts with 'chunk_text'
            
        Returns:
            Formatted combined string (limited to ~500 chars)
        """
        if not chunks:
            return ""
        
        formatted_chunks = []
        max_total_length = 500
        current_length = 0
        
        for i, chunk_data in enumerate(chunks, 1):
            # Extract chunk text
            chunk_text = chunk_data.get("chunk_text", "")
            if not chunk_text:
                continue
            
            # Trim each chunk to reasonable preview size (~100 chars per match)
            chunk_preview = chunk_text.strip()[:100]
            
            # Format: "[Match N]: <text>"
            formatted = f"[Match {i}]: {chunk_preview}"
            
            # Check length
            new_length = current_length + len(formatted) + 2  # +2 for newline
            if new_length > max_total_length and formatted_chunks:
                logger.debug(f"Combined chunks length exceeded, stopping at {i-1} matches")
                break
            
            formatted_chunks.append(formatted)
            current_length = new_length
        
        combined = "\n".join(formatted_chunks)
        logger.debug(f"Combined {len(formatted_chunks)} chunks into {len(combined)} characters")
        return combined
    
    def _apply_keyword_aware_boosting(
        self,
        chunk: Dict,
        keyword: str,
        embedding_similarity: float
    ) -> float:
        """
        Apply keyword-aware boosting to embedding similarity score.
        
        Uses hybrid scoring: final_score = (0.8 * embedding_sim) + (0.2 * keyword_match)
        where keyword_match = 1 if keyword is present, 0 otherwise.
        
        Args:
            chunk: Chunk dict with 'chunk_text'
            keyword: Keyword to check for (case-insensitive)
            embedding_similarity: Original embedding similarity score (0-1)
            
        Returns:
            Hybrid score combining embedding similarity and keyword presence (0-1)
        """
        # Check if keyword is present in chunk (case-insensitive)
        chunk_text = chunk.get("chunk_text", "").lower()
        keyword_lower = keyword.lower()
        
        # Binary keyword match score
        keyword_match = 1 if keyword_lower in chunk_text else 0
        
        # Calculate hybrid score
        hybrid_score = KeywordBoostConfig.calculate_hybrid_score(embedding_similarity, keyword_match)
        
        logger.debug(
            f"Keyword '{keyword}': embedding={embedding_similarity:.4f}, "
            f"keyword_present={bool(keyword_match)}, hybrid={hybrid_score:.4f}"
        )
        
        return hybrid_score
    
    # ==================== REPORT GENERATION ====================
    
    def generate_keyword_reports(
        self,
        analysis_result: Dict,
        output_prefix: str = "analysis"
    ) -> Dict[str, str]:
        """
        Generate keyword-based CSV reports.
        
        Args:
            analysis_result: Result from analyze_documents_from_folder
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary mapping report type to file path
        """
        logger.info("Generating keyword-based reports...")
        
        return self.report_gen.generate_keyword_reports(
            analysis_result.get("connections", []),
            analysis_result.get("missing_keywords", []),
            output_prefix
        )


def main():
    """Main CLI entry point - REFACTORED with two modes."""
    
    # ==================== INTERACTIVE MODE ====================
    # Show menu if no arguments provided
    if len(sys.argv) == 1:
        print("\n" + "=" * 70)
        print("DO-178 COMPLIANCE SYSTEM")
        print("=" * 70)
        print("1. Build Index (ReferencePDF)")
        print("2. Analyze Documents (VerifyDocumentCompliance)")
        print("\n" + "=" * 70)
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Build mode with ReferencePDF
            sys.argv = ["main.py", "--mode", "build", "--standard", "ReferencePDF"]
        elif choice == "2":
            # Analyze mode with VerifyDocumentCompliance and verbose enabled
            sys.argv = ["main.py", "--mode", "analyze", "--check", "VerifyDocumentCompliance", "--verbose"]
        else:
            print("Invalid choice. Please enter 1 or 2.")
            sys.exit(1)
    
    # ==================== ARGUMENT PARSING ====================
    parser = argparse.ArgumentParser(
        description="""
DO-178 Compliance System - Refactored Two-Mode Pipeline

MODE 1 - BUILD:     Index reference documents from StandardPDF folder
MODE 2 - ANALYZE:   Analyze documents using keyword mapping against index

Examples:
  python main.py --mode build --standard StandardPDF/
  python main.py --mode analyze --check CheckDocument/ --similarity 0.75
        """
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["build", "analyze"],
        help="Operation mode: 'build' to create index, 'analyze' to check documents"
    )
    
    parser.add_argument(
        "--standard",
        help="StandardPDF folder path (REQUIRED for 'build' mode)"
    )
    
    parser.add_argument(
        "--check",
        help="CheckDocument folder path (REQUIRED for 'analyze' mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for reports (default: outputs)"
    )
    
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.75,
        help="Similarity threshold for weak matches (default: 0.75)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K results to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Ensure required directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("DO-178 COMPLIANCE ANALYSIS SYSTEM - REFACTORED")
    logger.info("=" * 70)
    
    # Initialize analyzer
    config = {
        "output_dir": args.output_dir
    }
    analyzer = DO178ComplianceAnalyzer(config)
    
    # ==================== BUILD MODE ====================
    if args.mode == "build":
        logger.info("\n>>> BUILD MODE SELECTED")
        
        if not args.standard:
            logger.error("--standard folder path is REQUIRED for build mode")
            parser.print_help()
            sys.exit(1)
        
        # Build index from StandardPDF folder
        if analyzer.build_reference_index_from_folder(args.standard):
            logger.info("\n✓ BUILD COMPLETE - Ready for analysis")
            sys.exit(0)
        else:
            logger.error("\n✗ BUILD FAILED")
            sys.exit(1)
    
    # ==================== ANALYZE MODE ====================
    elif args.mode == "analyze":
        logger.info("\n>>> ANALYZE MODE SELECTED")
        
        if not args.check:
            logger.error("--check folder path is REQUIRED for analyze mode")
            parser.print_help()
            sys.exit(1)
        
        # Analyze documents from CheckDocument folder
        analysis_result = analyzer.analyze_documents_from_folder(
            args.check,
            similarity_threshold=args.similarity,
            top_k=args.top_k
        )
        
        if not analysis_result:
            logger.error("\n✗ ANALYSIS FAILED")
            sys.exit(1)
        
        # Generate keyword-based reports
        reports = analyzer.generate_keyword_reports(
            analysis_result,
            output_prefix=os.path.basename(args.check)
        )
        
        if not reports:
            logger.error("Failed to generate reports")
            sys.exit(1)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Documents analyzed: {analysis_result['documents_analyzed']}")
        logger.info(f"Keywords used: {analysis_result['keywords_count']}")
        logger.info(f"Index used: {'Yes' if analysis_result['index_exists'] else 'No (keyword-only matching)'}")
        logger.info(f"\nTotal connections: {len(analysis_result['connections'])}")
        logger.info(f"  - Strong (≥{SimilarityConfig.STRONG_THRESHOLD}): {len([c for c in analysis_result['connections'] if c['connection_type'] == 'strong'])}")
        logger.info(f"  - Weak (≥{SimilarityConfig.WEAK_THRESHOLD} to <{SimilarityConfig.STRONG_THRESHOLD}): {len([c for c in analysis_result['connections'] if c['connection_type'] == 'weak'])}")
        logger.info(f"  - Missing (<{SimilarityConfig.WEAK_THRESHOLD}): {len(analysis_result['missing_keywords'])}")
        logger.info(f"\nReports generated:")
        for report_type, report_path in reports.items():
            logger.info(f"  - {report_type}: {report_path}")
        logger.info("=" * 70)
        logger.info("\n✓ ANALYZE COMPLETE")
        sys.exit(0)


if __name__ == "__main__":
    main()
