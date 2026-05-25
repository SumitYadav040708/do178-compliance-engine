"""
PDF Parser Module
Extracts text from PDF files with page-level metadata preservation.
Uses PyMuPDF for efficient text extraction.
"""

import logging
import os
from typing import List, Dict, Optional, Any
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parses PDF documents and extracts structured text with metadata.
    
    Attributes:
        min_confidence: Minimum confidence threshold for text extraction (0-1)
    """
    
    def __init__(self, min_confidence: float = 0.0):
        """
        Initialize PDF Parser.
        
        Args:
            min_confidence: Confidence threshold for text extraction
        """
        self.min_confidence = min_confidence
    
    def extract_text_with_metadata(
        self, 
        pdf_path: str
    ) -> Dict[str, Any]:
        """
        Extract text from PDF file with comprehensive metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing:
                - text: Full extracted text
                - pages: List of page-level data
                - metadata: Document metadata
                - total_pages: Total page count
                
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF cannot be read
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            logger.info(f"Opening PDF: {pdf_path}")
            pdf_document = fitz.Document(pdf_path)
            
            # Extract metadata
            metadata = pdf_document.metadata
            total_pages = len(pdf_document)
            
            logger.info(f"PDF loaded successfully. Total pages: {total_pages}")
            
            # Extract text by page
            pages_data = []
            full_text = []
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                page_text = str(page.get_text())  # type: ignore
                
                # Extract page metadata
                page_data = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "length": len(page_text),
                    "has_content": bool(page_text.strip())
                }
                
                pages_data.append(page_data)
                full_text.append(page_text)
                
                logger.debug(f"Extracted page {page_num + 1}: {len(page_text)} characters")
            
            pdf_document.close()
            
            result = {
                "text": "\n\n".join(full_text),
                "pages": pages_data,
                "metadata": metadata or {},
                "total_pages": total_pages,
                "file_name": os.path.basename(pdf_path),
                "file_path": pdf_path
            }
            
            logger.info(f"Successfully extracted text from {total_pages} pages")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Cannot read PDF file: {str(e)}")
    
    def extract_sections(
        self,
        pdf_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract text organized by sections/chapters.
        Attempts to identify section boundaries based on structure.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of section dictionaries with structure metadata
        """
        extracted = self.extract_text_with_metadata(pdf_path)
        sections = []
        current_section = None
        section_counter = 1
        
        for page_data in extracted["pages"]:
            page_text = page_data["text"]
            
            # Simple heuristic: detect section starts (lines starting with digits/Roman numerals)
            lines = page_text.split("\n")
            
            for line in lines:
                stripped = line.strip()
                
                # Detect potential section header
                if self._is_section_header(stripped) and current_section is None:
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        "section_id": section_counter,
                        "title": stripped,
                        "page_start": page_data["page_number"],
                        "content": [],
                        "page_end": page_data["page_number"]
                    }
                    section_counter += 1
                
                elif current_section and stripped:
                    current_section["content"].append(stripped)
                    current_section["page_end"] = page_data["page_number"]
        
        # Add final section
        if current_section:
            current_section["text"] = "\n".join(current_section["content"])
            current_section.pop("content", None)
            sections.append(current_section)
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    @staticmethod
    def _is_section_header(line: str) -> bool:
        """
        Heuristic to detect if a line is a section header.
        
        Args:
            line: Line of text to check
            
        Returns:
            True if likely a section header, False otherwise
        """
        if len(line) > 100:
            return False
        
        # Check for numbering patterns (1.1, Chapter 1, Section A, etc.)
        patterns = [
            line[0].isdigit(),
            line.lower().startswith(("chapter", "section", "part", "clause")),
            any(c.isupper() for c in line) and len(line.split()) < 8
        ]
        
        return any(patterns)
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate if file is a readable PDF.
        
        Args:
            pdf_path: Path to file to validate
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                return False
            
            pdf_document = fitz.Document(pdf_path)
            is_valid = len(pdf_document) > 0
            pdf_document.close()
            return is_valid
            
        except Exception as e:
            logger.warning(f"PDF validation failed: {str(e)}")
            return False
