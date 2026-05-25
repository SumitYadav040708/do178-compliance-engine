import logging
import re
from typing import List, Dict
from typing import Any as any

logger = logging.getLogger(__name__)


class DocumentChunker:

    def __init__(
        self,
        min_chunk_size: int = 150,
        max_chunk_size: int = 300,
        overlap: int = 50
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

        logger.info(
            f"Chunker initialized: min={min_chunk_size}, max={max_chunk_size}, overlap={overlap}"
        )

    def chunk_text(
        self,
        text: str,
        source_file: str,
        page_number: int = 1,
        section_title: str = "Unknown",
        chunk_id_prefix: str = "chunk"
    ) -> List[Dict[str, any]]:

        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking from {source_file}")
            return []
        
        # Process entire page text directly
        major_chunks = [text]

        fine_chunks = []

        for major_chunk in major_chunks:
            if self._word_count(major_chunk) > self.max_chunk_size:
                fine_chunks.extend(self._split_by_sentences(major_chunk))
            else:
                fine_chunks.append(major_chunk)

        chunks_with_metadata = []
        chunk_counter = 1

        for chunk_text in fine_chunks:
            word_count = self._word_count(chunk_text)

            if word_count < self.min_chunk_size and chunk_counter > 1:
                if chunks_with_metadata:
                    chunks_with_metadata[-1]["chunk_text"] += " " + chunk_text
                    chunks_with_metadata[-1]["word_count"] += word_count
                continue

            chunk_dict = {
                "chunk_id": f"{chunk_id_prefix}_{chunk_counter:05d}",
                "source_file": source_file,
                "page_number": page_number,
                "section_title": section_title,
                "chunk_text": chunk_text.strip(),
                "word_count": word_count,
                "character_count": len(chunk_text)
            }

            chunks_with_metadata.append(chunk_dict)
            chunk_counter += 1

        logger.info(
            f"Created {len(chunks_with_metadata)} chunks from {source_file} "
            f"(section: {section_title})"
        )

        return chunks_with_metadata

    def chunk_pdf_pages(
        self,
        pages_data: List[Dict],
        source_file: str
    ) -> List[Dict[str, any]]:

        chunks = []

        for page_data in pages_data:
            page_text = page_data.get("text", "")
            page_num = page_data.get("page_number", 1)

            if not page_text.strip():
                continue

            page_chunks = self.chunk_text(
                text=page_text,
                source_file=source_file,
                page_number=page_num,
                section_title=f"Page {page_num}",
                chunk_id_prefix=f"{source_file}_p{page_num}"
            )

            chunks.extend(page_chunks)

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks using robust PDF-aware sentence detection."""
        # Handle PDF formatting: split on newlines, periods, exclamation, question marks
        sentences = re.split(r'[\n.!?]+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:  # Skip empty strings
                continue

            word_count = len(sentence.split())

            # If adding this sentence exceeds max size, save current chunk
            if current_size + word_count > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += word_count

        # Don't forget final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())