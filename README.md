# DO-178 Compliance Analysis System

A fully offline, production-ready DO-178C compliance analysis engine. Analyzes project documents against reference DO-178 compliant PDFs using advanced semantic search with TOP-K retrieval and keyword-aware hybrid scoring.

## ✨ Key Features

- **Fully Offline** - Complete local processing, no cloud dependencies
- **TOP-K Retrieval** - Retrieves multiple relevant chunks (top 3-4) instead of single best match
- **Keyword-Aware Scoring** - Hybrid scoring combining semantic similarity (80%) with keyword presence (20%)
- **Semantic Search** - FAISS vector database for efficient similarity retrieval
- **Embeddings** - SentenceTransformer embeddings for semantic understanding
- **Local LLM** - Ollama integration for explainable compliance analysis (with fallback)
- **Keyword-Based Analysis** - Keywords extracted from DO-178 standard (keywords.json)
- **Production Reports** - Excel and CSV compliance analysis reports
- **REST API** - Flask-based API for programmatic access
- **Modular Architecture** - Clean, extensible component design
- **Comprehensive Logging** - Full audit trail for compliance documentation  

## 🏗️ System Architecture

```
do178-compliance-engine/
├── app/
│   ├── pdf_parser.py              # PDF text extraction and parsing
│   ├── chunker.py                 # Document segmentation
│   ├── embedder.py                # Embedding generation (SentenceTransformer)
│   ├── retriever.py               # FAISS vector database management
│   ├── llm_explainer.py           # Ollama integration for explanations
│   ├── keyword_manager.py         # DO-178 keyword management
│   ├── report_generator.py        # Excel/CSV report generation
│   ├── config.py                  # Configuration classes and thresholds
│   ├── main.py                    # CLI interface (BUILD and ANALYZE modes)
│   └── api.py                     # Flask REST API
│
├── indexes/                       # FAISS index storage
│   ├── reference_index.faiss      # Vector index of reference chunks
│   └── metadata.json              # Chunk metadata and source tracking
│
├── ReferencePDF/                  # Reference documents (BUILD mode input)
├── VerifyDocumentCompliance/      # Documents to analyze (ANALYZE mode input)
├── outputs/                       # Generated reports
├── keywords.json                  # DO-178 keywords (user-managed)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔄 Two-Mode Architecture

### Mode 1: BUILD (Index Reference Documents)

Creates FAISS index from reference DO-178 compliant documents.

```
BUILD MODE
├── Input: ReferencePDF/ folder
├── Process:
│   ├── Extract text from PDFs
│   ├── Chunk documents into segments
│   ├── Generate embeddings for each chunk
│   └── Build FAISS index
└── Output:
    ├── indexes/reference_index.faiss
    └── indexes/metadata.json
```

**Run once** to create the index, then **reuse for all analyses**.

### Mode 2: ANALYZE (Compare Documents Against Reference)

Analyzes project documents using keywords and semantic search against the indexed reference.

```
ANALYZE MODE
├── Input: VerifyDocumentCompliance/ folder + keywords.json
├── Process:
│   ├── Extract text from project PDFs
│   ├── Chunk documents
│   ├── For each keyword:
│   │   ├── Find chunks containing keyword
│   │   ├── Embed those chunks
│   │   ├── TOP-K retrieval: get top_k matches from FAISS
│   │   ├── Apply keyword-aware boosting to each result
│   │   ├── Calculate max_similarity, avg_similarity
│   │   ├── Generate LLM explanation (if weak match)
│   │   └── Output: one row per keyword
│   └── Generate CSV reports
└── Output: keyword-per-row CSV files
```

**Run repeatedly** with different documents or keyword sets.

## 📊 Scoring Pipeline

### Traditional Embedding Similarity (OLD)

```
similarity_score = cosine_similarity(embedding_chunk, embedding_query)
Result: Single best match used
```

### NEW: TOP-K + Keyword-Aware Hybrid Scoring

```
For each chunk in TOP-K results:
  embedding_sim = FAISS_similarity_score
  keyword_match = 1 if keyword in chunk_text else 0
  
  hybrid_score = (0.8 × embedding_sim) + (0.2 × keyword_match)

Final metrics:
  max_similarity = max(hybrid_scores)
  avg_similarity = mean(hybrid_scores)
```

**Example:**
- Chunk contains "aircraft" + embedding_similarity=0.60
  - `hybrid = (0.8 × 0.60) + (0.2 × 1) = 0.76` ✓ Boosted!
- Chunk doesn't contain "aircraft" + embedding_similarity=0.60
  - `hybrid = (0.8 × 0.60) + (0.2 × 0) = 0.48` (no boost)

**Classification:**
- `hybrid_score ≥ 0.80` → **STRONG**
- `0.55 ≤ hybrid_score < 0.80` → **WEAK** (needs review)
- `hybrid_score < 0.55` → **MISSING**

## 📦 Installation

### 1. System Requirements

- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for indexes and models
- Windows/Linux/macOS

### 2. Clone Repository

```bash
git clone <repo-url> do178-compliance-engine
cd do178-compliance-engine
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download Embedding Model

The embedding model (all-MiniLM-L6-v2) downloads automatically on first use:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Downloads ~90MB to `.cache/huggingface/`.

### 5. Configure Ollama (Optional)

For LLM-based explanations, install Ollama:

- **Windows**: https://ollama.ai/download/windows
- **macOS**: https://ollama.ai/download/mac
- **Linux**: https://ollama.ai/download/linux

#### Pull LLM Model

```bash
ollama pull phi
```

#### Start Ollama Server

```bash
ollama serve
```

Runs at `http://localhost:11434` by default.

**Note:** If Ollama unavailable, system uses rule-based fallback explanations automatically.

## 🚀 Usage

### Quick Start (Interactive Mode)

```bash
cd app
python main.py
```

Menu prompts for:
1. Build Index (processes ReferencePDF/)
2. Analyze Documents (processes VerifyDocumentCompliance/)

### Option 1: Build Reference Index

```bash
cd app
python main.py --mode build --standard ReferencePDF
```

**Output:**
- `indexes/reference_index.faiss` - FAISS vector index
- `indexes/metadata.json` - Chunk metadata

**One-time setup** - index persists and can be reused for all analyses.

### Option 2: Analyze Documents

```bash
cd app
python main.py --mode analyze --check VerifyDocumentCompliance --verbose
```

**Configuration Options:**
```bash
--mode analyze                        # ANALYZE mode
--check FOLDER                        # Folder with PDFs to analyze
--top-k 3                             # Top-K retrieval (default: 3)
--similarity-threshold 0.75           # Not used in new scoring, kept for compatibility
--verbose                             # Enable debug logging
--output-dir outputs                  # Output directory for CSVs
```

**Output CSV Files:**
- `VerifyDocumentCompliance_connections_[timestamp].csv` - Matched keywords (strong + weak)
- `VerifyDocumentCompliance_missing_connections_[timestamp].csv` - Missing keywords

### CSV Output Format

**Connections CSV (keyword-per-row):**

| Column | Description |
|--------|-------------|
| filename | Document filename |
| keyword | DO-178 keyword analyzed |
| connection_type | strong / weak / missing |
| similarity_score | max_similarity (primary scoring metric) |
| matched_reference | Combined top-K chunks formatted as [Match 1]: ..., [Match 2]: ... |
| matched_page_number | All page numbers from top-K matches (deduplicated, sorted), comma-separated |
| max_similarity | Maximum score from top-K results (0-1) |
| avg_similarity | Average score from top-K results (0-1) |
| llm_explanation | LLM-generated explanation (only for weak matches) |

**Missing CSV:**

| Column | Description |
|--------|-------------|
| filename | Document filename |
| keyword | Missing DO-178 keyword |
| status | Always "missing" |
| notes | Additional notes |

### Example Workflow

```bash
# 1. Prepare reference documents
cp ~/DO-178-references/*.pdf ReferencePDF/

# 2. Build index (one-time)
cd app
python main.py --mode build --standard ReferencePDF
# Creates: indexes/reference_index.faiss, indexes/metadata.json

# 3. Place project documents
cp ~/projects/*.pdf VerifyDocumentCompliance/

# 4. Analyze documents
python main.py --mode analyze --check VerifyDocumentCompliance

# 5. Review outputs
ls ../outputs/
# → VerifyDocumentCompliance_connections_20260526_120000.csv
# → VerifyDocumentCompliance_missing_connections_20260526_120000.csv
```

## ⚙️ Configuration

### Similarity Thresholds

Edit `app/config.py`:

```python
class SimilarityConfig:
    STRONG_THRESHOLD = 0.80   # Score ≥ 0.80 = STRONG
    WEAK_THRESHOLD = 0.55     # 0.55 ≤ Score < 0.80 = WEAK
                              # Score < 0.55 = MISSING
```

### Keyword-Aware Boosting Weights

Edit `app/config.py`:

```python
class KeywordBoostConfig:
    EMBEDDING_WEIGHT = 0.8    # 80% weight on embedding similarity
    KEYWORD_WEIGHT = 0.2      # 20% weight on keyword presence
    
    # Weights must sum to 1.0 (enforced by assertion)
```

**Adjust weights to:**
- Increase KEYWORD_WEIGHT (e.g., 0.3) → Favor exact keyword matches
- Decrease KEYWORD_WEIGHT (e.g., 0.1) → Favor semantic similarity

### Chunking Parameters

Edit `app/main.py` (in `__init__` method):

```python
self.chunker = DocumentChunker(
    min_chunk_size=self.config.get("min_chunk_size", 60),      # Minimum words
    max_chunk_size=self.config.get("max_chunk_size", 120),     # Maximum words
    overlap=self.config.get("overlap", 25)                      # Overlap words
)
```

### TOP-K Retrieval

Adjust in analyze call:

```bash
python main.py --mode analyze --check VerifyDocumentCompliance --top-k 4
```

Default: `top_k=3` (retrieves 3 best matches per keyword)

### Embedding Model

Edit `app/embedder.py`:

```python
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 22MB, 384-dim, fast
```

Alternatives:
- `all-MiniLM-L12-v2` - Higher quality, slower (33MB)
- `all-mpnet-base-v2` - Best quality, slower (80MB)

### LLM Model

Edit `app/llm_explainer.py`:

```python
self.model_name = "phi"  # Lightweight, ~3B params
```

Alternatives:
- `mistral:7b` - Better quality, more capable


## 🌐 REST API

### Start API Server

```bash
cd app
python api.py
```

Runs at `http://localhost:5000`

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "DO-178 Compliance Analysis",
  "index_loaded": true,
  "index_stats": {
    "index_size": 500,
    "metadata_count": 500
  }
}
```

#### Upload Reference Documents
```bash
curl -X POST http://localhost:5000/upload-reference \
  -F "files=@reference1.pdf" \
  -F "files=@reference2.pdf"
```

#### Analyze Document
```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@project.pdf" \
  -F "top_k=3"
```

#### Download Report
```bash
curl -o report.csv \
  http://localhost:5000/download/VerifyDocumentCompliance_connections_20260526_120000.csv
```

#### System Information
```bash
curl http://localhost:5000/system-info
```

## 📝 Keywords Management

### DO-178 Keywords (keywords.json)

User-managed keyword list. The system:
- **Reads** keywords from `keywords.json`
- **Never modifies** `keywords.json` automatically
- **Matches** keywords case-insensitively in document text

### Edit Keywords

Edit `keywords.json` manually:

```json
{
  "do178_keywords": [
    "traceability",
    "requirement",
    "verification",
    "validation",
    "compliance",
    "safety-critical",
    "aircraft",
    "system requirement",
    ...
  ],
  "description": "DO-178 compliance keywords (READ-ONLY). Edit manually only.",
  "last_updated": "2026-05-26"
}
```

### Add Custom Keywords

Simply add to the `do178_keywords` array and save.

## 🔧 Advanced Topics

### Custom Scoring Formula

The hybrid scoring formula can be customized in `app/config.py`:

```python
class KeywordBoostConfig:
    EMBEDDING_WEIGHT = 0.8      # 80% weight on embedding similarity
    KEYWORD_WEIGHT = 0.2        # 20% weight on keyword presence
    
    @staticmethod
    def calculate_hybrid_score(embedding_similarity, keyword_match):
        """
        hybrid_score = (EMBEDDING_WEIGHT × embedding_sim) + 
                       (KEYWORD_WEIGHT × keyword_match)
        
        Args:
            embedding_similarity: float [0, 1] from FAISS
            keyword_match: 1 if keyword in chunk, 0 otherwise
            
        Returns:
            hybrid_score: float [0, 1] (clipped to valid range)
        """
        _weight_sum = (KeywordBoostConfig.EMBEDDING_WEIGHT + 
                      KeywordBoostConfig.KEYWORD_WEIGHT)
        assert _weight_sum == 1.0, "Weights must sum to 1.0"
        
        hybrid = (KeywordBoostConfig.EMBEDDING_WEIGHT * embedding_similarity +
                  KeywordBoostConfig.KEYWORD_WEIGHT * keyword_match)
        return min(max(hybrid, 0.0), 1.0)
```

**Example customizations:**
```python
# Favor keyword presence (strict matching)
EMBEDDING_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# Favor semantic similarity
EMBEDDING_WEIGHT = 0.95
KEYWORD_WEIGHT = 0.05
```

### Page Number Extraction (All Matches)

Page numbers are extracted from all top-K retrieved matches:

```python
# In _analyze_keyword_in_document():
all_page_numbers = self._extract_page_numbers(all_top_matches)
```

**Behavior:**
- Extracts page_number from each top-K chunk metadata
- Deduplicates (if multiple chunks from same page)
- Sorts in ascending numerical order
- Formats as comma-separated string: `"1, 2, 3"`

**Examples:**
- Top-3 matches from pages [5, 5, 10] → Output: `"5, 10"`
- Single match from page 3 → Output: `"3"`
- No matches → Output: `""`

### Programmatic Usage

```python
from app.main import ComplianceAnalyzer

analyzer = ComplianceAnalyzer()

# Build index (one-time)
analyzer.build_reference_index()

# Analyze multiple documents
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for doc in documents:
    analyzer.analyze_documents_from_folder(
        standard_folder="VerifyDocumentCompliance",
        search_by_keywords=True,
        top_k=3
    )
```

### GPU Acceleration

For NVIDIA GPUs with CUDA:

```bash
# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install GPU-enabled FAISS
pip install faiss-gpu
```

The system automatically detects and uses GPU when available.

## 📊 Architecture Details

### PDF Parsing & Chunking
- **Tool**: PyMuPDF for fast PDF text extraction
- **Chunking**: 60-120 word chunks with configurable overlap (25 words default)
- **Metadata**: Preserves page numbers and source document tracking

### Embedding Pipeline
- **Model**: all-MiniLM-L6-v2 (384 dimensions, 22MB)
- **Framework**: sentence-transformers
- **Computation**: Efficient batch processing

### Vector Search (FAISS)
- **Index Type**: IndexFlatIP (inner product for normalized embeddings)
- **TOP-K**: Returns top_k=3 closest matches (configurable)
- **Retrieval**: Linear search optimized for accuracy over speed

### Keyword-Aware Boosting
- **Case-Insensitive Matching**: `keyword.lower() in chunk_text.lower()`
- **Binary Weight**: 1.0 if keyword found, 0.0 otherwise
- **Hybrid Score**: Combines embedding + keyword presence

### LLM Integration (Ollama)
- **Server**: Local Ollama instance (no cloud)
- **Model**: Phi (lightweight, 7B parameters, fallback: Mistral)
- **Usage**: Generates explanations only for WEAK matches
- **Fallback**: Rule-based explanations if Ollama unavailable

### Report Generation
- **Format**: CSV with keyword-per-row structure
- **Columns**: filename, keyword, connection_type, similarity_score, matched_reference, matched_page_number (all pages from top-K), max_similarity, avg_similarity, llm_explanation
- **Page Numbers**: Extracted from all top-K matches, deduplicated, sorted ascending (e.g., "1, 2, 3")
- **Sorting**: By filename, then connection_type (strong→weak→missing)

## 🐛 Troubleshooting

### Cannot Connect to Ollama
**Error**: `ConnectionError: Failed to connect to Ollama`

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

If Ollama is unavailable, the system automatically falls back to rule-based explanations.

### FAISS Index Not Found
**Error**: `FileNotFoundError: indexes/reference_index.faiss`

**Solution**: Build the reference index first:
```bash
cd app
python main.py --mode build --standard ReferencePDF
```

### Embedding Model Not Found
**Error**: `Model not found: all-MiniLM-L6-v2`

**Solution**: Manually download the model:
```bash
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"
```

### Out of Memory
**Error**: `MemoryError` or `CUDA out of memory`

**Solutions**:
1. Reduce chunk size: `max_chunk_size=80`
2. Reduce TOP-K: `--top-k 2`
3. Use CPU-only mode: `export CUDA_VISIBLE_DEVICES=-1`
4. Close other applications

### CSV Not Generated
**Check**:
1. Input folder exists: `ls VerifyDocumentCompliance/`
2. PDFs are readable (not corrupted)
3. Output folder is writable: `ls -l outputs/`
4. Run with `--verbose` for debug logs

### Slow Analysis
**Diagnosis**: Run with `--verbose` to identify bottleneck:
```bash
python main.py --mode analyze --check VerifyDocumentCompliance --verbose
```

**Solutions by bottleneck**:
- **PDF parsing slow**: Split large PDFs, increase chunk size
- **Embedding slow**: Enable GPU support, use lighter model
- **FAISS search slow**: Reduce `--top-k`, reduce index size
- **LLM slow**: Use lighter Ollama model, check system resources

## 📋 Compliance & Auditability

### Design Principles
1. **Deterministic**: Same input → same output every time
2. **Traceable**: Every match linked to source chunks and page numbers
3. **Explainable**: LLM generates human-readable explanations
4. **Auditable**: Comprehensive CSV logging with all metrics
5. **Local**: No external dependencies or data transmission

### Audit Trail
- Logs in `compliance_analysis.log` (if present)
- CSV timestamps for every analysis run
- Source chunk references preserved in reports
- Similarity metrics documented for validation
- Configuration tracked in `app/config.py`

## 📡 API Response Format

All REST API responses follow this format:

```json
{
  "status": "success|error",
  "message": "Human-readable message",
  "data": {
    "index_stats": {...},
    "analysis_results": {...}
  }
}
```

### Example: Health Check Response
```json
{
  "status": "healthy",
  "service": "DO-178 Compliance Analysis Engine",
  "index_loaded": true,
  "index_stats": {
    "index_size": 500,
    "metadata_count": 500,
    "model_name": "all-MiniLM-L6-v2"
  }
}
```

## ⚠️ Known Limitations

1. **LLM Optional**: Ollama helpful for explanations but not required (fallback available)
2. **Keyword Sensitivity**: Case-insensitive matching may match partial words (e.g., "requirement" matches "required")
3. **Chunk Size Trade-off**: Larger chunks → fewer embeddings (faster) but less precision
4. **PDF Complexity**: Complex PDFs with special formatting may not extract perfectly
5. **Language**: Optimized for English documents
6. **TOP-K Trade-off**: Higher top_k → more context but slower retrieval

## 🚀 Future Enhancements

- Multi-language support (French, German, etc.)
- Batch analysis API endpoint
- Advanced filtering and sorting in CSV output
- Web dashboard UI for visualization
- Integration with document management systems (Sharepoint, etc.)
- Custom embedding model support
- Parallel document processing
- Caching layer for faster re-analysis

## 📞 Support & Troubleshooting

### Common Issues

**Q: Reports not generating?**  
A: Check that output folder exists: `mkdir -p outputs/`

**Q: Keywords not matching?**  
A: Verify keywords.json exists and is valid JSON. Keywords are matched case-insensitively.

**Q: Slow performance?**  
A: Enable GPU acceleration or increase chunk size to reduce embedding count.

**Q: Memory errors?**  
A: Reduce --top-k to 2, or reduce max_chunk_size in config.py.

### Debug Mode

Run with verbose logging:
```bash
python main.py --mode analyze --check VerifyDocumentCompliance --verbose
```

Logs include:
- PDF parsing details
- Chunk creation stats
- Embedding times
- FAISS retrieval details
- Boosting calculations
- LLM calls (if enabled)

## 📚 References

- **DO-178C**: Software Considerations in Airborne Systems and Equipment Certification
- **FAISS**: Facebook AI Similarity Search https://faiss.ai/
- **Ollama**: Local LLM Inference https://ollama.ai/
- **sentence-transformers**: https://www.sbert.net/
- **PyMuPDF**: PDF Text Extraction https://pymupdf.readthedocs.io/

---

**Version**: 2.0.0  
**Last Updated**: 2026-05-26  
**Status**: Production-Ready  
**Python**: 3.9+
