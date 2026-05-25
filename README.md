# DO-178 Compliance Analysis System

A fully offline DRDO-compatible DO-178 compliance analysis system built with Python. Analyzes project documents against reference DO-178 compliant PDFs using semantic search, embeddings, and local LLM reasoning.

## Features

**Fully Offline** - Works entirely locally after setup, no cloud dependencies  
 **Semantic Search** - Uses FAISS for efficient vector similarity retrieval  
**Embeddings** - Leverages sentence-transformers for semantic understanding  
**Local LLM** - Integrates Ollama Qwen 2.5 3B for explainable compliance analysis

 **Production Reports** - Generates Excel and CSV compliance analysis reports  
 **REST API** - Flask-based API for programmatic access  
 **Modular Architecture** - Clean, extensible component design  
 **Logging & Auditability** - Comprehensive logging for compliance documentation  

## System Architecture

```
do178-compliance-engine/
├── app/
│   ├── pdf_parser.py          # PDF text extraction and parsing
│   ├── chunker.py              # Document segmentation
│   ├── embedder.py             # Embedding generation (sentence-transformers)
│   ├── retriever.py            # FAISS vector database management
│   ├── llm_explainer.py        # Ollama integration for explanations
│   ├── report_generator.py     # Excel/CSV report generation
│   ├── main.py                 # CLI interface
│   └── api.py                  # Flask REST API
│
├── indexes/                    # FAISS index storage
│   ├── reference_index.faiss   # Vector index
│   └── metadata.json           # Chunk metadata
│
├── input_docs/                 # Reference documents
├── outputs/                    # Generated reports
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### 1. System Requirements

- Python 3.11
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for indexes and models
- Windows/Linux/macOS

### 2. Install Python Dependencies

```bash
cd d:\Work\do178-compliance-engine

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Embedding Model

The embedding model (all-MiniLM-L6-v2) will be automatically downloaded on first use:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

This downloads ~90MB model locally to `.cache/huggingface/`.

### 4. Install and Configure Ollama (Optional)

For LLM-based explanations, install Ollama:

**Windows:** https://ollama.ai/download/windows  
**macOS:** https://ollama.ai/download/mac  
**Linux:** https://ollama.ai/download/linux

#### Pull Qwen Model

```bash
ollama pull qwen2.5:3b
```

#### Start Ollama Server

```bash
ollama serve
```

The server runs at `http://localhost:11434` by default.

**Note:** If Ollama is not available, the system falls back to rule-based explanations.

## Usage

### Option 1: Command-Line Interface (CLI)

#### Step 1: Prepare Reference Documents

Place reference DO-178 compliant PDFs in `input_docs/`:

```bash
cp reference_documents/*.pdf input_docs/
```

#### Step 2: Build Reference Index

Build FAISS index from reference documents (one-time setup):

```bash
cd app

python main.py --build-index \
  --reference-docs ../input_docs/reference1.pdf ../input_docs/reference2.pdf \
  --output-dir ../outputs
```

This creates:
- `indexes/reference_index.faiss` - Vector index
- `indexes/metadata.json` - Chunk metadata

#### Step 3: Analyze Project

Analyze your project PDF:

```bash
python main.py path/to/project.pdf \
  --similarity-threshold 0.75 \
  --top-k 5 \
  --output-dir ../outputs
```

#### Step 4: Review Reports

Generated reports in `outputs/`:

- `project_YYYYMMDD_HHMMSS_report.xlsx` - Main compliance report
- `project_YYYYMMDD_HHMMSS_strong_matches.csv` - Strong matches
- `project_YYYYMMDD_HHMMSS_weak_matches.csv` - Weak matches
- `project_YYYYMMDD_HHMMSS_missing_clauses.csv` - Missing clauses
- `project_YYYYMMDD_HHMMSS_summary.json` - Analysis summary

### Option 2: REST API

#### Start API Server

```bash
cd app
python api.py
```

API runs at `http://localhost:5000`

#### API Endpoints

**Health Check**
```bash
curl http://localhost:5000/health
```

**Upload Reference Documents**
```bash
curl -X POST http://localhost:5000/upload-reference \
  -F "files=@reference1.pdf" \
  -F "files=@reference2.pdf"
```

**Analyze Project**
```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@project.pdf" \
  -F "similarity_threshold=0.75" \
  -F "top_k=5"
```

**Download Report**
```bash
curl -o report.xlsx \
  http://localhost:5000/download-report/excel/project_20240101_120000_report.xlsx
```

**List Reports**
```bash
curl http://localhost:5000/reports
```

**Index Statistics**
```bash
curl http://localhost:5000/index-stats
```

**System Information**
```bash
curl http://localhost:5000/system-info
```

### Command-Line Parameters

```
usage: main.py [-h] [--build-index] [--reference-docs REFERENCE_DOCS [REFERENCE_DOCS ...]]
               [--output-dir OUTPUT_DIR] [--similarity-threshold SIMILARITY_THRESHOLD]
               [--top-k TOP_K] [--verbose]
               project_pdf

positional arguments:
  project_pdf               Path to project PDF for analysis

optional arguments:
  -h, --help               Show help message
  --build-index            Build reference index from reference documents
  --reference-docs ...     Paths to reference DO-178 compliant PDFs
  --output-dir DIR         Output directory for reports (default: outputs)
  --similarity-threshold   Threshold for weak matches (default: 0.75)
  --top-k                  Top K results to retrieve (default: 5)
  --verbose                Enable verbose logging
```

## Output Reports

### Strong Matches Sheet
Compliance requirements with high confidence (> 0.90 similarity)

| Column | Description |
|--------|-------------|
| Project File | Name of project document |
| Project Section | Section from project |
| Page Number | Page number in project |
| Matched Clause | ID of matched reference clause |
| Reference Section | Section from reference |
| Similarity Score | Semantic similarity (0-1) |
| Explanation | LLM-generated compliance explanation |

### Weak Matches Sheet
Requirements with potential relevance (0.75-0.90 similarity)

| Column | Description |
|--------|-------------|
| Project File | Project document name |
| Project Section | Project section |
| Candidate Clause | Partially matching clause |
| Similarity Score | Semantic similarity score |
| Explanation | Compliance explanation |

### Missing Clauses Sheet
DO-178 compliance areas not found in project

| Column | Description |
|--------|-------------|
| Clause | DO-178 clause identifier |
| Section | Section name |
| Status | Compliance status |
| Notes | Additional notes |

## Configuration

### Chunking Parameters

Edit in `app/chunker.py`:
```python
self.min_chunk_size = 100      # Minimum words per chunk
self.max_chunk_size = 500      # Maximum words per chunk
self.overlap = 50              # Word overlap between chunks
```

### Embedding Model

Edit in `app/embedder.py`:
```python
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast, good quality
```

Alternative models:
- `all-MiniLM-L12-v2` - Higher quality, slower
- `all-mpnet-base-v2` - Better quality, 80MB

### LLM Model

Edit in `app/llm_explainer.py`:
```python
model_name = "qwen2.5:3b"  # Lightweight Qwen model
```

Alternative Ollama models:
- `qwen2.5:7b` - Better quality, more resource-intensive
- `mistral:7b` - Alternative model
- `neural-chat:7b` - Specialized for Q&A

### Similarity Thresholds

Thresholds can be configured via CLI:
```bash
python main.py project.pdf --similarity-threshold 0.80
```

- `> 0.90` - Strong match (high confidence)
- `0.75-0.90` - Weak match (needs review)
- `< 0.75` - No match or insufficient alignment

## Workflow

### First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup Ollama (optional)
ollama pull qwen2.5:3b
ollama serve  # Run in background

# 3. Build reference index (one-time)
cd app
python main.py --build-index --reference-docs ../input_docs/ref1.pdf ../input_docs/ref2.pdf

# This creates:
# - indexes/reference_index.faiss
# - indexes/metadata.json
```

### Recurring Analysis

```bash
# Analyze new project against existing index
cd app
python main.py ../input_docs/new_project.pdf

# Reports generated in outputs/
```

### Reuse Index Across Runs

The FAISS index persists between runs:
- `indexes/reference_index.faiss` - Reusable across all analyses
- `indexes/metadata.json` - Metadata for embeddings

No need to rebuild index for each project analysis.

## Logging

Logs are written to:
- **Console** - Real-time feedback
- **compliance_analysis.log** - Detailed log file

Enable verbose logging:
```bash
python main.py project.pdf --verbose
```

## Performance Considerations

### Memory Usage
- Embedding model: ~200MB
- FAISS index: ~10-50MB per 1000 reference chunks
- Ollama Qwen 3B: ~6GB GPU / ~8GB RAM (CPU mode)

### Processing Speed
- PDF parsing: ~100ms per page
- Embedding generation: ~50-100ms per page (CPU)
- FAISS search: ~5-10ms per query
- LLM explanation: ~2-5 seconds per explanation (depends on Ollama)

### Optimization Tips
1. **Use GPU for embeddings**: FAISS and sentence-transformers support GPU
2. **Batch processing**: Process multiple chunks in batch
3. **Increase chunk size**: Larger chunks = fewer embeddings to compute
4. **Reduce top_k**: Fewer results = faster retrieval

## Troubleshooting

### Issue: "Cannot connect to Ollama"
**Solution:** Ensure Ollama is running:
```bash
ollama serve
```
If not installed, the system falls back to rule-based explanations.

### Issue: "FAISS index not found"
**Solution:** Build the reference index first:
```bash
python main.py --build-index --reference-docs reference1.pdf reference2.pdf
```

### Issue: "Model not found"
**Solution:** Download embedding model:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: Out of Memory
**Solutions:**
1. Reduce chunk size in `chunker.py`
2. Use CPU-only mode instead of GPU
3. Increase `--top-k` parameter (fewer results to compute)

### Issue: Slow Analysis
**Solutions:**
1. Enable GPU support
2. Reduce similarity threshold
3. Use smaller reference index

## Architecture Details

### PDF Parsing
- **Tool**: PyMuPDF (fitz) - Fast, reliable PDF text extraction
- **Output**: Structured text with page numbers and metadata

### Chunking
- **Strategy**: Semantic-aware chunking with clause detection
- **Size**: 300-500 words or clause-level chunks
- **Metadata**: Preserves page numbers, section titles, source file

### Embeddings
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Framework**: sentence-transformers
- **Computation**: Batch processing for efficiency

### Vector Database
- **Index**: FAISS IndexFlatIP (inner product for normalized vectors)
- **Persistence**: Saves embeddings and metadata to disk
- **Retrieval**: O(n) linear search optimized for accuracy

### LLM Integration
- **Server**: Ollama (local, no cloud)
- **Model**: Qwen 2.5 3B (lightweight, 3B parameters)
- **Prompt**: Evidence-based, constrained to retrieved chunks
- **Fallback**: Rule-based explanations if Ollama unavailable

### Report Generation
- **Format**: Excel with formatting, CSV for data import
- **Sheets**: Strong matches, weak matches, missing clauses, summary
- **Features**: Color-coded, filterable, sortable

## Compliance & Auditability

### Design Principles
1. **Deterministic** - Same input produces same output
2. **Traceable** - Every match linked to source chunks
3. **Explainable** - LLM generates human-readable explanations
4. **Auditable** - Comprehensive logging of all operations
5. **Local** - No external dependencies or data transmission

### Audit Trail
- Detailed logs in `compliance_analysis.log`
- Report metadata includes timestamps
- Source chunk references preserved in output
- Similarity scores documented for validation

## API Documentation

### Response Format

All API responses follow this format:

```json
{
  "status": "success|error",
  "message": "Human-readable message",
  "data": {}
}
```

### Example: Upload and Analyze

```bash
# 1. Upload references
curl -X POST http://localhost:5000/upload-reference \
  -F "files=@compliance_ref.pdf"

# 2. Analyze project
curl -X POST http://localhost:5000/analyze \
  -F "file=@my_project.pdf" \
  -F "similarity_threshold=0.75"

# Response:
{
  "status": "success",
  "results": {
    "strong_matches": 42,
    "weak_matches": 15,
    "missing_clauses": 3,
    "project_chunks_total": 60,
    "reference_chunks_total": 500
  },
  "report_files": {
    "excel": "project_20240101_120000_report.xlsx"
  }
}
```

## Advanced Usage

### Programmatic Analysis

```python
from app.main import DO178ComplianceAnalyzer

# Initialize
analyzer = DO178ComplianceAnalyzer()

# Build index
analyzer.build_reference_index([
    "reference1.pdf",
    "reference2.pdf"
])

# Analyze
result = analyzer.analyze_project(
    "project.pdf",
    similarity_threshold=0.75,
    top_k=5
)

# Generate reports
reports = analyzer.generate_reports(result, "my_project")
```

### Custom Configuration

```python
config = {
    "min_chunk_size": 200,
    "max_chunk_size": 800,
    "output_dir": "custom_outputs",
    "device": "cuda"  # Use GPU
}

analyzer = DO178ComplianceAnalyzer(config)
```

## Known Limitations

1. **LLM Dependency**: Ollama/Qwen optional but helpful for explanations
2. **Chunk Size Trade-off**: Larger chunks = fewer embeddings but less precision
3. **Text Extraction**: Complex PDFs with special formatting may not extract perfectly
4. **Language**: Optimized for English documents

## Future Enhancements

- Multi-language support
- Batch analysis API
- Advanced filtering and sorting
- Dashboard UI
- Integration with document management systems
- Custom embedding models

## Support

For issues, errors, or questions:

1. Check `compliance_analysis.log` for detailed error messages
2. Verify requirements are installed: `pip show -r requirements.txt`
3. Ensure reference index is built
4. Run with `--verbose` for debug output

## License

This system is designed for DRDO and defense organization use. Internal use only.

## References

- DO-178C: Software Considerations in Airborne Systems and Equipment Certification
- FAISS: Facebook AI Similarity Search
- Ollama: Local LLM inference engine
- sentence-transformers: State-of-the-art semantic embeddings

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-01  
**Built with Python 3.11.9**
