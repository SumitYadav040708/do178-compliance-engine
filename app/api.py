"""
DO-178 Compliance Analysis System - Flask API
Provides REST API endpoints for compliance analysis.
Supports: upload, analyze, download reports.
"""

import logging
import os
import json
from typing import Tuple, Dict
from functools import wraps
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, current_app
from typing import Any as any
from main import DO178ComplianceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload/output folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize analyzer
analyzer = DO178ComplianceAnalyzer(config={"output_dir": OUTPUT_FOLDER})


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def error_handler(f):
    """Decorator for API error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500
    
    return decorated_function


# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with service status
    """
    try:
        index_stats = analyzer.retriever.get_index_stats()
        return jsonify({
            "status": "healthy",
            "service": "DO-178 Compliance Analysis",
            "index_loaded": analyzer.retriever.index is not None,
            "index_stats": index_stats
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# Upload Reference Documents
@app.route('/upload-reference', methods=['POST'])
@error_handler
def upload_reference():
    """
    Upload reference DO-178 compliant documents.
    Build FAISS index from uploaded PDFs.
    
    Form data:
        - files: Multiple PDF files
    
    Returns:
        JSON with upload status and index statistics
    """
    if 'files' not in request.files:
        raise ValueError("No files provided")
    
    files = request.files.getlist('files')
    
    if not files or all(f.filename == '' for f in files):
        raise ValueError("No selected files")
    
    uploaded_paths = []
    
    for file in files:
        if not file.filename:
            logger.warning("File has no filename")
            continue
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            continue
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"ref_{filename}")
        file.save(filepath)
        uploaded_paths.append(filepath)
        logger.info(f"Uploaded reference: {filepath}")
    
    if not uploaded_paths:
        raise ValueError("No valid PDF files uploaded")
    
    # Build index
    logger.info(f"Building index from {len(uploaded_paths)} documents")
    success = analyzer.build_reference_index(uploaded_paths)
    
    if not success:
        raise ValueError("Failed to build reference index")
    
    index_stats = analyzer.retriever.get_index_stats()
    
    return jsonify({
        "status": "success",
        "message": f"Index built from {len(uploaded_paths)} documents",
        "files_processed": len(uploaded_paths),
        "index_stats": index_stats
    }), 200


# Analyze Project
@app.route('/analyze', methods=['POST'])
@error_handler
def analyze_project():
    """
    Analyze project PDF against reference index.
    
    Form data:
        - file: Project PDF file
        - similarity_threshold: Threshold (default 0.75)
        - top_k: Top K results (default 5)
    
    Returns:
        JSON with analysis results
    """
    # Check if index is loaded
    if analyzer.retriever.index is None:
        raise ValueError("Reference index not loaded. Upload reference documents first.")
    
    if 'file' not in request.files:
        raise ValueError("No project file provided")
    
    file = request.files['file']
    
    if not file.filename:
        raise ValueError("No selected file")
    
    if not allowed_file(file.filename):
        raise ValueError("Only PDF files allowed")
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"proj_{filename}")
    file.save(filepath)
    logger.info(f"Processing project: {filepath}")
    
    # Get parameters
    similarity_threshold = request.form.get('similarity_threshold', 0.75, type=float)
    top_k = request.form.get('top_k', 5, type=int)
    
    # Analyze
    analysis_result = analyzer.analyze_project(
        filepath,
        similarity_threshold=similarity_threshold,
        top_k=top_k
    )
    
    if not analysis_result:
        raise ValueError("Analysis failed")
    
    # Generate reports
    project_name = filename.replace('.pdf', '')
    reports = analyzer.generate_reports(analysis_result, project_name)
    
    return jsonify({
        "status": "success",
        "message": "Analysis complete",
        "results": {
            "strong_matches": len(analysis_result['strong_matches']),
            "weak_matches": len(analysis_result['weak_matches']),
            "missing_clauses": len(analysis_result['missing_clauses']),
            "project_chunks_total": analysis_result.get('project_chunks_total', 0),
            "reference_chunks_total": analysis_result.get('reference_chunks_total', 0)
        },
        "reports_generated": list(reports.keys()),
        "report_files": reports
    }), 200


# Download Report
@app.route('/download-report/<report_type>/<filename>', methods=['GET'])
@error_handler
def download_report(report_type: str, filename: str):
    """
    Download generated report file.
    
    Args:
        report_type: Type of report (excel, csv, etc.)
        filename: Name of report file
    
    Returns:
        File attachment
    """
    # Security: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValueError("Invalid filename")
    
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    if not os.path.exists(filepath):
        raise ValueError(f"Report not found: {filename}")
    
    try:
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise


# List Available Reports
@app.route('/reports', methods=['GET'])
@error_handler
def list_reports():
    """
    List all available reports.
    
    Returns:
        JSON list of report files
    """
    try:
        if not os.path.exists(OUTPUT_FOLDER):
            return jsonify({"reports": []}), 200
        
        files = os.listdir(OUTPUT_FOLDER)
        files = [f for f in files if os.path.isfile(os.path.join(OUTPUT_FOLDER, f))]
        files.sort(reverse=True)  # Most recent first
        
        return jsonify({
            "status": "success",
            "count": len(files),
            "reports": files
        }), 200
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise


# Index Statistics
@app.route('/index-stats', methods=['GET'])
@error_handler
def index_stats():
    """
    Get statistics about loaded reference index.
    
    Returns:
        JSON with index statistics
    """
    if analyzer.retriever.index is None:
        return jsonify({"status": "no_index"}), 200
    
    stats = analyzer.retriever.get_index_stats()
    
    return jsonify({
        "status": "success",
        "index_stats": stats,
        "embedding_model": analyzer.embedder.get_model_info()
    }), 200


# System Information
@app.route('/system-info', methods=['GET'])
@error_handler
def system_info():
    """
    Get system information and configuration.
    
    Returns:
        JSON with system details
    """
    # Get explainer info safely
    explainer_info = "Fallback"
    if hasattr(analyzer.explainer, 'get_model_info'):
        explainer_info = analyzer.explainer.get_model_info()
    
    return jsonify({
        "system": "DO-178 Compliance Analysis Engine",
        "version": "1.0.0",
        "components": {
            "embedder": analyzer.embedder.get_model_info(),
            "explainer": explainer_info,
            "upload_folder": UPLOAD_FOLDER,
            "output_folder": OUTPUT_FOLDER
        },
        "configuration": {
            "max_file_size": MAX_FILE_SIZE,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }
    }), 200


# BUILD PHASE: ReferencePDF Processing
@app.route('/upload-standard', methods=['POST'])
@error_handler
def upload_standard():
    """
    Upload and process ReferencePDF document (BUILD PHASE).
    Chunks are created and embedded into FAISS index.
    
    DESIGN: ReferencePDF processing does NOT modify keywords.json
    Keywords are user-controlled only and loaded from file at runtime.
    
    Form data:
        - file: ReferencePDF file
    
    Returns:
        JSON with processing status and index statistics
    """
    if 'file' not in request.files:
        raise ValueError("No file provided")
    
    file = request.files['file']
    
    if not file.filename:
        raise ValueError("No selected file")
    
    if not allowed_file(file.filename):
        raise ValueError("Only PDF files allowed")
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"standard_{filename}")
    file.save(filepath)
    logger.info(f"Processing ReferencePDF: {filepath}")
    
    # Process ReferencePDF
    success = analyzer.process_standard_pdf(filepath)
    
    if not success:
        raise ValueError("Failed to process ReferencePDF")
    
    index_stats = analyzer.retriever.get_index_stats()
    
    return jsonify({
        "status": "success",
        "message": "ReferencePDF processed and indexed successfully",
        "file": filename,
        "index_stats": index_stats,
        "note": "keywords.json was NOT modified (user-controlled only)"
    }), 200


# ANALYSIS PHASE: VerifyDocumentCompliance Analysis
@app.route('/check-document', methods=['POST'])
@error_handler
def check_document():
    """
    Check document against previously processed ReferencePDF (ANALYSIS PHASE).
    
    STRICT: VerifyDocumentCompliance analysis does NOT modify:
    - FAISS index
    - ReferencePDF data
    - keywords.json
    
    Form data:
        - file: VerifyDocumentCompliance (document to check)
        - similarity_threshold: Threshold (default 0.75)
        - top_k: Top K results (default 5)
    
    Returns:
        JSON with analysis results
    """
    # Check if standard index is loaded
    if analyzer.retriever.index is None:
        raise ValueError("ReferencePDF not processed. Upload ReferencePDF first using /upload-standard")
    
    if 'file' not in request.files:
        raise ValueError("No file provided")
    
    file = request.files['file']
    
    if not file.filename:
        raise ValueError("No selected file")
    
    if not allowed_file(file.filename):
        raise ValueError("Only PDF files allowed")
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"check_{filename}")
    file.save(filepath)
    logger.info(f"Checking document: {filepath}")
    
    # Get parameters
    similarity_threshold = request.form.get('similarity_threshold', 0.75, type=float)
    top_k = request.form.get('top_k', 5, type=int)
    
    # Check document
    analysis_result = analyzer.check_document(
        filepath,
        similarity_threshold=similarity_threshold,
        top_k=top_k
    )
    
    if not analysis_result:
        raise ValueError("Analysis failed")
    
    # Generate reports
    check_name = filename.replace('.pdf', '')
    reports = analyzer.generate_reports(analysis_result, check_name)
    
    return jsonify({
        "status": "success",
        "message": "Check complete",
        "results": {
            "strong_matches": len(analysis_result['strong_matches']),
            "weak_matches": len(analysis_result['weak_matches']),
            "missing_clauses": len(analysis_result['missing_clauses']),
            "check_chunks_total": analysis_result.get('check_chunks_total', 0),
            "standard_chunks_total": analysis_result.get('standard_chunks_total', 0)
        },
        "reports_generated": list(reports.keys()),
        "report_files": reports
    }), 200


# Keywords Management
@app.route('/keywords', methods=['GET'])
@error_handler
def get_keywords():
    """
    Get current DO-178 keywords from keywords.json (READ-ONLY).
    
    Keywords are user-managed. Edit keywords.json directly to add/modify.
    
    Returns:
        JSON with keywords list
    """
    keywords = analyzer.keyword_manager.get_keywords_list()
    
    return jsonify({
        "status": "success",
        "count": len(keywords),
        "keywords": keywords,
        "note": "Keywords are READ-ONLY. Edit keywords.json to modify keywords."
    }), 200


# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        "error": f"File too large (max {MAX_FILE_SIZE / (1024*1024)}MB)",
        "status": "error"
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500


if __name__ == '__main__':
    logger.info("Starting DO-178 Compliance Analysis API Server")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("\n  BUILD PHASE (ReferencePDF):")
    logger.info("  POST /upload-standard - Process ReferencePDF and build FAISS index")
    logger.info("\n  ANALYSIS PHASE (VerifyDocumentCompliance):")
    logger.info("  POST /check-document - Analyze VerifyDocumentCompliance against ReferencePDF")
    logger.info("\n  KEYWORDS MANAGEMENT (READ-ONLY):")
    logger.info("  GET  /keywords - Get current keywords from keywords.json")
    logger.info("       (Edit keywords.json manually to modify keywords)")
    logger.info("\n  UTILITIES:")
    logger.info("  GET  /download-report/<type>/<filename> - Download report")
    logger.info("  GET  /reports - List all reports")
    logger.info("  GET  /index-stats - Get index statistics")
    logger.info("  GET  /system-info - Get system information")
    logger.info("\n  LEGACY (Deprecated):")
    logger.info("  POST /upload-reference - Upload reference documents")
    logger.info("  POST /analyze - Analyze project")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
