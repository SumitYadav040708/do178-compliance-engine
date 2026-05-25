from app.llm_explainer import OllamaExplainer, LocalExplainerFallback

explainer = OllamaExplainer()
print('connection:', explainer.check_connection())
print('model available:', explainer.check_model_available())
print('model info:', explainer.get_model_info())

keyword = "Traceability"
chunk = "The system logs all runtime activities for traceability."
standard_chunk = "All safety-critical systems must maintain execution logs for traceability."
similarity = 0.92

result = explainer.generate_explanation(
    keyword=keyword,
    similarity_score=similarity,
    chunk=chunk,
    standard_chunk=standard_chunk
)
print('result:', result)