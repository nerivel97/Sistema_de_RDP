# Configuración del sistema
class Config:
    # Parámetros TF-IDF
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.8
    
    # Recomendaciones
    DEFAULT_RECOMMENDATIONS = 5
    SIMILARITY_THRESHOLD = 60
    
    # Interfaz
    SHOW_SIMILARITY_SCORES = True
    ENABLE_ANALYTICS = True