# --- 1. Importar las librerías necesarias ---
import pandas as pd
import numpy as np # Necesario para operaciones con vectores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

# --- 2. Cargar el catálogo de películas desde un archivo CSV ---
try:
    # Usamos encoding='utf-8' para asegurar la compatibilidad con caracteres especiales
    df = pd.read_csv('peliculas.csv', encoding='utf-8')
except FileNotFoundError:
    print("Error: No se encontró el archivo 'peliculas.csv'.")
    print("Asegúrate de que el archivo esté en la misma carpeta que el script.")
    exit()

# Crear una serie para mapear títulos a índices
indices = pd.Series(df.index, index=df['titulo'])

# --- 3. Procesamiento de Texto y Cálculo de Similitud ---
tfidf = TfidfVectorizer(stop_words='english') 
df['sinopsis'] = df['sinopsis'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['sinopsis'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- 4. Función Principal de Recomendación (MODIFICADA) ---
def obtener_recomendaciones(titulo, cosine_sim=cosine_sim, data=df):
    """
    Función que encuentra la película en el catálogo y devuelve las más similares.
    Ahora devuelve los títulos, el índice original y los índices de las recomendadas.
    """
    opciones = data['titulo'].tolist()
    mejor_coincidencia = process.extractOne(titulo, opciones)

    if mejor_coincidencia[1] < 70:
        return "Película no encontrada en nuestro catálogo."
    
    titulo_encontrado = mejor_coincidencia[0]
    idx = indices[titulo_encontrado]
    
    print(f"\nMostrando recomendaciones para: '{titulo_encontrado}'")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]

    # Devolvemos un paquete de datos: los títulos, el índice original y los índices de las películas recomendadas
    return (data['titulo'].iloc[movie_indices], idx, movie_indices)

# --- 5. NUEVA FUNCIÓN PARA GENERAR EL RESUMEN ---
def generar_resumen(idx_original, indices_recomendados, vectorizer, matrix):
    """
    Genera una explicación sobre por qué se recomendaron las películas,
    basándose en las palabras clave (features) con mayor puntaje TF-IDF.
    """
    # Obtener los nombres de todas las palabras clave (features)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Obtener el vector TF-IDF de la película original
    original_vector = matrix[idx_original].toarray().flatten()
    
    # Encontrar los índices de las palabras clave más importantes de la película original
    # (aquellas con el puntaje TF-IDF más alto)
    top_indices = original_vector.argsort()[-7:][::-1] # Tomamos las 7 más importantes
    
    # Obtener los nombres de esas palabras clave
    top_keywords = feature_names[top_indices]
    
    # Filtrar para tener una lista más limpia y relevante
    # (por ejemplo, palabras con más de 3 letras)
    top_keywords_cleaned = [word for word in top_keywords if len(word) > 3]

    # Si tenemos palabras clave, las unimos en un string para el mensaje
    if top_keywords_cleaned:
        keywords_str = ", ".join(top_keywords_cleaned[:5]) # Mostramos hasta 5
        return f"\n-> ¿Por qué estas películas? Probablemente porque comparten temas sobre: {keywords_str}."
    else:
        return "" # No devolvemos nada si no hay palabras clave relevantes

# --- 6. Interacción con el Usuario (MODIFICADA) ---
if __name__ == "__main__":
    while True:
        pelicula_usuario = input("Escribe el nombre de una película que te guste (o 'salir' para terminar): ")

        if pelicula_usuario.lower() == 'salir':
            print("¡Hasta luego! 👋")
            break

        resultado = obtener_recomendaciones(pelicula_usuario)
        
        # Comprobamos si el resultado es un string (error) o una tupla (éxito)
        if isinstance(resultado, str):
            print(f"\n{resultado}\n")
        else:
            # Desempaquetamos los resultados si la búsqueda fue exitosa
            recomendaciones, idx_original, indices_recomendados = resultado
            
            print("--- Películas recomendadas para ti ---")
            for i, pelicula in enumerate(recomendaciones):
                print(f"{i+1}. {pelicula}")
            
            # Generar y mostrar el resumen
            resumen = generar_resumen(idx_original, indices_recomendados, tfidf, tfidf_matrix)
            print(resumen)

        otra_consulta = input("\n¿Deseas buscar recomendaciones para otra película? (s/n): ")
        if otra_consulta.lower() not in ['s', 'si', 'y', 'yes']:
            print("¡Gracias por usar el recomendador! ¡Adiós! 👋")
            break
        else:
            print("-" * 40)