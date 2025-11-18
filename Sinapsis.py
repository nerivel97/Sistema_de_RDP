import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process
import re
import warnings
warnings.filterwarnings('ignore')

# Carga y preprocesamiento de datos de nuestro archivo peliculas.csv
class MovieDataLoader:
    """Clase para cargar y preprocesar datos de películas con soporte bilingüe"""
    
    def __init__(self, filepath='peliculas.csv'):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Carga y preprocesa los datos con soporte bilingüe"""
        try:
            self.df = pd.read_csv(self.filepath, encoding='utf-8')
            print(f" Datos cargados: {len(self.df)} películas encontradas")
        except FileNotFoundError:
            print(" Error: No se encontró el archivo 'peliculas.csv'.")
            print(" Asegúrate de que el archivo esté en la misma carpeta que el script.")
            return False
        except Exception as e:
            print(f" Error inesperado: {e}")
            return False
            
        # Preprocesamiento con soporte bilingüe para una busqueda pues en español e ingles xd
        self._preprocess_data()
        return True
    
    def _preprocess_data(self):
        """Limpia y preprocesa el texto"""
        # Llenado de valores nulos
        self.df['sinopsis'] = self.df['sinopsis'].fillna('')
        
        # Limpieza básica del texto
        self.df['sinopsis_limpia'] = self.df['sinopsis'].apply(self._clean_text)
        
        # Aqui se crea nuestra lista o nuestro indice de todos los títulos para una búsqueda en inglés y español
        self.all_titles = []
        self.title_to_index = {}
        
        for idx, row in self.df.iterrows():
            # Título en inglés
            eng_title = row['titulo'].strip()
            self.all_titles.append(eng_title)
            self.title_to_index[eng_title.lower()] = idx
            
            # Título en español
            if 'titulo_espanol' in row and pd.notna(row['titulo_espanol']):
                esp_title = row['titulo_espanol'].strip()
                self.all_titles.append(esp_title)
                self.title_to_index[esp_title.lower()] = idx
    
    def _clean_text(self, text):
        """Limpia el texto removiendo caracteres especiales y normalizando"""
        if pd.isna(text):
            return ""
        # Funcion para convertir a minúsculas y remover caracteres especiales
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def find_movie_by_title(self, user_input):
        """Busca película por título con soporte bilingüe - CORREGIDO"""
        user_input_lower = user_input.lower().strip()
        
        # PRIMERO: Buscar coincidencia exacta
        if user_input_lower in self.title_to_index:
            idx = self.title_to_index[user_input_lower]
            original_title = self.df.iloc[idx]['titulo']
            return original_title, 100  # 100% de confianza para coincidencia exacta
        
        # SEGUNDO: Buscar con fuzzy matching
        best_match, score = process.extractOne(user_input, self.all_titles)
        
        if score >= 50:  # Umbral de confianza
            # Encontrar el título original en inglés
            best_match_lower = best_match.lower()
            if best_match_lower in self.title_to_index:
                idx = self.title_to_index[best_match_lower]
                original_title = self.df.iloc[idx]['titulo']
                return original_title, score
        
        return None, score

# Sistema de Recomendación (Donde ocurre la magia) / Funciones principales para el sistema
class MovieRecommender:
    """Sistema de recomendación preciso y confiable"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        
    def build_model(self):
        """Construye el modelo de recomendación optimizado"""
        print(" Construyendo modelo de recomendación...")
        
        # Vectorización TF-IDF la cual esta optimizada para velocidad y una busqueda eficiente
        self.tfidf = TfidfVectorizer(
            stop_words=['spanish', 'english'],
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.85
        )
        
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.data_loader.df['sinopsis_limpia']
        )
        
        # Calcular similitud del coseno
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        print(" Modelo de recomendación listo!")
    
    def get_recommendations(self, movie_title, n_recommendations=6):
        """Obtiene recomendaciones con soporte bilingüe"""
        # Esta es una seccion nueva en la que su funcion es buscar la película con soporte bilingüe
        matched_movie, confidence = self.data_loader.find_movie_by_title(movie_title)
        
        if not matched_movie:
            if confidence < 40:
                return {
                    'error': f"❌ No encontré '{movie_title}'. ¿Está bien escrito?",
                    'confianza': confidence
                }
            else:
                return {
                    'error': f"❌ Coincidencia baja ({confidence}%). Intenta con otro título.",
                    'confianza': confidence
                }
        
        # Obtener índice de la película
        matched_movie_lower = matched_movie.lower()
        if matched_movie_lower not in self.data_loader.title_to_index:
            return {
                'error': "❌ Error interno: Película no encontrada en índice.",
                'confianza': confidence
            }
        
        idx = self.data_loader.title_to_index[matched_movie_lower]
        
        # Calcular similitudes
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filtrar y obtener recomendaciones
        recommendations = []
        for movie_idx, score in sim_scores[1:]:  # Saltar la propia película para que no se muestra como recomendacion, seria ilogico xd
            if len(recommendations) >= n_recommendations:
                break
            
            movie_title = self.data_loader.df.iloc[movie_idx]['titulo']
            similarity_percent = round(score * 100, 1)
            
            # Aqui solo se incluyen recomendaciones con similitud razonable
            if similarity_percent > 1:  # Umbral mínimo de similitud
                recommendations.append({
                    'titulo': movie_title,
                    'similitud': similarity_percent,
                    'indice': movie_idx
                })
        
        return {
            'pelicula_original': matched_movie,
            'confianza_coincidencia': confidence,
            'recomendaciones': recommendations,
            'indice_original': idx,
            'exito': True
        }
    
    def generate_explanation(self, original_idx, top_keywords=6):
        """Genera explicación basada en palabras clave"""
        feature_names = np.array(self.tfidf.get_feature_names_out())
        original_vector = self.tfidf_matrix[original_idx].toarray().flatten()
        
        # Encontrar palabras clave más importantes
        top_indices = original_vector.argsort()[-10:][::-1]
        top_keywords_list = feature_names[top_indices]
        
        # Filtrar palabras relevantes
        relevant_keywords = [
            word for word in top_keywords_list 
            if len(word) > 2 and not word.isdigit() and word not in ['película', 'historia', 'hombre', 'mujer', 'debe', 'puede', 'siendo']
        ][:top_keywords]
        
        if relevant_keywords:
            keywords_str = ", ".join(relevant_keywords)
            return f" **Temas en común**: {keywords_str}"
        return ""

# Interfaz de Usuario
class MovieRecommendationApp:
    """Interfaz de usuario con información detallada"""
    
    def __init__(self):
        self.data_loader = MovieDataLoader()
        self.recommender = None
        
    def initialize(self):
        """Inicializa la aplicación rápidamente"""
        print("" + "="*60)
        print("     S I N A P S I S  v2 (Ahora con soporte bilingue jeje)")
        print("     Sistema de Recomendación de Películas")
        print("="*60 + "")
        
        if not self.data_loader.load_data():
            return False
            
        self.recommender = MovieRecommender(self.data_loader)
        self.recommender.build_model()
        
        print("\n Sistema listo! Puedes buscar en español o inglés.")
        return True
    
    def show_quick_help(self):
        """Muestra ayuda rápida"""
        print("\n **Puedes buscar en español o inglés**:")
        print("   • 'The Matrix' o 'Matrix'")
        print("   • 'The Mask' o 'La Máscara'") 
        print("   • 'El Padrino' o 'The Godfather'")
        print("   • 'Sueños de Fuga' o 'The Shawshank Redemption'")
        print("\n Se mostrará: Precisión del título + % de similitud")
    
    def run(self):
        """Ejecuta la aplicación principal"""
        if not self.initialize():
            return
        
        self.show_quick_help()
        
        while True:
            print("\n" + "─" * 60)
            pelicula_usuario = input(
                "\n **Escribe una película** (o 'salir' para terminar): "
            ).strip()
            
            if pelicula_usuario.lower() == 'salir':
                break
            elif not pelicula_usuario:
                print(" Por favor, escribe el nombre de una película.")
                continue
            
            # Obtener recomendaciones
            resultado = self.recommender.get_recommendations(pelicula_usuario)
            self._display_results(resultado, pelicula_usuario)
            
            # Preguntar por otra búsqueda
            if not self._ask_another_search():
                break
        
        self._show_goodbye()
    
    def _display_results(self, resultado, user_input):
        """Muestra los resultados con toda la información - MEJORADO"""
        if 'error' in resultado:
            print(f"\n{resultado['error']}")
            if 'confianza' in resultado:
                print(f"   (Precisión de búsqueda: {resultado['confianza']}%)")
            return
        
        if not resultado.get('exito', False):
            print("❌ No se pudieron generar recomendaciones.")
            return
        
        # Mostrar información de la película encontrada
        confianza = resultado['confianza_coincidencia']
        precision_color = "🟢" if confianza >= 80 else "🟡" if confianza >= 60 else "🟠"
        
        print(f"\n🎬 **Película encontrada**: '{resultado['pelicula_original']}'")
        print(f"   {precision_color} **Precisión del título**: {confianza}%")
        print(f"    **Búsqueda original**: '{user_input}'")
        
        # Se muestran las recomendaciones con porcentajes de similitud
        if resultado['recomendaciones']:
            print(f"\n **Recomendaciones similares**:")
            for i, rec in enumerate(resultado['recomendaciones'], 1):
                sim_color = "🟢" if rec['similitud'] > 30 else "🟡" if rec['similitud'] > 15 else "🟠"
                print(f"   {i}. {rec['titulo']} {sim_color} {rec['similitud']}%")
            
            # Aqui se genera la explicacion o el resultado acerca de los temas en comun
            explanation = self.recommender.generate_explanation(
                resultado['indice_original']
            )
            if explanation:
                print(f"\n{explanation}")
        else:
            print("\n No se encontraron recomendaciones suficientemente similares.")
    
    def _ask_another_search(self):
        """Pregunta si hacer otra búsqueda"""
        respuesta = input("\n ¿Buscar otra película? (s/n): ").strip().lower()
        return respuesta in ['s', 'si', 'y', 'yes']
    
    def _show_goodbye(self):
        """Mensaje de despedida"""
        print("         ¡Gracias por usar Sinapsis!")
        print("         Hasta la próxima!")

# Ejecución principal
if __name__ == "__main__":
    try:
        app = MovieRecommendationApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n Programa terminado.")
    except Exception as e:
        print(f"\n Error inesperado: {e}")
        import traceback
        traceback.print_exc()