Sinapsis: Sistema de Recomendación de Películas

Un script de Python que recomienda películas basándose en la similitud de sus sinopsis. Utiliza técnicas de NLP para analizar texto y encontrar las películas más relevantes.

Cómo Usarlo

1. Requisitos

Asegúrate de tener Python y las siguientes librerías instaladas:
Bash

pip install pandas scikit-learn numpy thefuzz python-Levenshtein

2. Estructura de Archivos

Para que el script funcione, tu carpeta de proyecto debe tener la siguiente estructura:

    Sinapsis.py (El script principal de Python)

    peliculas.csv (El archivo con los títulos y sinopsis)

3. Ejecución

En tu terminal, navega a la carpeta del proyecto y ejecuta el siguiente comando:
Bash

python Sinapsis.py

El programa te pedirá que escribas el nombre de una película para comenzar.

Cómo Funciona

    Vectorización TF-IDF: Convierte el texto de las sinopsis en vectores numéricos, dando más importancia a las palabras clave que son significativas para definir una película.

    Similitud del Coseno: Calcula la similitud matemática entre estos vectores para encontrar las películas con las temáticas más parecidas.

    TheFuzz: Identifica el título de la película en la base de datos incluso si el usuario lo escribe con errores ortográficos.