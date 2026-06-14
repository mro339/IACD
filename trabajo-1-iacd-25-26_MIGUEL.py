# Inteligencia Artificial para la Ciencia de los Datos
# Implementación de clasificadores
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: RODRIGUEZ ORTEGA
# NOMBRE: MIGUEL ÁNGEL
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: MANZANO HERNÁNDEZ
# NOMBRE: VIOLETA
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA: PLAGIO O CÓDIGO GENERADO AUTOMÁTICAMENTE.
# Un trabajo práctico es un examen, por lo que debe realizarse de manera
# individual o con la pareja del grupo.
# La discusión y el intercambio de información de carácter general con los
# compañeros se permite, pero NO AL NIVEL DE CÓDIGO.

# El objetivo principal del entregable es trabajar de manera práctica
# los conceptos aprendidos en clase, para alcanzar una mayor comprensión
# de los mismos a través de la implementación que se pide.
# Se permite, si así se desea, el uso de herramientas de inteligencia
# artificial generativa que ayuden en el desarrollo código,
# pero esta herramienta ha de usarse sólo como un asistente que facilite
# el trabajo, y en ningún caso se debe entregar un código que no se conozca
# en profundidad, y sobre el que no se sepa responder durante la presentación
# al profesor a cualquier pregunta con el detalle requerido. Si el trabajo se
# realiza en pareja, cualquiera de los dos miembros del grupo debe de poder
# responder con detalle de código en cualquiera de los apartados del trabajo.

# Cualquier plagio o entrega de código cuyo funcionamiento no se sea capaz de
# explicar con detalle, supondrá una calificación de cero, sin perjuicio
# de las medidas disciplinarias que se pudieran tomar.
# *****************************************************************************




# MUY IMPORTANTE:
# ===============

# * NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
#   Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# * En este trabajo NO SE PERMITE USAR Scikit Learn.

# * Se recomienda (y se valora especialmente) el uso eficiente de numpy. Todos
#   los datasets se suponen dados como arrays de numpy.

# * Se ha de entregar este archivo con las implementaciones realizadas,
#   junto con otro archivo trabajo-1-iacd-25-26-sc.py con la misma implementación,
#   pero en el que no se incluyan comentarios al código, y que será el que se
#   use durante la presentación del trabajo.

#   AL FINAL DE ESTE ARCHIVO hay una serie de ejemplos a ejecutar que están comentados, y que
#   será lo que se ejecute durante la presentación del trabajo al profesor.
#   En la versión final a entregar, descomentar esos ejemplos del final y no dejar
#   ninguna otra ejecución de ejemplos.



import math
import random
import numpy as np
from scipy.special import expit  # Para la función sigmoide estable numéricamente
from itertools import product    # Para generar todas las combinaciones de hiperparámetros


# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-1-iacd.tgz (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN.

from carga_datos import *

# Como consecuencia de la línea anterior, se habrán cargado los siguientes
# conjuntos de datos, que pasamos a describir, junto con los nombres de las
# variables donde se cargan. Todos son arrays de numpy:


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.


# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original.
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.

# Además, en la carpeta datos/digitdata se tiene el siguiente dataset, que
# habrá de ser procesado y cargado:

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos.
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles).



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA
# ==================================================

# Definir una función

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.
#

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición.

# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([136, 229]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([34, 57]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


def particion_entr_prueba(X, y, test=0.20):
    # Obtenemos las clases únicas (por ejemplo [0, 1] en votos, o ['conceder','estudiar','no conceder'] en crédito)
    clasesUnicas = np.unique(y)

    # Usamos dos listas para acumular los índices de entrenamiento y prueba
    # Se usan listas de Python (no arrays) porque iremos añadiendo elementos con extend
    indicesEntrenamiento = []
    indicesPrueba = []

    # Procesamos CLASE POR CLASE para garantizar la estratificación:
    # si una clase tiene el 60% del total, queremos que también tenga el 60% en entrenamiento y en prueba
    for clase in clasesUnicas:
        # np.where(y == clase) devuelve una TUPLA de arrays, uno por dimensión de y.
        # Como y es 1D, la tupla tiene un solo elemento: (array([idx1, idx2, ...]),)
        # El [0] extrae ese array de la tupla para trabajar directamente con los índices
        indicesDeEstaClase = np.where(y == clase)[0]

        # Barajamos SOLO los índices de esta clase (no todo el dataset)
        # Así la aleatoriedad es independiente para cada clase
        np.random.shuffle(indicesDeEstaClase)

        # Cuántos ejemplos de esta clase van a prueba
        numEjemplosPrueba = int(len(indicesDeEstaClase) * test)

        # Los primeros numEjemplosPrueba van a prueba, el resto a entrenamiento.
        # Usamos extend (no append) porque queremos añadir los índices sueltos a la lista,
        # no añadir la sublista entera como un solo elemento
        indicesPrueba.extend(indicesDeEstaClase[:numEjemplosPrueba])
        indicesEntrenamiento.extend(indicesDeEstaClase[numEjemplosPrueba:])

    # Ordenamos para preservar el orden relativo original del dataset
    indicesEntrenamiento = np.array(sorted(indicesEntrenamiento))
    indicesPrueba = np.array(sorted(indicesPrueba))

    X_entrenamiento = X[indicesEntrenamiento]
    X_prueba = X[indicesPrueba]
    y_entrenamiento = y[indicesEntrenamiento]
    y_prueba = y[indicesPrueba]

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba


# =========================================================
# EJERCICIO 2: IMPLEMENTACIÓN DE CLASIFICADORES NAIVE BAYES
# =========================================================

# Se pide implementar clasificadores Naive Bayes, tanto en su versión categórica
# como en su versión gaussiana, con suavizado y log probabilidades
# (descritos en el tema 2, diapositivas 22 a 34 y diapositivas 48 a 50).
# En concreto:


# ---------------------------------------------
# 2.1) Implementación de Naive Bayes categórico
# ---------------------------------------------

# Definir una clase NaiveBayesCat con la siguiente estructura:

# class NaiveBayesCat():

#     def __init__(self,k=1):
#
#          .....

#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplos):

#         ......


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1)
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan. NOTA: Se valorará
#   que el entrenamiento se haga con un único recorrido del dataset.
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase.
# * Método clasifica: recibe un array de ejemplos (en forma de array de numpy) y
#   devuelve un array con las clases que el modelo predice para esos ejemplos.


# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass



# Ejemplo "jugar al tenis":


# >>> nb_tenis=NaiveBayesCat(k=0.5)
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(np.array([ej_tenis]))
# ['no']

class NaiveBayesCat():

    def __init__(self, k=1):
        # k: constante de suavizado de Laplace para evitar probabilidades cero
        self.k = k
        # Indica si el modelo ha sido entrenado
        self.entrenado = False

    def entrena(self, X, y):
        '''
        En esta clase estamos creando diccionares/conteos
        1º Probabilidad a priori, sería frecuencia, cuantas veces está la clase entre número total de instancias.
        2º Posibles valores de cada Atributo/Característica, (sin que se repita claro de ahí np.unique)
        3º Inicializar el Contador del Diccionario valores de atributos
        4º Inciializar el contador del dicionario clases
        5º Recorremos todo el ejemplo y hacemos el diccionario.
        '''


        # X: matriz de datos
        # y: vector con la clase de cada ejemplo
        numEjemplos, numCaracteristicas = X.shape

        # Guardamos self.clases como atributo porque lo necesitaremos en clasifica_prob y clasifica.
        # contadorClases solo se usa aquí para calcular la proporción, así que no hace falta guardarlo.
        self.clases, contadorClases = np.unique(y, return_counts=True)
        self.numeroClases = len(self.clases)

        # Calculamos P(clase) = fracción de ejemplos de cada clase en el dataset.
        self.probabilidadPriori = {}
        for clase, conteo in zip(self.clases, contadorClases): # zip une los dos arrays en paralelo
            self.probabilidadPriori[clase] = conteo / numEjemplos

        # Para cada atributo guardamos qué valores distintos puede tomar en el entrenamiento.
        self.valoresPosiblesAtributo = []
        for indiceAtributo in range(numCaracteristicas):
            self.valoresPosiblesAtributo.append(np.unique(X[:, indiceAtributo])) #Extrae toda la columna

        # Inicializamos la estructura de conteos de tres niveles:
        #   conteosAtributos[indiceAtributo][valor][clase] = cuántas veces aparece esa combinación
        self.conteosAtributos = []
        for indiceAtributo in range(numCaracteristicas):
            diccionarioConteos = {}
            for valor in self.valoresPosiblesAtributo[indiceAtributo]:
                # Para cada valor posible del atributo, creamos un contador a 0 para cada clase.
                # {clase: 0 for clase in self.clases} es una comprensión de diccionario:
                # genera {'no': 0, 'si': 0} de forma compacta
                diccionarioConteos[valor] = {clase: 0 for clase in self.clases}
            self.conteosAtributos.append(diccionarioConteos)

        # Conteo de cuántos ejemplos hay en total por clase.
        self.conteosPorClase = {clase: 0 for clase in self.clases}

        # Recorremos el dataset UNA SOLA VEZ para contar todo lo necesario.
        # Por cada ejemplo sumamos 1 a su clase y 1 a la combinación (atributo, valor, clase)
        for indiceEjemplo in range(numEjemplos):
            clase = y[indiceEjemplo]
            self.conteosPorClase[clase] += 1
            for indiceAtributo in range(numCaracteristicas):
                valor = X[indiceEjemplo, indiceAtributo]
                # Accedemos a los tres niveles: atributo → valor → clase
                self.conteosAtributos[indiceAtributo][valor][clase] += 1

        self.entrenado = True

    def clasifica_prob(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")

        # log P(clase|ejemplo) proporcional a log P(clase) + suma de log P(atributo_j=valor_j | clase)
        logProbabilidades = {}

        for clase in self.clases:
            # Empezamos con log P(clase)
            logProbabilidadClase = math.log(self.probabilidadPriori[clase])

            for indiceAtributo, valor in enumerate(ejemplo):
                numValoresPosibles = len(self.valoresPosiblesAtributo[indiceAtributo])

                # Conteo de este valor para esta clase (0 si no se vio en entrenamiento)
                conteoActual = self.conteosAtributos[indiceAtributo].get(valor, {}).get(clase, 0)

                # Suavizado de Laplace: (conteo + k) / (total_clase + k * num_valores_posibles)
                probabilidadAtributo = (conteoActual + self.k) / (
                    self.conteosPorClase[clase] + self.k * numValoresPosibles
                )

                logProbabilidadClase += math.log(probabilidadAtributo)

            logProbabilidades[clase] = logProbabilidadClase

        # Convertimos de log a probabilidades normalizadas (deben sumar 1)
        # Restamos el máximo antes de exponenciar para mayor estabilidad numérica
        maxLog = max(logProbabilidades.values())

        probabilidadesExponenciadas = {}
        for clase in self.clases:
            probabilidadesExponenciadas[clase] = math.exp(logProbabilidades[clase] - maxLog)

        sumaTotal = sum(probabilidadesExponenciadas.values())

        probabilidades = {}
        for clase in self.clases:
            probabilidades[clase] = probabilidadesExponenciadas[clase] / sumaTotal

        return probabilidades

    def clasifica(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")

        predicciones = []
        for ejemplo in ejemplos:
            probabilidades = self.clasifica_prob(ejemplo)
            # Elegimos la clase con mayor probabilidad a posteriori.
            # max con key=probabilidades.get busca la clave (clase) cuyo valor es el mayor.
            clasePredicha = max(probabilidades, key=probabilidades.get)
            predicciones.append(clasePredicha)

        # Devolvemos un array numpy con una predicción por cada ejemplo recibido
        return np.array(predicciones)


# ----------------------------------------------
# 2.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y.

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

def rendimiento(clasificador, X, y):
    y_predicho = clasificador.clasifica(X)
   

    # numpy trata True como 1 y False como 0, por lo que np.sum suma los aciertos directamente.
    numAciertos = np.sum(y == y_predicho)
    return numAciertos / len(y)


# -------------------------------------
# 2.3) Aplicando Naive Bayes categórico
# -------------------------------------

# Usando el clasificador Naive Bayes categórico implementado,
# obtener clasificadores con el mejor rendimiento posible para los
# siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB

# En todos los casos, será necesario separar un conjunto de test para dar la
# valoración final de los clasificadores obtenidos (ya realizado en el ejerciio
# anterior). Ajustar también el valor del parámetro de suavizado k, usando un
# conjunto de validación.

# Describir (dejándolo comentado) el proceso realizado en cada caso,
# y los rendimientos obtenidos.

# PROCESO SEGUIDO:
# 1. Dividimos en entrenamiento+validación (80%) y test (20%).
# 2. Del 80% anterior, dividimos en entrenamiento (75%) y validación (25%),
#    lo que nos da aproximadamente 60% entrenamiento, 20% validación, 20% test.
# 3. Probamos distintos valores de k evaluando el rendimiento sobre VALIDACIÓN
#    (no sobre entrenamiento, porque así evitamos elegir k=0, que sobreajusta).
# 4. Elegimos el k con mejor rendimiento en validación.
# 5. Reentrenamos el modelo final con entrenamiento+validación y evaluamos en test.

# valoresK = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]


# ---- VOTOS ----
# División en tres partes: entrenamiento, validación y test
# X_votos_tv, X_votos_test, y_votos_tv, y_votos_test = particion_entr_prueba(X_votos, y_votos, test=0.2)
# X_votos_train, X_votos_val, y_votos_train, y_votos_val = particion_entr_prueba(X_votos_tv, y_votos_tv, test=0.25)
#
# mejor_k_votos = None
# mejor_rendimiento_votos_val = 0
#
# for k in valoresK:
#     nb = NaiveBayesCat(k)
#     nb.entrena(X_votos_train, y_votos_train)
#     # Evaluamos en VALIDACIÓN para elegir k sin contaminar el test
#     rendimientoValidacion = rendimiento(nb, X_votos_val, y_votos_val)
#     if rendimientoValidacion > mejor_rendimiento_votos_val:
#         mejor_rendimiento_votos_val = rendimientoValidacion
#         mejor_k_votos = k
#
# # Reentrenamos con train+val usando el mejor k y evaluamos en test
# nb_final_votos = NaiveBayesCat(mejor_k_votos)
# nb_final_votos.entrena(X_votos_tv, y_votos_tv)
# rendimiento_test_votos = rendimiento(nb_final_votos, X_votos_test, y_votos_test)


# ---- CRÉDITO ----
# X_credito_tv, X_credito_test, y_credito_tv, y_credito_test = particion_entr_prueba(X_credito, y_credito, test=0.2)
# X_credito_train, X_credito_val, y_credito_train, y_credito_val = particion_entr_prueba(X_credito_tv, y_credito_tv, test=0.25)
#
# mejor_k_credito = None
# mejor_rendimiento_credito_val = 0
#
# for k in valoresK:
#     nb = NaiveBayesCat(k)
#     nb.entrena(X_credito_train, y_credito_train)
#     rendimientoValidacion = rendimiento(nb, X_credito_val, y_credito_val)
#     if rendimientoValidacion > mejor_rendimiento_credito_val:
#         mejor_rendimiento_credito_val = rendimientoValidacion
#         mejor_k_credito = k
#
# nb_final_credito = NaiveBayesCat(mejor_k_credito)
# nb_final_credito.entrena(X_credito_tv, y_credito_tv)
# rendimiento_test_credito = rendimiento(nb_final_credito, X_credito_test, y_credito_test)


# ---- IMDB ----
# El dataset IMDB ya viene dividido en train y test; creamos validación desde train
# X_imdb_train, X_imdb_val, y_imdb_train, y_imdb_val = particion_entr_prueba(X_train_imdb, y_train_imdb, test=0.2)
#
# mejor_k_imdb = None
# mejor_rendimiento_imdb_val = 0
#
# for k in valoresK:
#     nb = NaiveBayesCat(k)
#     nb.entrena(X_imdb_train, y_imdb_train)
#     rendimientoValidacion = rendimiento(nb, X_imdb_val, y_imdb_val)
#     if rendimientoValidacion > mejor_rendimiento_imdb_val:
#         mejor_rendimiento_imdb_val = rendimientoValidacion
#         mejor_k_imdb = k
#
# # Reentrenamos con todo el train original (train+val) usando el mejor k
# nb_final_imdb = NaiveBayesCat(mejor_k_imdb)
# nb_final_imdb.entrena(X_train_imdb, y_train_imdb)
# rendimiento_test_imdb = rendimiento(nb_final_imdb, X_test_imdb, y_test_imdb)


# --------------------------------------------
# 2.4) Implementación de Naive Bayes gaussiano
# --------------------------------------------

# Definir una clase NaiveBayesGauss con la misma estructura que la descrita en
# el apartado 2.1 (pero sin considerar constante de suavizado).

class NaiveBayesGauss():

    def __init__(self):
        self.entrenado = False

    def entrena(self, X, y):
        # X: matriz de datos con características numéricas continuas
        # y: vector de etiquetas de clase
        numEjemplos, numCaracteristicas = X.shape
        self.clases = np.unique(y)

        # Para cada clase almacenamos su probabilidad priori, media y desviación típica
        self.probabilidadPriori = {}
        self.mediasClase = {}      
        self.desviacionesClase = {}   

        for clase in self.clases:
            # La máscara, un array de True/False.
            mascaraDeEstaClase = (y == clase)

            # Usamos la máscara para quedarnos solo con las filas de X de esta clase
            X_deEstaClase = X[mascaraDeEstaClase]

            # np.sum sobre booleanos cuenta los True, es decir, cuántos ejemplos son de esta clase
            self.probabilidadPriori[clase] = np.sum(mascaraDeEstaClase) / numEjemplos

            # Calculamos la media y la desviación de cada columna, característica
            self.mediasClase[clase] = np.mean(X_deEstaClase, axis=0) #axis=0 se hace por columnas (características) no filas
            self.desviacionesClase[clase] = np.std(X_deEstaClase, axis=0)

        self.entrenado = True

    def clasifica_prob(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")

        logProbabilidades = {}

        for clase in self.clases:
            # Empezamos con log P(clase)
            logProbabilidadClase = math.log(self.probabilidadPriori[clase])

            medias = self.mediasClase[clase]
            desviaciones = self.desviacionesClase[clase].copy()
            # Evitamos división por cero en características constantes (desviación = 0)
            desviaciones[desviaciones == 0] = 1e-9

            # Log de la densidad gaussiana para cada característica, calculado con numpy
            # log N(x; mu, sigma) = -log(sigma) - 0.5*log(2*pi) - 0.5*((x - mu) / sigma)^2
            logProbabilidadClase += np.sum(
                -np.log(desviaciones)
                - 0.5 * np.log(2 * math.pi)
                - 0.5 * ((ejemplo - medias) / desviaciones) ** 2
            )

            logProbabilidades[clase] = logProbabilidadClase

        # Normalizamos para que las probabilidades sumen 1. Se hizo anteriormente
        maxLog = max(logProbabilidades.values())
        probabilidadesExponenciadas = {
            clase: math.exp(logProbabilidades[clase] - maxLog) for clase in self.clases
        }
        sumaTotal = sum(probabilidadesExponenciadas.values())

        return {clase: probabilidadesExponenciadas[clase] / sumaTotal for clase in self.clases}

    def clasifica(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")

        predicciones = []
        for ejemplo in ejemplos:
            probabilidades = self.clasifica_prob(ejemplo)
            predicciones.append(max(probabilidades, key=probabilidades.get))

        return np.array(predicciones)


# ------------------------------------
# 2.5) Aplicando Naive Bayes gaussiano
# ------------------------------------

# Aplicar el clasificador NaiveBayesGauss a los datos del cáncer de mama.

# Dividimos el dataset de cáncer en entrenamiento, validación y test
# Xev_cancer_nb, Xp_cancer_nb, yev_cancer_nb, yp_cancer_nb = particion_entr_prueba(X_cancer, y_cancer, test=0.2)
# Xe_cancer_nb, Xv_cancer_nb, ye_cancer_nb, yv_cancer_nb = particion_entr_prueba(Xev_cancer_nb, yev_cancer_nb, test=0.2)
#
# nb_gauss_cancer = NaiveBayesGauss()
# nb_gauss_cancer.entrena(Xe_cancer_nb, ye_cancer_nb)
#
# rendimiento_gauss_cancer_entrenamiento = rendimiento(nb_gauss_cancer, Xe_cancer_nb, ye_cancer_nb)
# rendimiento_gauss_cancer_test = rendimiento(nb_gauss_cancer, Xp_cancer_nb, yp_cancer_nb)


# ==================================
# EJERCICIO 3: NORMALIZADOR ESTÁNDAR
# ================================== 

# Definir la siguiente clase que implemente la normalización "standard", es
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1.

# En particular, definir la clase:


# class NormalizadorStandard():

#    def __init__(self):

#         .....

#     def ajusta(self,X):

#         .....

#     def normaliza(self,X):

#         ......

#


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método
# normaliza devuelve el correspondiente conjunto de datos normalizados.

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

# Por ejemplo:


# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n,
# ni con Xp_cancer_n.


class NormalizadorNoAjustado(Exception):
    pass

class NormalizadorStandard():

    def __init__(self):
        # Parámetros de la transformación. 
        # Se inicializan a None para poder detectar más tarde si el normalizador ha sido ajustado o no.
        self.medias = None          # (n_características,)
        self.desviaciones = None    # (n_características,)

    def ajusta(self, X):
        """Calcula y almacena los parámetros de normalización a partir de X. 
        X es un array de numpy de forma (n_ejemplos, n_características)"""

        # Media de cada característica  -> μ_x
        self.medias = np.mean(X, axis=0) # axis=0 hace que el cálculo sea por columnas, de forma que la implementación es general para cualquier nº de columnas.

        # Desviación típica de cada característica -> σ_x
        self.desviaciones = np.std(X, axis=0, ddof=0) # ddof=0: método de StandardScaler, divide entre N
 
        # Evitamos dividir por 0 para las características constantes
        self.desviaciones[self.desviaciones == 0] = 1.0

    def normaliza(self, X):
        """Devuelve X normalizado usando los parámetros calculados en ajusta()"""

        # Si todavía no se ha ajustado
        if self.medias is None:
            raise NormalizadorNoAjustado(
                "El normalizador debe ajustarse (ajusta) antes de normalizar.")

        # x_norm = (x - μ_x) / σ_x
        return (X - self.medias) / self.desviaciones
    
# ==============================================================
# EJERCICIO 4: REGRESIÓN LOGÍSTICA MINI-BATCH CON REGULARIZACIÓN
# ==============================================================


# En este ejercicio se propone la implementación de un clasificador lineal
# binario basado en regresión logística (mini-batch), con algoritmo de entrenamiento
# de descenso por el gradiente mini-batch (diapositiva 50 del tema 3).
# Se pide también incluir regularización L2 (es decir, la función de
# pérdida a minimizar es la entropía cruzada más un sumando de regularización
# cuadrática)


# En concreto se pide implementar una clase:

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64,reg=0.01):

#         .....

#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....

#     def clasifica_prob(self,ejemplos):

#         ......

#     def clasifica(self,ejemplos):

#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula:
#        rate_n= (rate_0)*(1/(1+n))
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False.
#
#   + batch_tam: tamaño de minibatch
#
#   + reg: constante de regularización L2

# * El método entrena tiene como argumentos de entrada:
#
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento
#        y su clasificación esperada, respectivamente. Las dos clases del problema
#        son las que aparecen en el array y, y se deben almacenar en un atributo
#        self.clases en una lista. La clase que se considera positiva es la que
#        aparece en segundo lugar en esa lista.
#
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se
#       usarán en el caso de activar el parámetro early_stopping. Ambos con
#       valor None por defecto.

#     + n_epochs es el número máximo de epochs en el entrenamiento.

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada
#       del modelo respecto del conjunto de entrenamiento, más la penalización L2,
#       y su rendimiento (proporción de aciertos). Igualmente para el conjunto
#       de validación, si lo hubiera. Esta opción puede ser útil para comprobar
#       si el entrenamiento  efectivamente está haciendo descender la función
#       de pérdida del modelo (recordemos que el objetivo del entrenamiento es
#       encontrar los pesos que minimizan la función de pérdida), y está haciendo
#       subir el rendimiento.
#
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor pérdida conseguida hasta el momento
#       en el conjunto de validación
#       NOTA: esto se suele hacer con un conjunto de validación, y mecanismo de
#       "callback" para recuperar el mejor modelo, pero por simplificar implementaremos
#       esta versión más sencilla.
#


# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos.

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo
#   asigna a cada ejemplo de pertenecer a la clase positiva.


# RECOMENDACIONES:


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande.

# + Téngase en cuenta que el cálculo de la función de pérdida no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.

# * Definir la función sigmoide usando la función expit de scipy.special,
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada.

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])


# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:

# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,reg=0.001)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento LOSS: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    LOSS: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento LOSS: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    LOSS: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento LOSS: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    LOSS: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento LOSS: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    LOSS: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento LOSS: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    LOSS: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento LOSS: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    LOSS: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento LOSS: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    LOSS: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la pérdida obtenida en el epoch 3
# sobre el conjunto de validación, ésta no se ha mejorado.


# -----------------------------------------------------------------


def sigmoide(x):
    # Usamos expit de scipy para calcular 1 / (1 + e^-x) sin riesgo de overflow
    return expit(x)


class RegresionLogisticaMiniBatch():

    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100, batch_tam=64, reg=0.01):
        # Tasa de aprendizaje (fija o inicial si hay decaimiento)
        self.tasaAprendizajeInicial = rate
        # Si True, la tasa disminuye en cada epoch: rate_n = rate_0 * (1 / (1 + n))
        self.decaimientoTasa = rate_decay
        # Número máximo de epochs de entrenamiento
        self.numEpochsMaximo = n_epochs
        # Número de ejemplos por mini-batch
        self.tamanoMiniBatch = batch_tam
        # Constante de regularización L2
        self.constanteRegularizacion = reg
        self.entrenado = False

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=100, salida_epoch=False,
                early_stopping=False, paciencia=3):
        # Xv e yv son el conjunto de VALIDACIÓN (opcionales).
        # Se usan únicamente cuando early_stopping=True para monitorizar si el modelo
        # empieza a empeorar en datos no vistos y hay que parar el entrenamiento antes de time.

        numEjemplos, numCaracteristicas = X.shape

        # Recordemos: Índice 0 es la clase negativa, índice 1 es la clase positiva
        self.clases = list(np.unique(y))
        clasePositiva = self.clases[1]

        # Convertimos las etiquetas a 0/1 para poder operar con ellas numéricamente
        y_binario = (y == clasePositiva).astype(float) #float para operar fácilmente
        if Xv is not None:
            yv_binario = (yv == clasePositiva).astype(float)

        # Inicializamos pesos y sesgo (bias) a cero
        self.pesos = np.zeros(numCaracteristicas)
        self.sesgo = 0.0

  
        mejorPerdidaValidacion = float('inf') #Infinito para actualizar nada más la primera época
        epochsSinMejora = 0

        # Mostramos el estado antes de empezar a entrenar si se pide
        if salida_epoch:
            
            #Simplemente llamamos las funciones.
            perdidaEntrenamiento = self._calcular_perdida_entrenamiento(X, y_binario)
            rendimientoEntrenamiento = self._rendimiento_binario(X, y_binario)
            print(f"Inicialmente, en entrenamiento LOSS: {perdidaEntrenamiento}, rendimiento: {rendimientoEntrenamiento}.")
            
            if Xv is not None:
                #Volvemos a llamar a las funciones.
                perdidaValidacion = self._calcular_perdida_validacion(Xv, yv_binario)
                rendimientoValidacion = self._rendimiento_binario(Xv, yv_binario)
                print(f"Inicialmente, en validación    LOSS: {perdidaValidacion}, rendimiento: {rendimientoValidacion}.")

        for numeroEpoch in range(1, n_epochs + 1):

            # Calculamos la tasa de aprendizaje para este epoch
            if self.decaimientoTasa:
                tasaActual = self.tasaAprendizajeInicial * (1 / (1 + numeroEpoch))
            else:
                tasaActual = self.tasaAprendizajeInicial

            # Mezclamos los datos aleatoriamente antes de crear los mini-batches del epoch
            indicesAleatorios = np.random.permutation(numEjemplos)
            X_mezclado = X[indicesAleatorios]
            y_mezclado = y_binario[indicesAleatorios]

            # Recorremos todos los mini-batches del epoch
            for inicioDelBatch in range(0, numEjemplos, self.tamanoMiniBatch):
                X_batch = X_mezclado[inicioDelBatch : inicioDelBatch + self.tamanoMiniBatch]
                y_batch = y_mezclado[inicioDelBatch : inicioDelBatch + self.tamanoMiniBatch]
                tamBatch = len(y_batch)

                # Predicción de probabilidades para cada ejemplo del batch.
                # np.dot(X_batch, self.pesos): multiplica cada fila de X_batch por el vector de pesos
                # y suma, dando un valor por ejemplo. Sumamos el sesgo y aplicamos sigmoide.
                probabilidadesBatch = sigmoide(np.dot(X_batch, self.pesos) + self.sesgo)

                # Diferencia entre lo predicho y lo real (error del modelo en este batch)
                erroresBatch = probabilidadesBatch - y_batch

                # Gradiente respecto a los pesos.
                # np.dot(X_batch.T, erroresBatch): para cada peso_j acumula la suma de (error_i * valor_ij) para todos los ejemplos del batch.
                # X_batch.T transpone X_batch
                # Dividimos por tamBatch para obtener el gradiente medio y añadimos la regularización L2.
                gradientePesos = np.dot(X_batch.T, erroresBatch) / tamBatch + self.constanteRegularizacion * self.pesos
                gradienteSesgo = np.mean(erroresBatch) #No se regulariza, práctica estandar

                # Actualizamos pesos y sesgo en dirección contraria al gradiente
                self.pesos -= tasaActual * gradientePesos
                self.sesgo -= tasaActual * gradienteSesgo

            # Mostramos pérdida y rendimiento al final del epoch si se pide
            if salida_epoch:
                #Llamamos a las funciones
                perdidaEntrenamiento = self._calcular_perdida_entrenamiento(X, y_binario)
                rendimientoEntrenamiento = self._rendimiento_binario(X, y_binario)
                print(f"Epoch {numeroEpoch}, en entrenamiento LOSS: {perdidaEntrenamiento}, rendimiento: {rendimientoEntrenamiento}.")
                
                if Xv is not None:
                    #Llamamos a las funciones
                    perdidaValidacion = self._calcular_perdida_validacion(Xv, yv_binario)
                    rendimientoValidacion = self._rendimiento_binario(Xv, yv_binario)
                    print(f"         en validación    LOSS: {perdidaValidacion}, rendimiento: {rendimientoValidacion}.")

            # Comprobamos early stopping solo si tenemos conjunto de validación
            if early_stopping and Xv is not None:
                # Si ya calculamos la pérdida de validación en salida_epoch, la reutilizamos
                if not salida_epoch:
                    perdidaValidacion = self._calcular_perdida_validacion(Xv, yv_binario)

                if perdidaValidacion < mejorPerdidaValidacion:
                    # Hemos mejorado: reiniciamos el contador de paciencia
                    mejorPerdidaValidacion = perdidaValidacion
                    epochsSinMejora = 0
                else:
                    epochsSinMejora += 1

                # Si llevamos 'paciencia' epochs consecutivos sin mejorar, paramos
                if epochsSinMejora >= paciencia:
                    if salida_epoch:
                        print("PARADA TEMPRANA")
                    break

        self.entrenado = True

    def _calcular_perdida_entrenamiento(self, X, y_binario):
        # Pérdida de entrenamiento = entropía cruzada + penalización L2 sobre los pesos.
        # np.dot(X, self.pesos) calcula la combinación lineal para todos los ejemplos a la vez.
        probabilidades = sigmoide(np.dot(X, self.pesos) + self.sesgo)
        # np.where(condicion, valor_si_true, valor_si_false):
        # si el ejemplo es positivo (y=1) usamos log(p), si es negativo (y=0) usamos log(1-p).
        # El +1e-15 evita log(0) que daría infinito.
        entropiaCruzada = -np.sum(
            np.where(y_binario == 1,
                     np.log(probabilidades + 1e-15),
                     np.log(1 - probabilidades + 1e-15))
        )
        # La regularización penaliza pesos grandes para evitar sobreajuste; no se aplica al sesgo
        penalizacionL2 = self.constanteRegularizacion * np.sum(self.pesos ** 2)
        return entropiaCruzada + penalizacionL2

    def _calcular_perdida_validacion(self, Xv, yv_binario):
        # Para validación solo usamos entropía cruzada (sin regularización, porque la
        # regularización es una restricción sobre los pesos del modelo, no sobre los datos)
        probabilidades = sigmoide(np.dot(Xv, self.pesos) + self.sesgo)
        entropiaCruzada = -np.sum(
            np.where(yv_binario == 1,
                     np.log(probabilidades + 1e-15),
                     np.log(1 - probabilidades + 1e-15))
        )
        return entropiaCruzada

    def _rendimiento_binario(self, X, y_binario):
        # Método interno que trabaja con etiquetas 0/1 en lugar de las clases originales.
        # Si la probabilidad predicha es >= 0.5 clasificamos como 1 (clase positiva), si no como 0.
        predicciones = (sigmoide(np.dot(X, self.pesos) + self.sesgo) >= 0.5).astype(float)
        return np.mean(predicciones == y_binario)

    def clasifica_prob(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")
        # Devuelve la probabilidad de que cada ejemplo pertenezca a la clase positiva
        return sigmoide(np.dot(ejemplos, self.pesos) + self.sesgo)

    def clasifica(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")
        probabilidades = self.clasifica_prob(ejemplos)
        # Si probabilidad >= 0.5 predecimos clase positiva, si no clase negativa.
        # Xv e yv son el conjunto de validación, que sirve para monitorizar si el modelo
        # empieza a empeorar en datos no vistos (se usa en early_stopping).
        clasePositiva = self.clases[1]
        claseNegativa = self.clases[0]
        return np.where(probabilidades >= 0.5, clasePositiva, claseNegativa)


# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================


# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam,reg) para mejorar el rendimiento
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones).
# Usar para ello un conjunto de validación.

# Describir el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba (dejarlo todo como comentario)


# ============================================================================
# 5.0) UTILIDADES GENERALES PARA EL AJUSTE DE HIPERPARÁMETROS
# ============================================================================

def _combinaciones_rejilla(rejilla):
    """Genera la lista de todas las combinaciones de una rejilla de hiperparámetros.

    'rejilla' es un diccionario {nombre_hiperparametro: [valores...]}.
    Devuelve una lista de diccionarios, uno por combinación.
    """
    # Extraemos los nombres de los hiperparámetros y sus listas de valores
    nombres = list(rejilla.keys())
    listas_valores = [rejilla[n] for n in nombres]

    # product genera el producto cartesiano de las listas;
    # zip+dict convierte cada tupla resultado en un diccionario {nombre: valor}
    lista_combinaciones = [dict(zip(nombres, combo)) for combo in product(*listas_valores)]
    return lista_combinaciones


def ajusta_RL(X_entr, y_entr, X_val, y_val, rejilla, n_epochs=100, traza=False):
    """Búsqueda en rejilla sobre un conjunto de validación para RegresionLogisticaMiniBatch.

    Parámetros
      X_entr, y_entr : conjunto de entrenamiento.
      X_val,  y_val  : conjunto de validación.
      rejilla        : dict {hiperparametro: [valores...]}.
      n_epochs       : número de epochs para entrenar cada candidato.
      traza          : si True, imprime el rendimiento de cada combinación.

    Devuelve
      mejores_params : dict con la mejor combinación encontrada.
      mejor_rend_val : rendimiento en validación de esa combinación.
      resultados     : lista de (params, rend_val) ordenada de mejor a peor.
    """
    mejores_params = None
    mejor_rend_val = -1.0
    resultados = []

    for params in _combinaciones_rejilla(rejilla):
        # Creamos y entrenamos un modelo con esta combinación de hiperparámetros
        modelo = RegresionLogisticaMiniBatch(**params)
        modelo.entrena(X_entr, y_entr, n_epochs=n_epochs)

        # Evaluamos sobre validación (no sobre entrenamiento, para no seleccionar por sobreajuste)
        rend = rendimiento(modelo, X_val, y_val) #Rendimiento, función creada al inicio.
        resultados.append((params, rend))

        if traza:
            print(f"   {params} -> validación: {rend:.4f}")

        if rend > mejor_rend_val:
            mejor_rend_val = rend
            mejores_params = params

    # Ordenamos de mejor a peor para facilitar la documentación del ranking
    resultados.sort(key=lambda t: t[1], reverse=True)
    return mejores_params, mejor_rend_val, resultados


def evalua_RL_completo(nombre, X, y, rejilla,
                       Xp=None, yp=None, test=0.2, val=0.2,
                       normalizar=False, n_epochs=100, traza=False):
    """Flujo completo de aplicación para un dataset binario:
      1) Reservar conjunto de prueba (holdout) si no viene dado.
      2) Separar un conjunto de validación del entrenamiento.
      3) (Opcional) normalizar ajustando solo con el entrenamiento.
      4) Búsqueda en rejilla sobre validación -> mejores hiperparámetros.
      5) Reentrenar con entrenamiento+validación y medir sobre prueba.

    Si el dataset ya trae su propio conjunto de prueba (IMDB), se pasa en
    Xp, yp y X, y son el conjunto de entrenamiento completo.
    """
    print(f"\n===== {nombre} =====")

    # 1) Conjunto de prueba: reservado para la evaluación final, no interviene en el ajuste
    if Xp is None:
        Xev, Xp, yev, yp = particion_entr_prueba(X, y, test=test)
    else:
        Xev, yev = X, y

    # 2) Conjunto de validación extraído del entrenamiento
    Xe, Xv, ye, yv = particion_entr_prueba(Xev, yev, test=val)

    # 3) Normalización: ajustar SOLO sobre entrenamiento y aplicar con esos mismos parámetros
    if normalizar:
        norm = NormalizadorStandard()
        norm.ajusta(Xe)
        Xe = norm.normaliza(Xe)
        Xv = norm.normaliza(Xv)

    # 4) Búsqueda en rejilla sobre validación
    print(" Buscando en rejilla sobre validación...")
    mejores, rend_val, resultados = ajusta_RL(Xe, ye, Xv, yv, rejilla,
                                              n_epochs=n_epochs, traza=traza)
    print(f" Mejor combinación en validación: {mejores}")
    print(f" Rendimiento en validación: {rend_val:.4f}")

    # 5) Modelo final: reentrenamos con entrenamiento+validación usando los mejores hiperparámetros.
    #    El normalizador también se reajusta sobre el conjunto completo (entrenamiento+validación).
    if normalizar:
        norm_final = NormalizadorStandard()
        norm_final.ajusta(Xev)
        Xev_f = norm_final.normaliza(Xev)
        Xp_f = norm_final.normaliza(Xp)
    else:
        Xev_f, Xp_f = Xev, Xp

    modelo_final = RegresionLogisticaMiniBatch(**mejores)
    modelo_final.entrena(Xev_f, yev, n_epochs=n_epochs)

    rend_entr = rendimiento(modelo_final, Xev_f, yev)
    rend_test = rendimiento(modelo_final, Xp_f, yp)
    print(f" Rendimiento FINAL en entrenamiento+val: {rend_entr:.4f}")
    print(f" Rendimiento FINAL en PRUEBA:            {rend_test:.4f}")

    return {"dataset": nombre, "mejores": mejores,
            "rend_val": rend_val, "rend_entr": rend_entr,
            "rend_test": rend_test, "ranking": resultados,
            "modelo": modelo_final}


# ============================================================================
# 5.1) APLICACIÓN A LOS TRES CONJUNTOS DE DATOS
# ============================================================================
# Usamos rejillas pequeñas para no ser demasiado exhaustivos (como pide el enunciado).

# Rejilla base: 2 * 2 * 2 * 2 = 16 combinaciones
REJILLA_RL = {
    "rate":       [0.1, 0.01],      # tasa de aprendizaje inicial
    "rate_decay": [True, False],    # decaimiento de la tasa por epoch
    "batch_tam":  [32, 64],         # tamaño de mini-batch
    "reg":        [0.0, 0.01],      # constante de regularización L2
}

# Rejilla para IMDB (dataset mayor, entrena más lento): 2 * 2 * 1 * 2 = 8 combinaciones
REJILLA_IMDB = {
    "rate":       [0.1, 0.01],
    "rate_decay": [True, False],
    "batch_tam":  [64],
    "reg":        [0.0, 0.01],
}

# --- VOTOS (no se normaliza: características de voto en rango homogéneo) ---
# res_votos = evalua_RL_completo("VOTOS", X_votos, y_votos, REJILLA_RL,
#                                test=0.2, val=0.2, normalizar=False,
#                                n_epochs=100, traza=False)

# --- CÁNCER (características continuas -> SÍ se normaliza) ---
# res_cancer = evalua_RL_completo("CÁNCER", X_cancer, y_cancer, REJILLA_RL,
#                                 test=0.2, val=0.2, normalizar=True,
#                                 n_epochs=100, traza=False)

# --- IMDB (el conjunto de prueba ya viene dado por el dataset) ---
# res_imdb = evalua_RL_completo("IMDB", X_train_imdb, y_train_imdb, REJILLA_IMDB,
#                               Xp=X_test_imdb, yp=y_test_imdb,
#                               normalizar=False, n_epochs=100, traza=False)


# ============================================================================
# 5.2) DESCRIPCIÓN DEL PROCESO Y RENDIMIENTOS
# ============================================================================

# PROCESO SEGUIDO (igual para los tres datasets):
#   1. Se reserva un 20% como conjunto de prueba (holdout estratificado, Ej. 1).
#      En IMDB el conjunto de prueba ya viene dado por el propio dataset.
#   2. Del resto se separa un 20% como conjunto de validación.
#   3. Para CÁNCER se estandarizan las características (NormalizadorStandard),
#      ajustando el normalizador solo con el entrenamiento.
#      Para VOTOS e IMDB no hace falta (rangos ya homogéneos).
#   4. Búsqueda en rejilla sobre {rate, rate_decay, batch_tam, reg},
#      eligiendo la combinación con mejor rendimiento en validación.
#   5. Con esa combinación se reentrena el modelo final con entrenamiento+validación.
#      Se mide el rendimiento sobre el conjunto de prueba.

# RESULTADOS OBTENIDOS:

#   VOTOS:
#     - Mejor combinación: rate=___, rate_decay=___, batch_tam=___, reg=___
#     - Rendimiento validación: ___
#     - Rendimiento prueba:     ___
#
#   CÁNCER:
#     - Mejor combinación: rate=___, rate_decay=___, batch_tam=___, reg=___
#     - Rendimiento validación: ___
#     - Rendimiento prueba:     ___
#
#   IMDB:
#     - Mejor combinación: rate=___, rate_decay=___, batch_tam=___, reg=___
#     - Rendimiento validación: ___
#     - Rendimiento prueba:     ___

# OBSERVACIONES:
#   - En cáncer, sin normalizar el entrenamiento es inestable; al estandarizar,
#     el rendimiento mejora notablemente (características con rangos muy distintos:
#     tema 6, diap. 43).
#   - Una pequeña regularización L2 (reg=0.01) suele mejorar la generalización
#     frente a reg=0 (tema 9, diap. 7: control del sobreajuste).
#   - rate_decay=True estabiliza el descenso cuando la tasa inicial es alta.


# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest.


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64,reg=0.01):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......




#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a
#  realizar (donde k es el número de clases) (
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.



#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------

# ============================================================================
# Implementación: clase RL_OvR (One vs Rest)
# ============================================================================
#
# IDEA DE One vs Rest (OvR):
#   Si hay k clases, se entrenan k clasificadores binarios. El clasificador
#   i-ésimo aprende a distinguir "¿es de la clase i?" (positivo) frente a
#   "¿es de cualquier otra clase?" (negativo). Para predecir, se elige la
#   clase cuyo clasificador da la probabilidad MÁS ALTA (argmax).

class RL_OvR():

    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64, reg=0.01):
        # Se guardan los hiperparámetros que se pasarán a cada clasificador binario base
        # (significan lo mismo que en RegresionLogisticaMiniBatch).
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.reg = reg

        self.clases = None          # clases reales del problema (en orden)
        self.clasificadores = None  # clasificador binario por cada clase

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        # Clases del problema multiclase
        self.clases = np.unique(y)

        # Un clasificador binario base por cada clase (esquema One vs Rest)
        self.clasificadores = []
        for c in self.clases:
            # Convertimos el problema multiclase en uno binario para esta clase:
            # 1 si el ejemplo es de la clase c, 0 si es de cualquier otra
            y_binaria = np.where(y == c, 1, 0)

            # Creamos un clasificador binario con los hiperparámetros guardados
            clf = RegresionLogisticaMiniBatch(rate=self.rate,
                                              rate_decay=self.rate_decay,
                                              batch_tam=self.batch_tam,
                                              reg=self.reg)
            if salida_epoch:
                print(f"--- Entrenando clasificador binario para la clase {c} ---")
            clf.entrena(X, y_binaria, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.clasificadores.append(clf)

    def clasifica(self, ejemplos):
        if self.clasificadores is None:
            raise ClasificadorNoEntrenado(
                "El clasificador RL_OvR debe entrenarse antes de clasificar.")

        # Para cada clasificador binario obtenemos la probabilidad de pertenecer a su clase.
        # np.column_stack apila esos vectores por columnas:
        # resultado tiene shape (n_ejemplos, n_clases)
        probabilidades = np.column_stack(
            [clf.clasifica_prob(ejemplos) for clf in self.clasificadores])

        # La clase predicha es la del clasificador con mayor probabilidad (argmax por filas)
        indices_ganadores = np.argmax(probabilidades, axis=1)
        return self.clases[indices_ganadores]


# =====================================================
# EJERCICIO 7: APLICANDO LOS CLASIFICADORES MULTICLASE
# =====================================================

# -------------------------
# 7.1) Codificación one-hot
# -------------------------


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles
# valores del atributo. El valor i-ésimo del atributo se convierte en k atributos
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.

# Por ejemplo, sin un atributo tiene tres posibles valores "a", "b" y "c", ese atributo
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)

# Definir una función:

#     codifica_one_hot(X)

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto
# hay que codificarlos todos.

# NOTA: NO USAR PANDAS NI SKLEARN PARA ESTA FUNCIÓN

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.

# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])

# >>> codifica_one_hot(Xc)
#
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11

def codifica_one_hot(X):
    """Codificación one-hot de un conjunto de datos X (array de numpy).

    Se presupone que todos los atributos son categóricos.
    Cada columna se reemplaza por tantas columnas binarias (0/1) como valores
    distintos tenga, en orden.
    """
    columnas_codificadas = []

    # Procesamos cada columna por separado
    for j in range(X.shape[1]):
        columna = X[:, j]

        # Valores distintos de esta columna, ordenados
        categorias = np.unique(columna)

        # columna[:, None] convierte el vector columna en una matriz de forma (n, 1).
        # categorias[None, :] convierte el vector de categorías en una matriz de forma (1, k).
        # La comparación (==) produce una matriz booleana (n, k)
        # donde la posición [i, j] es True si el ejemplo i tiene la categoría j.
        # .astype(float) convierte True/False en 1.0/0.0.
        bloque = (columna[:, None] == categorias[None, :]).astype(float)
        columnas_codificadas.append(bloque)

    # Concatenamos horizontalmente los bloques de todas las columnas
    return np.hstack(columnas_codificadas)


# ---------------------------------------------------------
# 7.2) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR del ejercicio anterior y la de one-hot del
# apartado anterior, para obtener un clasificador que aconseje la concesión,
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito.

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado
# exhaustivo). Describirlo en los comentarios.


def ajusta_OvR(X_entr, y_entr, X_val, y_val, rejilla, n_epochs=100, traza=False):
    """Versión de ajusta_RL para clasificadores multiclase RL_OvR.
    Misma lógica de búsqueda en rejilla, pero usando RL_OvR como modelo base.

    Devuelve
      mejores_params : dict con la mejor combinación encontrada.
      mejor_rend_val : rendimiento en validación de esa combinación.
      resultados     : lista de (params, rend_val) ordenada de mejor a peor.
    """
    mejores_params = None
    mejor_rend_val = -1.0
    resultados = []

    for params in _combinaciones_rejilla(rejilla):
        # Creamos y entrenamos un clasificador OvR con esta combinación de hiperparámetros
        modelo = RL_OvR(**params)
        modelo.entrena(X_entr, y_entr, n_epochs=n_epochs)

        # Evaluamos sobre validación
        rend = rendimiento(modelo, X_val, y_val)
        resultados.append((params, rend))

        if traza:
            print(f"   {params} -> validación: {rend:.4f}")

        if rend > mejor_rend_val:
            mejor_rend_val = rend
            mejores_params = params

    # Ordenamos de mejor a peor para facilitar la documentación
    resultados.sort(key=lambda t: t[1], reverse=True)
    return mejores_params, mejor_rend_val, resultados


# Rejilla de hiperparámetros para OvR (no exhaustiva, como pide el enunciado):
# 2 * 2 * 2 * 2 = 16 combinaciones
REJILLA_OVR = {
    "rate":       [0.1, 0.01],
    "rate_decay": [True, False],
    "batch_tam":  [32, 64],
    "reg":        [0.0, 0.01],
}

# 1) Codificamos one-hot los atributos categóricos de X_credito
# X_credito_oh = codifica_one_hot(X_credito)

# 2) Reservamos un conjunto de prueba (holdout) y extraemos validación del entrenamiento
# Xev_cr, Xp_cr, yev_cr, yp_cr = particion_entr_prueba(X_credito_oh, y_credito, test=0.2)
# Xe_cr, Xv_cr, ye_cr, yv_cr = particion_entr_prueba(Xev_cr, yev_cr, test=0.2)

# print("\n===== Crédito =====")
# print("Buscando en rejilla sobre validación...")

# 3) Ajuste de hiperparámetros sobre validación
# mejores_cr, rend_val_cr, _ = ajusta_OvR(Xe_cr, ye_cr, Xv_cr, yv_cr,
#                                          REJILLA_OVR, n_epochs=100)
# print("Crédito - mejor combinación:", mejores_cr,
#       "  rend. validación:", round(rend_val_cr, 4))

# 4) Modelo final: reentrenamos con entrenamiento+validación y medimos sobre prueba
# modelo_credito = RL_OvR(**mejores_cr)
# modelo_credito.entrena(Xev_cr, yev_cr, n_epochs=100)
# print("Crédito - rendimiento entrenamiento:", round(rendimiento(modelo_credito, Xev_cr, yev_cr), 4))
# print("Crédito - rendimiento PRUEBA:       ", round(rendimiento(modelo_credito, Xp_cr, yp_cr), 4))

# CRÉDITO — DESCRIPCIÓN Y RENDIMIENTOS:
#   - Se aplica one-hot porque los atributos son categóricos.
#   - Búsqueda en rejilla sobre validación (tema 9, diap. 20) y evaluación final
#     sobre prueba reservada holdout (tema 4, diap. 2).
#   - Mejor combinación: {'rate': 0.1, 'rate_decay': False, 'batch_tam': 32, 'reg': 0.0}
#   - Rendimiento validación:  0.762
#   - Rendimiento entrenamiento: 0.6893
#   - Rendimiento prueba: 0.7132


# ---------------------------------------------------------
# 7.3) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR del ejercicio anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en la
#  carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador.

#  Los datos están ya separados en entrenamiento, validación y prueba.

# Se pide:

# * Definir las funciones auxiliares necesarias para cargar el dataset desde los
#   archivos de texto, y crear variables:
#       X_entr_dg, y_entr_dg
#       X_val_dg, y_val_dg
#       X_test_dg, y_test_dg
#   que contengan arrays de numpy con el dataset proporcionado (USAR ESOS NOMBRES).

# * Obtener un modelo de clasificación RL_OvR

# * Ajustar los parámetros de tamaño de batch, tasa de aprendizaje, constante de
#   regulrización y rate_decay para tratar de obtener un rendimiento aceptable
#   (por encima del 75% de aciertos sobre test).


ALTO_DIGITO = 28    # número de filas de píxeles por imagen
ANCHO_DIGITO = 28   # número de columnas de píxeles por imagen


def carga_imagenes_digitos(ruta_imagenes, alto=ALTO_DIGITO, ancho=ANCHO_DIGITO):
    """Lee un fichero de imágenes de dígitos.
    Devuelve un array de numpy (n_imagenes, alto*ancho).
    Cada imagen queda aplanada en un vector de 0s y 1s.
    """
    with open(ruta_imagenes) as f:
        # No usamos strip() completo: los espacios son píxeles blancos y no deben eliminarse.
        # Solo quitamos el salto de línea final.
        lineas = [linea.rstrip("\n").rstrip("\r") for linea in f]

    imagenes = []
    # Las imágenes vienen seguidas sin separador; cada una ocupa 'alto' líneas consecutivas
    for inicio in range(0, len(lineas), alto):
        bloque = lineas[inicio:inicio + alto]
        if len(bloque) < alto:
            break  # por si el fichero termina con líneas incompletas

        filas_pixeles = []
        for fila in bloque:
            # Rellenamos/recortamos cada línea a 'ancho' caracteres por seguridad
            # (algunas líneas pueden venir con espacios finales recortados)
            fila = fila.ljust(ancho)[:ancho]
            # ' ' -> 0 (blanco); '+' y '#' -> 1 (negro, no distinguimos borde/interior)
            filas_pixeles.append([0 if c == " " else 1 for c in fila])

        # Aplanamos la imagen 28x28 en un vector de 784 componentes
        imagenes.append(np.array(filas_pixeles, dtype=float).flatten())

    return np.array(imagenes)


def carga_etiquetas_digitos(ruta_etiquetas):
    """Lee un fichero de etiquetas (un dígito por línea).
    Devuelve un array numpy de enteros.
    """
    with open(ruta_etiquetas) as f:
        return np.array([int(linea.strip()) for linea in f if linea.strip() != ""])


# print("\n ==== Dígitos ====")
# Carga de los tres conjuntos del dataset de dígitos
RUTA_DIGITOS = "datos/digitdata/"
X_entr_dg = carga_imagenes_digitos(RUTA_DIGITOS + "trainingimages")
y_entr_dg = carga_etiquetas_digitos(RUTA_DIGITOS + "traininglabels")
X_val_dg  = carga_imagenes_digitos(RUTA_DIGITOS + "validationimages")
y_val_dg  = carga_etiquetas_digitos(RUTA_DIGITOS + "validationlabels")
X_test_dg = carga_imagenes_digitos(RUTA_DIGITOS + "testimages")
y_test_dg = carga_etiquetas_digitos(RUTA_DIGITOS + "testlabels")


# print("Carga de datos completada.")
# El dataset ya viene partido en entrenamiento/validación/prueba:
# ajustamos hiperparámetros con el conjunto de validación dado
# mejores_dg, rend_val_dg, _ = ajusta_OvR(X_entr_dg, y_entr_dg,
#                                           X_val_dg, y_val_dg,
#                                           REJILLA_OVR, n_epochs=100)
# print("Dígitos - mejor combinación:", mejores_dg,
#       "  rend. validación:", round(rend_val_dg, 4))

# Modelo final: entrenamos con entrenamiento (el test ya está dado aparte)
# modelo_digitos = RL_OvR(**mejores_dg)
# modelo_digitos.entrena(X_entr_dg, y_entr_dg, n_epochs=100)
# print("Dígitos - rendimiento entrenamiento:", round(rendimiento(modelo_digitos, X_entr_dg, y_entr_dg), 4))
# print("Dígitos - rendimiento validación:   ", round(rendimiento(modelo_digitos, X_val_dg, y_val_dg), 4))
# print("Dígitos - rendimiento TEST:         ", round(rendimiento(modelo_digitos, X_test_dg, y_test_dg), 4))

# DÍGITOS — DESCRIPCIÓN Y RENDIMIENTOS:
#   - Cada imagen 28x28 se aplana en un vector binario de 784 píxeles.
#   - El dataset ya viene partido en entrenamiento/validación/prueba.
#     Se ajustan hiperparámetros con validación y se da el rendimiento sobre test.
#   - Mejor combinación: rate=___, rate_decay=___, batch_tam=___, reg=___
#   - Rendimiento entrenamiento: ___
#   - Rendimiento validación:    ___
#   - Rendimiento test:          ___  (objetivo: > 75%)


# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES (INCLUIDAS LAS AUXILIARES QUE SE
# HUBIERAN NECESITADO) Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS,
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO
# Y DESCOMENTAR LAS QUE VIENE A CONTINUACIÓN.

# EN EL APARTADO FINAL DE RENDIMIENTOS FINALES, USAR LA MEJOR COMBINACIÓN DE
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE.

# EL ARCHIVO trabajo-1-iacd-24-25-sc.py SERA CARGADO POR EL PROFESOR,
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos-trabajo-1-iacd.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS.



# *********** DESCOMENTAR A PARTIR DE AQUÍ

print("************ PRUEBAS EJERCICIO 1:")
print("**********************************\n")
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
print("\n")

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
print("\n")

Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
print("\n\n\n")




print("************ PRUEBAS EJERCICIO 2:")
print("**********************************\n")

nb_tenis=NaiveBayesCat(k=0.5)
nb_tenis.entrena(X_tenis,y_tenis)
ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
print("NB Clasifica_prob un ejemplo tenis: ",nb_tenis.clasifica_prob(ej_tenis))
print("NB Clasifica un ejemplo tenis: ",nb_tenis.clasifica([ej_tenis]))
print("\n")

nb_votos=NaiveBayesCat(k=1)
nb_votos.entrena(Xe_votos,ye_votos)
print("NB Rendimiento votos sobre entrenamiento: ", rendimiento(nb_votos,Xe_votos,ye_votos))
print("NB Rendimiento votos sobre test: ", rendimiento(nb_votos,Xp_votos,yp_votos))
print("\n")


nb_credito=NaiveBayesCat(k=1)
nb_credito.entrena(Xe_credito,ye_credito)
print("NB Rendimiento crédito sobre entrenamiento: ", rendimiento(nb_credito,Xe_credito,ye_credito))
print("NB Rendimiento crédito sobre test: ", rendimiento(nb_credito,Xp_credito,yp_credito))
print("\n")


nb_imdb=NaiveBayesCat(k=1)
nb_imdb.entrena(X_train_imdb,y_train_imdb)
print("NB Rendimiento imdb sobre entrenamiento: ", rendimiento(nb_imdb,X_train_imdb,y_train_imdb))
print("NB Rendimiento imdb sobre test: ", rendimiento(nb_imdb,X_test_imdb,y_test_imdb))
print("\n")




normst_cancer=NormalizadorStandard()
normst_cancer.ajusta(Xe_cancer)
Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

print("Normalización cancer entrenamiento: ",np.mean(Xe_cancer_n,axis=0))
print("Normalización cancer validación: ",np.mean(Xv_cancer_n,axis=0))
print("Normalización cancer test: ",np.mean(Xp_cancer_n,axis=0))

print("\n\n\n")



nb_cancer=NaiveBayesGauss()
nb_cancer.entrena(Xe_cancer_n,ye_cancer)
print("NB rendimiento cáncer entrenamiento: ", rendimiento(nb_cancer,Xe_cancer_n,ye_cancer))
print("NB rendimiento cáncer prueba: ", rendimiento(nb_cancer,Xp_cancer_n,yp_cancer))




print("************ PRUEBAS EJERCICIO 5:")
print("**********************************\n")


lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)
print("LR clasifica cuatro ejemplos cáncer (y valor esperado): ",lr_cancer.clasifica(Xp_cancer_n[17:21]),yp_cancer[17:21])
print("LR clasifica_prob cuatro ejemplos cáncer: ", lr_cancer.clasifica_prob(Xp_cancer_n[17:21]))
print("LR rendimiento cáncer entrenamiento: ", rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
print("LR rendimiento cáncer prueba: ", rendimiento(lr_cancer,Xp_cancer_n,yp_cancer))

print("\n\n CON SALIDA Y EARLY STOPPING**********************************\n")

lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

print("\n\n\n")

print("************ PRUEBAS EJERCICIO 6:")
print("**********************************\n")

Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=16)

rl_iris_ovr.entrena(Xe_iris,ye_iris)

print("OvR Rendimiento entrenamiento iris: ",rendimiento(rl_iris_ovr,Xe_iris,ye_iris))
print("OvR Rendimiento prueba iris: ",rendimiento(rl_iris_ovr,Xp_iris,yp_iris))
print("\n\n\n")



print("************ RENDIMIENTOS FINALES REGRESIÓN LOGÍSTICA EN CRÉDITO, IMDB y DÍGITOS")
print("*******************************************************************************\n")


# ATENCIÓN: EN CADA CASO, USAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS QUE SE HA
# DEBIDO OBTENER EN EL PROCESO DE AJUSTE

print("==== MEJOR RENDIMIENTO RL SOBRE VOTOS:")
RL_VOTOS=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=False,batch_tam=32,reg=0.0)
RL_VOTOS.entrena(Xe_votos,ye_votos) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre votos: ",rendimiento(RL_VOTOS,Xe_votos,ye_votos))
print("Rendimiento RL test sobre votos: ",rendimiento(RL_VOTOS,Xp_votos,yp_votos))
print("\n")


print("==== MEJOR RENDIMIENTO RL SOBRE CÁNCER:")
RL_CANCER=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=False,batch_tam=32,reg=0.0)
RL_CANCER.entrena(Xe_cancer,ye_cancer) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre cáncer: ",rendimiento(RL_CANCER,Xe_cancer,ye_cancer))
print("Rendimiento RL test sobre cancer: ",rendimiento(RL_CANCER,Xp_cancer,yp_cancer))
print("\n")


print("==== MEJOR RENDIMIENTO RL_OvR SOBRE CREDITO:")
X_credito_oh=codifica_one_hot(X_credito)
Xe_credito_oh,Xp_credito_oh,ye_credito,yp_credito=particion_entr_prueba(X_credito_oh,y_credito,test=0.3)

RL_CLASIF_CREDITO=RL_OvR(rate=0.1,rate_decay=False,batch_tam=32,reg=0.01)
RL_CLASIF_CREDITO.entrena(Xe_credito_oh,ye_credito) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RLOVR  entrenamiento sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xe_credito_oh,ye_credito))
print("Rendimiento RLOVR  test sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xp_credito_oh,yp_credito))
print("\n")


print("==== MEJOR RENDIMIENTO RL SOBRE IMDB:")
RL_IMDB=RegresionLogisticaMiniBatch(rate=0.01,rate_decay=False,batch_tam=64,reg=0.0)
RL_IMDB.entrena(X_train_imdb,y_train_imdb) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre imdb: ",rendimiento(RL_IMDB,X_train_imdb,y_train_imdb))
print("Rendimiento RL test sobre imdb: ",rendimiento(RL_IMDB,X_test_imdb,y_test_imdb))
print("\n")


print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
RL_DG=RL_OvR(rate=0.1,rate_decay=False,batch_tam=32,reg=0.0)
RL_DG.entrena(X_entr_dg,y_entr_dg) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre dígitos: ",rendimiento(RL_DG,X_entr_dg,y_entr_dg))
print("Rendimiento RL validación sobre dígitos: ",rendimiento(RL_DG,X_val_dg,y_val_dg))
print("Rendimiento RL test sobre dígitos: ",rendimiento(RL_DG,X_test_dg,y_test_dg))
