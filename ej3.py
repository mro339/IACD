# ==================================
# EJERCICIO 3: NORMALIZADOR ESTÁNDAR
# ==================================
#
# Implementación de la normalización "standard" (z-score): se traslada y escala
# cada característica para que tenga MEDIA 0 y DESVIACIÓN TÍPICA 1.
#
# ----------------------------------------------------------------------------
# REFERENCIAS EN EL MATERIAL DE CLASE
# ----------------------------------------------------------------------------
# Teoría (tema06preprocesadodedatos.pdf):
#   - Diapositiva 43: "Escalado". Motivación (diferencia de magnitud entre
#                     características) y tipos de ajuste (máximo-mínimo y
#                     estandarización).
#   - Diapositiva 46: "Estandarización". Da la fórmula exacta que implementamos:
#                         x_norm^(i) = (x^(i) - μ_x) / σ_x
#                     donde μ_x es la media de la característica y σ_x su
#                     desviación típica.
#   - Diapositiva 47: ejemplo numérico de aplicar esa fórmula sobre una tabla.
#   - Diapositiva 49: punto CLAVE para el diseño de la clase. Indica que hay que
#                     "guardar los parámetros necesarios para replicar las
#                     transformaciones" (media, etc.), que esos parámetros "se
#                     ajustan sobre el conjunto de entrenamiento" y que "los
#                     datos de prueba se transforman con parámetros ya ajustados".
#                     De ahí la separación en dos métodos: ajusta() (calcula y
#                     guarda μ y σ) y normaliza() (aplica la transformación).
#
# Código de prácticas (tema-06.py):
#   - Líneas 715-717: describen el patrón fit/transform que reproducimos con
#                     ajusta/normaliza: 1) se AJUSTA el estimador sobre el
#                     conjunto de entrenamiento; 2) se usa ese mismo objeto ya
#                     ajustado para TRANSFORMAR entrenamiento y prueba (NO se
#                     reajusta sobre prueba).
#   - Líneas 741-744: uso concreto de StandardScaler de scikit-learn
#                     (scaler.fit(npdata) + scaler.transform(npdata)). Aquí
#                     reproducimos su comportamiento "a mano" con numpy, ya que
#                     en el trabajo NO se permite usar Scikit Learn.
# ----------------------------------------------------------------------------

import numpy as np

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