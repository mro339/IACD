# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================
#
# Implementación de un clasificador multiclase (más de dos clases) mediante la
# técnica One vs Rest (OvR, "uno contra el resto"), usando como clasificador
# binario base la RegresionLogisticaMiniBatch del Ejercicio 4.
#
# IDEA DE One vs Rest:
#   Si hay k clases, se entrenan k clasificadores binarios. El clasificador
#   i-ésimo aprende a distinguir "¿es de la clase i?" (positivo) frente a
#   "¿es de cualquier otra clase?" (negativo). Para predecir un ejemplo nuevo,
#   se pregunta a los k clasificadores por la probabilidad de pertenecer a su
#   clase, y se elige la clase cuyo clasificador da la probabilidad MÁS ALTA
#   (argmax). Es una forma general de convertir un clasificador binario en uno
#   multiclase sin cambiar el algoritmo base.
#
# ----------------------------------------------------------------------------
# REFERENCIAS EN EL MATERIAL DE CLASE
# ----------------------------------------------------------------------------
# Teoría — tema4_Evaluacion_de_modelos.pdf:
#   - Diapositivas 45-46: "Evaluación multiclase" / "Métricas en multiclase".
#     Justifican que en multiclase la tasa de acierto (accuracy) se define igual
#     (proporción de ejemplos bien clasificados), que es lo que mide
#     'rendimiento' (Ej. 2) sobre las predicciones de este clasificador OvR.
#
# Dependencias de otros ejercicios del mismo trabajo:
#   - RegresionLogisticaMiniBatch (Ej. 4): clasificador binario base. Usamos su
#     método clasifica_prob, que devuelve la probabilidad de la clase positiva.
#     Recordatorio del Ej. 4: la clase positiva es la que aparece EN SEGUNDO
#     lugar en su atributo self.clases. Por eso, al construir cada problema
#     binario etiquetamos con 0/1 (0 = "resto", 1 = "es la clase c"): como
#     np.unique([0,1]) = [0,1], la clase positiva es 1, es decir, "es la clase c".
#   - rendimiento (Ej. 2): se usa para evaluar el modelo (en el Ej. 7).
#
# NOTA del enunciado: por simplicidad, en OvR NO hay conjunto de validación ni
# parada temprana; solo rate, rate_decay, batch_tam y reg.
# ----------------------------------------------------------------------------

class RL_OvR():

    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64, reg=0.01):
        """Se guardan los hiperparámetros que se pasarán a cada clasificador binario base""" 
        # (significan lo mismo que en RegresionLogisticaMiniBatch).
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.reg = reg

        self.clases = None # clases reales del problema (en orden)
        self.clasificadores = None # clasificador binario por cada clase

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        """Entrena k clasificadores binarios, uno por clase (esquema One vs Rest)"""

        # Clases del problema
        self.clases = np.unique(y) # ordenadas y sin repetir

        # Un clasificador binario base por cada clase.
        self.clasificadores = []
        for c in self.clases:
            y_binaria = np.where(y == c, 1, 0) # convierte el problema multiclase en uno binario para la clase c

            # Se crea un clasificador binario con los hiperparámetros guardados.
            clf = RegresionLogisticaMiniBatch(rate=self.rate,
                                              rate_decay=self.rate_decay,
                                              batch_tam=self.batch_tam,
                                              reg=self.reg)

            # Se entrena con el problema binario
            if salida_epoch:
                print(f"--- Entrenando clasificador binario para la clase {c} ---")
            clf.entrena(X, y_binaria, n_epochs=n_epochs, salida_epoch=salida_epoch)

            self.clasificadores.append(clf)

    def clasifica(self, ejemplos):
        """Devuelve el array de clases predichas para un array de ejemplos"""

        # Si no se ha entrenado todavía, se lanza la excepción pedida.
        if self.clasificadores is None:
            raise ClasificadorNoEntrenado(
                "El clasificador RL_OvR debe entrenarse antes de clasificar.")

        # Para cada clasificador binario obtenemos la probabilidad de que cada ejemplo pertenezca a su clase. 
        # Apilando esos vectores por columnas construimos una matriz (n_ejemplos x n_clases) de probabilidades.
        probabilidades = np.column_stack(
            [clf.clasifica_prob(ejemplos) for clf in self.clasificadores])

        # Para cada ejemplo, la clase predicha es la del clasificador con mayor probabilidad (argmax sobre las columnas)
        indices_ganadores = np.argmax(probabilidades, axis=1) # devuelve el índice de columna
        return self.clases[indices_ganadores] # clase real