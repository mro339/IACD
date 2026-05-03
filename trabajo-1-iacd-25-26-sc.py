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

import math
import random
import numpy as np

from carga_datos import *   

# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

def particion_entr_prueba(X,y,test=0.20):

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    proporcion = len(X) * (1 - test)

    proporcion_train = indices[:proporcion]
    proporcion_test = indices[proporcion:]

    X_train, X_test = X[:proporcion_train], X[proporcion_test:]
    y_train, y_test = y[:proporcion_test], X[proporcion_test:]

    return X_train, X_test, y_train, y_test


# =========================================================
# EJERCICIO 2: IMPLEMENTACIÓN DE CLASIFICADORES NAIVE BAYES
# =========================================================

class ClasificadorNoEntrenado(Exception): pass

# ---------------------------------------------
# 2.1) Implementación de Naive Bayes categórico
# ---------------------------------------------

class NaiveBayesCat():

    def __init__(self,k=1):
        self.k = k
        self.entrenado = False         
            
    def entrena(self,X,y):
        n_ejemplos, n_caracteristicas = X.shape
        self.clases, clases_contador = np.unique(y, return_counts=True)
        self.numero_clases = len(self.clases)

        self.prior = {}

        for c, count in zip(self.clases, clases_contador):
            self.prior[c] = count / n_ejemplos
        
  
        self.valores_atributo = []
      
        for j in range(n_caracteristicas):
            self.valores_atributo.append(np.unique(X[:, j])) 

        self.conteos = []
        for j in range(n_caracteristicas): 
            dic_valores = {}
            for valor in self.valores_atributo[j]: 
                dic_valores[valor] = {c: 0 for c in self.clases} 
            self.conteos.append{dic_valores} 

        self.conteos_clase = {c: 0 for c in self.clases}

        for i in range(n_ejemplos):
            c = y[i]
            self.conteos_clase[c] += 1
            for j in range(n_caracteristicas):
                valor = X[i,j]
                self-self.conteos[j][valor][c] +=1


        self.entrenado = True

    def clasifica_prob(self,ejemplo):
        if self.entrenado == False:
            ClasificadorNoEntrenado("El modelo no ha sido Entrenado")

        log_probs = {}

        for c in self.clases:

            log_prob = math.log(self.prior[c])

            for j, valor in enumerate(ejemplo):
                valores_posibles = self.valores_atributo[j]
                num_valores = len(valores_posibles)

                
                count = self.conteos[j].get(valor, {}).get(c, 0)

              
                prob = (count + self.k) / (
                    self.conteo_clase[c] + self.k * num_valores
                )

                log_prob += math.log(prob)

            log_probs[c] = log_prob

        max_log = max(log_probs.values())  

        exp_probs = {}
        for c in self.clases:
            exp_probs[c] = math.exp(log_probs[c] - max_log)

        suma = sum(exp_probs.values())

        probs = {}
        for c in self.clases:
            probs[c] = exp_probs[c] / suma

        return probs



    def clasifica(self,ejemplos):

        if self.entrenado == False:
            ClasificadorNoEntrenado("El modelo no ha sido Entrenado")

        predicciones = []

        for ejemplo in ejemplos:
            probs = self.clasifica_prob(ejemplo)

            clase_predicha = max(probs, key=probs.get)
            predicciones.append(clase_predicha)

        return np.array(predicciones)