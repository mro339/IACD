import math
import random
import numpy as np
from scipy.special import expit
from itertools import product

from carga_datos import *


# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA
# ==================================================

def particion_entr_prueba(X, y, test=0.20):
    clasesUnicas = np.unique(y)

    indicesEntrenamiento = []
    indicesPrueba = []

    for clase in clasesUnicas:
        indicesDeEstaClase = np.where(y == clase)[0]
        np.random.shuffle(indicesDeEstaClase)
        numEjemplosPrueba = int(len(indicesDeEstaClase) * test)
        indicesPrueba.extend(indicesDeEstaClase[:numEjemplosPrueba])
        indicesEntrenamiento.extend(indicesDeEstaClase[numEjemplosPrueba:])

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

class ClasificadorNoEntrenado(Exception): pass


class NaiveBayesCat():

    def __init__(self, k=1):
        self.k = k
        self.entrenado = False

    def entrena(self, X, y):
        numEjemplos, numCaracteristicas = X.shape

        self.clases, contadorClases = np.unique(y, return_counts=True)
        self.numeroClases = len(self.clases)
        
        self.probabilidadPriori = {}
        for clase, conteo in zip(self.clases, contadorClases):
            self.probabilidadPriori[clase] = conteo / numEjemplos
        
        self.valoresPosiblesAtributo = []
        for indiceAtributo in range(numCaracteristicas):
            self.valoresPosiblesAtributo.append(np.unique(X[:, indiceAtributo]))
        
        self.conteosAtributos = []
        for indiceAtributo in range(numCaracteristicas):
            
            diccionarioConteos = {}
            for valor in self.valoresPosiblesAtributo[indiceAtributo]:
                diccionarioConteos[valor] = {clase: 0 for clase in self.clases}
            
            self.conteosAtributos.append(diccionarioConteos)

        self.conteosPorClase = {clase: 0 for clase in self.clases}

        for indiceEjemplo in range(numEjemplos):
            clase = y[indiceEjemplo]
            self.conteosPorClase[clase] += 1

            for indiceAtributo in range(numCaracteristicas):
                valor = X[indiceEjemplo, indiceAtributo]
                self.conteosAtributos[indiceAtributo][valor][clase] += 1

        self.entrenado = True

    def clasifica_prob(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")
        
        logProbabilidades = {}

        for clase in self.clases:
            logProbabilidadClase = math.log(self.probabilidadPriori[clase])

            for indiceAtributo, valor in enumerate(ejemplo):
                numValoresPosibles = len(self.valoresPosiblesAtributo[indiceAtributo])
                conteoActual = self.conteosAtributos[indiceAtributo].get(valor, {}).get(clase, 0)

                probabilidadAtributo = (conteoActual + self.k) / (
                    self.conteosPorClase[clase] + self.k * numValoresPosibles
                )
                logProbabilidadClase += math.log(probabilidadAtributo)

            logProbabilidades[clase] = logProbabilidadClase

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
            clasePredicha = max(probabilidades, key=probabilidades.get)
            predicciones.append(clasePredicha)

        return np.array(predicciones)


def rendimiento(clasificador, X, y):
    y_predicho = clasificador.clasifica(X)
    numAciertos = np.sum(y == y_predicho)

    return numAciertos / len(y)


class NaiveBayesGauss():

    def __init__(self):
        self.entrenado = False

    def entrena(self, X, y):
        numEjemplos, numCaracteristicas = X.shape
        self.clases = np.unique(y)
        self.probabilidadPriori = {}
        self.mediasClase = {}
        self.desviacionesClase = {}

        for clase in self.clases:
            mascaraDeEstaClase = (y == clase)
            X_deEstaClase = X[mascaraDeEstaClase]

            self.probabilidadPriori[clase] = np.sum(mascaraDeEstaClase) / numEjemplos
            self.mediasClase[clase] = np.mean(X_deEstaClase, axis=0)
            self.desviacionesClase[clase] = np.std(X_deEstaClase, axis=0)
        self.entrenado = True

    def clasifica_prob(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")
        
        logProbabilidades = {}
        for clase in self.clases:
            logProbabilidadClase = math.log(self.probabilidadPriori[clase])
            medias = self.mediasClase[clase]
            desviaciones = self.desviacionesClase[clase].copy()
            desviaciones[desviaciones == 0] = 1e-9
            logProbabilidadClase += np.sum(
                -np.log(desviaciones)
                - 0.5 * np.log(2 * math.pi)
                - 0.5 * ((ejemplo - medias) / desviaciones) ** 2
            )
            logProbabilidades[clase] = logProbabilidadClase
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


# ==================================
# EJERCICIO 3: NORMALIZADOR ESTÁNDAR
# ==================================

class NormalizadorNoAjustado(Exception):
    pass


class NormalizadorStandard():

    def __init__(self):
        self.medias = None
        self.desviaciones = None

    def ajusta(self, X):
        self.medias = np.mean(X, axis=0)
        self.desviaciones = np.std(X, axis=0, ddof=0)
        self.desviaciones[self.desviaciones == 0] = 1.0

    def normaliza(self, X):
        if self.medias is None:
            raise NormalizadorNoAjustado(
                "El normalizador debe ajustarse (ajusta) antes de normalizar.")
        return (X - self.medias) / self.desviaciones


# ==============================================================
# EJERCICIO 4: REGRESIÓN LOGÍSTICA MINI-BATCH CON REGULARIZACIÓN
# ==============================================================

def sigmoide(x):
    return expit(x)


class RegresionLogisticaMiniBatch():

    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100, batch_tam=64, reg=0.01):
        self.tasaAprendizajeInicial = rate
        self.decaimientoTasa = rate_decay
        self.numEpochsMaximo = n_epochs
        self.tamanoMiniBatch = batch_tam
        self.constanteRegularizacion = reg
        self.entrenado = False

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=100, salida_epoch=False,
                early_stopping=False, paciencia=3):
        numEjemplos, numCaracteristicas = X.shape
        self.clases = list(np.unique(y))
        clasePositiva = self.clases[1]
        y_binario = (y == clasePositiva).astype(float)
        
        if Xv is not None:
            yv_binario = (yv == clasePositiva).astype(float)
        
        self.pesos = np.zeros(numCaracteristicas)
        self.sesgo = 0.0
        mejorPerdidaValidacion = float('inf')
        
        epochsSinMejora = 0
        if salida_epoch:
            perdidaEntrenamiento = self._calcular_perdida_entrenamiento(X, y_binario)
            rendimientoEntrenamiento = self._rendimiento_binario(X, y_binario)
            print(f"Inicialmente, en entrenamiento LOSS: {perdidaEntrenamiento}, rendimiento: {rendimientoEntrenamiento}.")
            if Xv is not None:
                perdidaValidacion = self._calcular_perdida_validacion(Xv, yv_binario)
                rendimientoValidacion = self._rendimiento_binario(Xv, yv_binario)
                print(f"Inicialmente, en validación    LOSS: {perdidaValidacion}, rendimiento: {rendimientoValidacion}.")
        
        for numeroEpoch in range(1, n_epochs + 1):
            if self.decaimientoTasa:
                tasaActual = self.tasaAprendizajeInicial * (1 / (1 + numeroEpoch))
            else:
                tasaActual = self.tasaAprendizajeInicial
            indicesAleatorios = np.random.permutation(numEjemplos)
            X_mezclado = X[indicesAleatorios]
            y_mezclado = y_binario[indicesAleatorios]
            
            for inicioDelBatch in range(0, numEjemplos, self.tamanoMiniBatch):
                X_batch = X_mezclado[inicioDelBatch : inicioDelBatch + self.tamanoMiniBatch]
                y_batch = y_mezclado[inicioDelBatch : inicioDelBatch + self.tamanoMiniBatch]
                tamBatch = len(y_batch)
                probabilidadesBatch = sigmoide(np.dot(X_batch, self.pesos) + self.sesgo)
                erroresBatch = probabilidadesBatch - y_batch
                gradientePesos = np.dot(X_batch.T, erroresBatch) / tamBatch + self.constanteRegularizacion * self.pesos
                gradienteSesgo = np.mean(erroresBatch)
                self.pesos -= tasaActual * gradientePesos
                self.sesgo -= tasaActual * gradienteSesgo
            
            if salida_epoch:
                perdidaEntrenamiento = self._calcular_perdida_entrenamiento(X, y_binario)
                rendimientoEntrenamiento = self._rendimiento_binario(X, y_binario)
                print(f"Epoch {numeroEpoch}, en entrenamiento LOSS: {perdidaEntrenamiento}, rendimiento: {rendimientoEntrenamiento}.")
                if Xv is not None:
                    perdidaValidacion = self._calcular_perdida_validacion(Xv, yv_binario)
                    rendimientoValidacion = self._rendimiento_binario(Xv, yv_binario)
                    print(f"         en validación    LOSS: {perdidaValidacion}, rendimiento: {rendimientoValidacion}.")
            
            if early_stopping and Xv is not None:
                
                if not salida_epoch:
                    perdidaValidacion = self._calcular_perdida_validacion(Xv, yv_binario)
                
                if perdidaValidacion < mejorPerdidaValidacion:
                    mejorPerdidaValidacion = perdidaValidacion
                    epochsSinMejora = 0
                
                else:
                    epochsSinMejora += 1
                
                if epochsSinMejora >= paciencia:
                    if salida_epoch:
                        print("PARADA TEMPRANA")
                    break
        
        self.entrenado = True

    def _calcular_perdida_entrenamiento(self, X, y_binario):
        probabilidades = sigmoide(np.dot(X, self.pesos) + self.sesgo)
        entropiaCruzada = -np.sum(
            np.where(y_binario == 1,
                     np.log(probabilidades + 1e-15),
                     np.log(1 - probabilidades + 1e-15))
        )
        
        penalizacionL2 = self.constanteRegularizacion * np.sum(self.pesos ** 2)
        
        return entropiaCruzada + penalizacionL2

    def _calcular_perdida_validacion(self, Xv, yv_binario):
        probabilidades = sigmoide(np.dot(Xv, self.pesos) + self.sesgo)
        entropiaCruzada = -np.sum(
            np.where(yv_binario == 1,
                     np.log(probabilidades + 1e-15),
                     np.log(1 - probabilidades + 1e-15))
        )
        
        return entropiaCruzada

    def _rendimiento_binario(self, X, y_binario):
        predicciones = (sigmoide(np.dot(X, self.pesos) + self.sesgo) >= 0.5).astype(float)
        
        return np.mean(predicciones == y_binario)

    def clasifica_prob(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")
        
        return sigmoide(np.dot(ejemplos, self.pesos) + self.sesgo)

    def clasifica(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama primero a entrena().")
        
        probabilidades = self.clasifica_prob(ejemplos)
        clasePositiva = self.clases[1]
        claseNegativa = self.clases[0]
        
        return np.where(probabilidades >= 0.5, clasePositiva, claseNegativa)


# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================

def _combinaciones_rejilla(rejilla):
    nombres = list(rejilla.keys())
    listas_valores = [rejilla[n] for n in nombres]
    lista_combinaciones = [dict(zip(nombres, combo)) for combo in product(*listas_valores)]
    
    return lista_combinaciones


def ajusta_RL(X_entr, y_entr, X_val, y_val, rejilla, n_epochs=100, traza=False):
    mejores_params = None
    mejor_rend_val = -1.0
    resultados = []
    
    for params in _combinaciones_rejilla(rejilla):
        modelo = RegresionLogisticaMiniBatch(**params)
        modelo.entrena(X_entr, y_entr, n_epochs=n_epochs)
        rend = rendimiento(modelo, X_val, y_val)
        resultados.append((params, rend))
        
        if traza:
            print(f"   {params} -> validación: {rend:.4f}")
        
        if rend > mejor_rend_val:
            mejor_rend_val = rend
            mejores_params = params
    resultados.sort(key=lambda t: t[1], reverse=True)
    
    return mejores_params, mejor_rend_val, resultados


def evalua_RL_completo(nombre, X, y, rejilla,
                       Xp=None, yp=None, test=0.2, val=0.2,
                       normalizar=False, n_epochs=100, traza=False):
    print(f"\n===== {nombre} =====")
    
    if Xp is None:
        Xev, Xp, yev, yp = particion_entr_prueba(X, y, test=test)
    
    else:
        Xev, yev = X, y
    
    Xe, Xv, ye, yv = particion_entr_prueba(Xev, yev, test=val)
    
    if normalizar:
        norm = NormalizadorStandard()
        norm.ajusta(Xe)
        Xe = norm.normaliza(Xe)
        Xv = norm.normaliza(Xv)
    
    print(" Buscando en rejilla sobre validación...")
    
    mejores, rend_val, resultados = ajusta_RL(Xe, ye, Xv, yv, rejilla,
                                              n_epochs=n_epochs, traza=traza)
    
    print(f" Mejor combinación en validación: {mejores}")
    print(f" Rendimiento en validación: {rend_val:.4f}")
    
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


REJILLA_RL = {
    "rate":       [0.1, 0.01],
    "rate_decay": [True, False],
    "batch_tam":  [32, 64],
    "reg":        [0.0, 0.01],
}

REJILLA_IMDB = {
    "rate":       [0.1, 0.01],
    "rate_decay": [True, False],
    "batch_tam":  [64],
    "reg":        [0.0, 0.01],
}


# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

class RL_OvR():

    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64, reg=0.01):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.reg = reg
        self.clases = None
        self.clasificadores = None

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        self.clases = np.unique(y)
        self.clasificadores = []
        
        for c in self.clases:
            y_binaria = np.where(y == c, 1, 0)
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
        
        probabilidades = np.column_stack(
            [clf.clasifica_prob(ejemplos) for clf in self.clasificadores])
        indices_ganadores = np.argmax(probabilidades, axis=1)
        
        return self.clases[indices_ganadores]


# =====================================================
# EJERCICIO 7: APLICANDO LOS CLASIFICADORES MULTICLASE
# =====================================================

def codifica_one_hot(X):
    columnas_codificadas = []
    
    for j in range(X.shape[1]):
        columna = X[:, j]
        categorias = np.unique(columna)
        bloque = (columna[:, None] == categorias[None, :]).astype(float)
        columnas_codificadas.append(bloque)
    
    return np.hstack(columnas_codificadas)


def ajusta_OvR(X_entr, y_entr, X_val, y_val, rejilla, n_epochs=100, traza=False):
    mejores_params = None
    mejor_rend_val = -1.0
    resultados = []
    
    for params in _combinaciones_rejilla(rejilla):
        modelo = RL_OvR(**params)
        modelo.entrena(X_entr, y_entr, n_epochs=n_epochs)
        rend = rendimiento(modelo, X_val, y_val)
        resultados.append((params, rend))
        
        if traza:
            print(f"   {params} -> validación: {rend:.4f}")
        
        if rend > mejor_rend_val:
            mejor_rend_val = rend
            mejores_params = params
    
    resultados.sort(key=lambda t: t[1], reverse=True)
    
    return mejores_params, mejor_rend_val, resultados


REJILLA_OVR = {
    "rate":       [0.1, 0.01],
    "rate_decay": [True, False],
    "batch_tam":  [32, 64],
    "reg":        [0.0, 0.01],
}

ALTO_DIGITO = 28
ANCHO_DIGITO = 28


def carga_imagenes_digitos(ruta_imagenes, alto=ALTO_DIGITO, ancho=ANCHO_DIGITO):
    with open(ruta_imagenes) as f:
        lineas = [linea.rstrip("\n").rstrip("\r") for linea in f]
    
    imagenes = []
    
    for inicio in range(0, len(lineas), alto):
        bloque = lineas[inicio:inicio + alto]
        if len(bloque) < alto:
            break
        filas_pixeles = []
        
        for fila in bloque:
            fila = fila.ljust(ancho)[:ancho]
            filas_pixeles.append([0 if c == " " else 1 for c in fila])
        imagenes.append(np.array(filas_pixeles, dtype=float).flatten())
    
    return np.array(imagenes)


def carga_etiquetas_digitos(ruta_etiquetas):
    with open(ruta_etiquetas) as f:
        return np.array([int(linea.strip()) for linea in f if linea.strip() != ""])


RUTA_DIGITOS = "datos/digitdata/"
X_entr_dg = carga_imagenes_digitos(RUTA_DIGITOS + "trainingimages")
y_entr_dg = carga_etiquetas_digitos(RUTA_DIGITOS + "traininglabels")
X_val_dg  = carga_imagenes_digitos(RUTA_DIGITOS + "validationimages")
y_val_dg  = carga_etiquetas_digitos(RUTA_DIGITOS + "validationlabels")
X_test_dg = carga_imagenes_digitos(RUTA_DIGITOS + "testimages")
y_test_dg = carga_etiquetas_digitos(RUTA_DIGITOS + "testlabels")



# ==================
# EJEMPLOS DE PRUEBA
# ==================


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

print("==== MEJOR RENDIMIENTO RL SOBRE VOTOS:")
RL_VOTOS=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=False,batch_tam=32,reg=0.0)
RL_VOTOS.entrena(Xe_votos,ye_votos)
print("Rendimiento RL entrenamiento sobre votos: ",rendimiento(RL_VOTOS,Xe_votos,ye_votos))
print("Rendimiento RL test sobre votos: ",rendimiento(RL_VOTOS,Xp_votos,yp_votos))
print("\n")

print("==== MEJOR RENDIMIENTO RL SOBRE CÁNCER:")
RL_CANCER=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=False,batch_tam=32,reg=0.0)
RL_CANCER.entrena(Xe_cancer,ye_cancer)
print("Rendimiento RL entrenamiento sobre cáncer: ",rendimiento(RL_CANCER,Xe_cancer,ye_cancer))
print("Rendimiento RL test sobre cancer: ",rendimiento(RL_CANCER,Xp_cancer,yp_cancer))
print("\n")

print("==== MEJOR RENDIMIENTO RL_OvR SOBRE CREDITO:")
X_credito_oh=codifica_one_hot(X_credito)
Xe_credito_oh,Xp_credito_oh,ye_credito,yp_credito=particion_entr_prueba(X_credito_oh,y_credito,test=0.3)

RL_CLASIF_CREDITO=RL_OvR(rate=0.1,rate_decay=False,batch_tam=32,reg=0.01)
RL_CLASIF_CREDITO.entrena(Xe_credito_oh,ye_credito)
print("Rendimiento RLOVR  entrenamiento sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xe_credito_oh,ye_credito))
print("Rendimiento RLOVR  test sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xp_credito_oh,yp_credito))
print("\n")

print("==== MEJOR RENDIMIENTO RL SOBRE IMDB:")
RL_IMDB=RegresionLogisticaMiniBatch(rate=0.01,rate_decay=False,batch_tam=64,reg=0.0)
RL_IMDB.entrena(X_train_imdb,y_train_imdb)
print("Rendimiento RL entrenamiento sobre imdb: ",rendimiento(RL_IMDB,X_train_imdb,y_train_imdb))
print("Rendimiento RL test sobre imdb: ",rendimiento(RL_IMDB,X_test_imdb,y_test_imdb))
print("\n")

print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
RL_DG=RL_OvR(rate=0.1,rate_decay=False,batch_tam=32,reg=0.0)
RL_DG.entrena(X_entr_dg,y_entr_dg)
print("Rendimiento RL entrenamiento sobre dígitos: ",rendimiento(RL_DG,X_entr_dg,y_entr_dg))
print("Rendimiento RL validación sobre dígitos: ",rendimiento(RL_DG,X_val_dg,y_val_dg))
print("Rendimiento RL test sobre dígitos: ",rendimiento(RL_DG,X_test_dg,y_test_dg))
