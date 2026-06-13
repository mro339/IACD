# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================
#
# Objetivo: usando la RegresionLogisticaMiniBatch (Ejercicio 4), obtener el mejor
# clasificador posible para VOTOS, CÁNCER e IMDB, ajustando los hiperparámetros
# (rate, rate_decay, batch_tam, reg) con un CONJUNTO DE VALIDACIÓN, y dando la
# valoración final sobre un CONJUNTO DE PRUEBA reservado.
#
# La implementación es GENERAL: la función de ajuste sirve para cualquier
# problema de clasificación binaria, no solo para estos tres datasets.
#
# ----------------------------------------------------------------------------
# REFERENCIAS EN EL MATERIAL DE CLASE
# ----------------------------------------------------------------------------
# Teoría — tema_09_Ajuste_de_modelos.pdf:
#   - Diapositivas 4-7: qué son los hiperparámetros (variables externas que se
#                       fijan ANTES del entrenamiento y no se aprenden de los
#                       datos) y por qué importan (equilibrio sesgo-varianza,
#                       riesgo de sobreajuste/subajuste). Justifica POR QUÉ hay
#                       que ajustarlos: rate, rate_decay, batch_tam, reg son
#                       hiperparámetros de la regresión logística.
#   - Diapositiva 10: planteamiento formal del problema -> buscar la
#                     configuración h que minimiza el error (o maximiza el
#                     acierto): min_{h in H} f(h). Aquí f(h) = error sobre
#                     validación de un modelo entrenado con la config. h.
#   - Diapositivas 12-13: definición del espacio/dominio de búsqueda y tipos de
#                         hiperparámetros (continuos como rate/reg, discretos
#                         como batch_tam, categóricos como rate_decay).
#   - Diapositiva 20: "Búsqueda en rejilla" (Grid Search), el método que
#                     implemento: se define una rejilla de valores y se exploran
#                     TODAS las combinaciones; garantiza la mejor del espacio
#                     definido (aunque es costoso si hay muchos HP).
#
# Teoría — tema4_Tecnicas_avanzadas_de_validacion.pdf:
#   - Diapositiva 2: predicciones Out-Of-Sample y método Holdout. Fundamenta
#                    reservar un conjunto de prueba (no usado en el ajuste) para
#                    medir de forma HONESTA cómo generalizará el modelo.
#
# Código de prácticas — tema_09.py:
#   - Líneas 133-143: división en entrenamiento/prueba; el de entrenamiento se
#                     usa para la optimización y el de prueba se RESERVA para la
#                     evaluación final. Aquí reproducimos esa idea (con un
#                     conjunto de validación explícito en lugar de CV, como pide
#                     el enunciado del trabajo).
#   - Líneas 161-204: definición de la rejilla de hiperparámetros (param_grid)
#                     y selección de la mejor combinación según el rendimiento
#                     (best_params_ / best_score_). Replico ese flujo a mano,
#                     ya que NO se permite usar Scikit Learn.
#
# Código de prácticas — tema-06.py:
#   - Líneas 715-717: el normalizador se AJUSTA solo con el entrenamiento y se
#                     aplica a validación/prueba sin reajustar (relevante para
#                     CÁNCER, cuyas características son continuas).
#
# Dependencias de OTROS ejercicios del mismo trabajo:
#   - particion_entr_prueba (Ej. 1): partición aleatoria y estratificada.
#   - rendimiento (Ej. 2): proporción de aciertos (accuracy) -> es nuestro f(h).
#   - NormalizadorStandard (Ej. 3): estandarización (solo para cáncer).
#   - RegresionLogisticaMiniBatch (Ej. 4): el clasificador a ajustar.
# ----------------------------------------------------------------------------

import numpy as np
from itertools import product


# ============================================================================
# 5.0) UTILIDADES GENERALES PARA EL AJUSTE DE HIPERPARÁMETROS
# ============================================================================

def _combinaciones_rejilla(rejilla):
    """ Genera la lista de todas las combinaciones de una rejilla de
        hiperparámetros.

        'rejilla' es un diccionario {nombre_hiperparametro: [valores...]}.

        Devuelve una lista de diccionarios, uno por combinación.
    """
    # ej: {"rate": [0.1, 0.01], "batch": [32, 64]}
    nombres = list(rejilla.keys()) #  ["rate", "batch"]
    listas_valores = [rejilla[n] for n in nombres] #  [[0.1, 0.01], [32, 64]]
    lista_combinaciones = [dict(zip(nombres, combo)) for combo in product(*listas_valores)] 
    # [{"rate": 0.1,  "batch": 32}, {"rate": 0.1,  "batch": 64}, {"rate": 0.01, "batch": 32}, {"rate": 0.01, "batch": 64}]

    return lista_combinaciones


def ajusta_RL(X_entr, y_entr, X_val, y_val, rejilla,
              n_epochs=100, traza=False):
    """
    Búsqueda en rejilla (Grid Search) sobre un conjunto de validación para la clase RegresionLogisticaMiniBatch.
    Generalizada a cualquier problema de clasificación binaria.

    Parámetros
      X_entr, y_entr : conjunto de entrenamiento.
      X_val,  y_val  : conjunto de validación
      rejilla        : dict {hiperparametro: [valores...]}.
      n_epochs       : nº de epochs para entrenar cada candidato.
      traza          : si True, imprime el rendimiento de cada combinación.

    Devuelve
      mejores_params : dict con la mejor combinación encontrada.
      mejor_rend_val : rendimiento (accuracy) de esa combinación en validación.
      resultados     : lista de (params, rendimiento_val) de todas las pruebas, ordenada de mejor a peor.
    """
    # Implementa min_{h in H} error(h)  <=>  max_{h in H} rendimiento(h)
    mejores_params = None
    mejor_rend_val = -1.0
    resultados = []

    for params in _combinaciones_rejilla(rejilla):
        # Cada 'params' es una configuración h del espacio de búsqueda H.
        # Se crea y entrena un modelo con esa configuración
        modelo = RegresionLogisticaMiniBatch(**params)
        modelo.entrena(X_entr, y_entr, n_epochs=n_epochs)

        # se evalúa f(h) sobre validación
        rend = rendimiento(modelo, X_val, y_val)
        resultados.append((params, rend))

        if traza:
            print(f"   {params} -> validación: {rend:.4f}")

        # Nos quedamos con la mejor combinación vista hasta el momento.
        if rend > mejor_rend_val:
            mejor_rend_val = rend
            mejores_params = params

    # Ordenamos de mejor a peor para poder documentar el ranking.
    resultados.sort(key=lambda t: t[1], reverse=True) # True = descendente
    return mejores_params, mejor_rend_val, resultados


def evalua_RL_completo(nombre, X, y, rejilla,
                       Xp=None, yp=None, test=0.2, val=0.2,
                       normalizar=False, n_epochs=100, traza=False):
    """
    Flujo completo de aplicación para un dataset binario:

      1) Reservar conjunto de prueba (holdout) si no viene dado.
      2) Separar un conjunto de validación del entrenamiento.
      3) (Opcional) normalizar: ajustar el normalizador solo con entrenamiento.
      4) Búsqueda en rejilla sobre validación -> mejores hiperparámetros.
      5) Reentrenar el modelo final con (entrenamiento + validación) y los
         mejores hiperparámetros, y medir su rendimiento sobre prueba.

    Si el dataset ya trae su propio conjunto de prueba (caso IMDB), se pasa en
    Xp, yp y entonces X, y se toman como el conjunto de entrenamiento completo.

    Devuelve un diccionario con el resumen del proceso.
    """
    print(f"\n===== {nombre} =====")

    # ---- 1) Conjunto de prueba (reservado para la evaluación final) ----------
    if Xp is None:
        # Partición aleatoria y estratificada (Ej. 1).
        Xev, Xp, yev, yp = particion_entr_prueba(X, y, test=test)
    else:
        # El test ya viene dado: X, y son el entrenamiento+validación completo.
        Xev, yev = X, y

    # ---- 2) Conjunto de validación (se extrae del entrenamiento) -------------
    Xe, Xv, ye, yv = particion_entr_prueba(Xev, yev, test=val)

    # ---- 3) Normalización (solo características continuas) ------
    # El normalizador se ajusta con el entrenamiento y se aplica a validación con esos mismos parámetros.
    if normalizar:
        norm = NormalizadorStandard()
        norm.ajusta(Xe)
        Xe = norm.normaliza(Xe)
        Xv = norm.normaliza(Xv)

    # ---- 4) Búsqueda en rejilla sobre validación -----------------------------
    print(" Buscando en rejilla sobre validación: ")
    mejores, rend_val, resultados = ajusta_RL(Xe, ye, Xv, yv, rejilla,
                                              n_epochs=n_epochs, traza=traza)
    print(f" Mejor combinación en validación: {mejores}")
    print(f" Rendimiento en validación: {rend_val:.4f}")

    # ---- 5) Modelo final: reentreno con entrenamiento+validación -------------
    # Una vez elegidos los hiperparámetros, entrenamos con todos los datos (entrenamiento + validación)
    # Medimos sobre el conjunto de prueba reservado
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
#
# Probamos rejillas pequeñas a modo de prueba ya que el coste crece como el producto de los tamaños.

# --- Rejilla (sirve de base para los tres datasets) -------------
REJILLA_RL = {
    "rate":       [0.1, 0.01],      # tasa de aprendizaje (continuo, diap. 13)
    "rate_decay": [True, False],    # ¿decae la tasa? (categórico/booleano)
    "batch_tam":  [32, 64],         # tamaño de mini-batch (discreto)
    "reg":        [0.0, 0.001, 0.01],  # constante de regularización L2 (continuo)
}
# 2 * 2 * 2 * 3 = 24 combinaciones.

# --- VOTOS (características categóricas/numéricas de rango pequeño) ----------
# No se normaliza: los votos están codificados en un rango ya homogéneo.
res_votos = evalua_RL_completo("VOTOS", X_votos, y_votos, REJILLA_RL,
                               test=0.2, val=0.2, normalizar=False,
                               n_epochs=100, traza=False)

# --- CÁNCER (características CONTINUAS -> SÍ se normaliza) -------------------
res_cancer = evalua_RL_completo("CÁNCER", X_cancer, y_cancer, REJILLA_RL,
                                test=0.2, val=0.2, normalizar=True,
                                n_epochs=100, traza=False)

#                                   C A M B I A R
##################################################################################################################
##################################################################################################################
# --- IMDB (vectores binarios; el test YA viene dado por el dataset) ---------
# Pasamos el test predefinido; la validación se extrae del train.
# Rejilla algo más reducida porque el dataset es mayor y entrena más lento.
# REJILLA_IMDB = {"rate": [0.1], "rate_decay": [True, False],
#                 "batch_tam": [64], "reg": [0.0, 0.01]}
res_imdb = evalua_RL_completo("IMDB", X_train_imdb, y_train_imdb, REJILLA_IMDB,
                              Xp=X_test_imdb, yp=y_test_imdb,
                              normalizar=False, n_epochs=100, traza=False)
##################################################################################################################
##################################################################################################################

# ============================================================================
# 5.2) DESCRIPCIÓN DEL PROCESO Y RENDIMIENTOS (rellenar tras EJECUTAR)
# ============================================================================

# PROCESO SEGUIDO (igual para los tres datasets):
#   1. Se reserva un 20% como conjunto de prueba (estratificado), que no se usa
#      en ninguna decisión de ajuste.

#      En IMDB el conjunto de prueba ya viene dado por el propio dataset.

#   2. Del resto se separa un 20% como conjunto de validación.

#   3. Para CÁNCER se estandarizan las características (NormalizadorStandard),
#      ajustando el normalizador solo con el entrenamiento. 
#       
#      Para VOTOS e IMDB no hace falta (rangos ya homogéneos: códigos de voto / vectores 0-1).

#   4. Se realiza una búsqueda en rejilla sobre {rate, rate_decay, batch_tam, reg}
#      eligiendo la combinación con mejor rendimiento (accuracy) en validación.

#   5. Con esa combinación se reentrena el modelo final usando entrenamiento+validación .
#      Se mide el rendimiento sobre el conjunto de prueba.


#                                   C A M B I A R
##################################################################################################################
##################################################################################################################
# RESULTADOS OBTENIDOS  (>>> COMPLETAR con los números reales tras ejecutar <<<
#  porque dependen de la implementación del Ej. 4 y de la partición aleatoria):

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
##################################################################################################################
##################################################################################################################


#                                   C A M B I A R
##################################################################################################################
##################################################################################################################
# OBSERVACIONES (ejemplo de lo que se puede comentar):
#   - En cáncer, sin normalizar el entrenamiento es inestable; al estandarizar,
#     el rendimiento mejora notablemente (las características tenían rangos muy
#     dispares: tema 6, diap. 43).
#   - Una pequeña regularización L2 (reg≈0.001-0.01) suele mejorar la
#     generalización frente a reg=0 (tema 9, diap. 7: control del sobreajuste).
#   - rate_decay=True estabiliza el descenso por el gradiente cuando la tasa
#     inicial es alta.
##################################################################################################################
##################################################################################################################