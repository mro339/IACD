# =====================================================
# EJERCICIO 7: APLICANDO LOS CLASIFICADORES MULTICLASE
# =====================================================

# ----------------------------------------------------------------------------
# REFERENCIAS EN EL MATERIAL DE CLASE
# ----------------------------------------------------------------------------
# Teoría — tema06preprocesadodedatos.pdf:
#   - Diapositiva 15: "Datos categóricos" -> entre los ajustes posibles está la
#                     "Codificación One-Hot".
#   - Diapositivas 20-21: ejemplo de codificación One-Hot. Una columna nominal
#                         (p.ej. Color con valores rosa/azul/rojo) se reemplaza
#                         por una columna binaria por cada valor posible. La
#                         diap. 21 avisa: "Estamos aumentando la dimensión".
#   - Diapositiva 25: por qué hace falta one-hot para modelos NO basados en
#                     árboles (como la regresión logística): la correspondencia
#                     numérica introduciría un orden artificial entre categorías.
#   - Diapositiva 2: el preprocesado forma parte de la "construcción del
#                    conjunto de datos (tabular)"; aquí también construimos a
#                    mano el dataset de dígitos a partir de ficheros de texto.
#
# Código de prácticas — tema-06.py:
#   - Líneas 438-468: uso de OneHotEncoder de scikit-learn. Aquí reproducimos
#                     ESA transformación a mano con numpy (en el trabajo NO se
#                     permite Scikit Learn ni Pandas para esta función).
# ----------------------------------------------------------------------------

# ============================================================================
# 7.1) CODIFICACIÓN ONE-HOT
# ============================================================================

def codifica_one_hot(X):
    """
    Codificación one-hot de un conjunto de datos X (array de numpy).    
    Se presupone que todos los atributos son categóricos.

    Cada columna se reemplaza por tantas columnas binarias (0/1) como valores
    distintos tenga, en orden.
    """
    columnas_codificadas = []

    # Se procesa cada columna por separado.
    for j in range(X.shape[1]):
        columna = X[:, j]

        # Categorías distintas de esta columna
        categorias = np.unique(columna) # ordenadas

        # columna[:, None] se convierte en columna
        bloque = (columna[:, None] == categorias[None, :]).astype(float) # astype convierte a 0.0/1.0
        columnas_codificadas.append(bloque)

    return np.hstack(columnas_codificadas) # Se concatenan horizontalmente los bloques de todas las columnas


# ============================================================================
# 7.2) CONJUNTO DE DATOS DE LA CONCESIÓN DE CRÉDITO
# ============================================================================

# Los atributos de X_credito son categóricos, así que no se pueden usar directamente con regresión logística.
# Hay que aplicar one-hot primero.
# Después se entrena un RL_OvR (3 clases: concesión, estudio, no concesión).
# Se ajustan hiperparámetros con validación.

# Rejilla de ejemplo
REJILLA_OVR = {
    "rate":       [0.1, 0.01],
    "rate_decay": [True, False],
    "batch_tam":  [32, 64],
    "reg":        [0.0, 0.01],
}

# --- Proceso para CRÉDITO ---

# # 1) Codificar one-hot los atributos categóricos.
X_credito_oh = codifica_one_hot(X_credito)
#
# # 2) Reservar prueba (holdout) y extraer validación del entrenamiento.
Xev, Xp, yev, yp = particion_entr_prueba(X_credito_oh, y_credito, test=0.2)
Xe, Xv, ye, yv  = particion_entr_prueba(Xev, yev, test=0.2)

# # 3) Ajuste de hiperparámetros sobre validación.
mejores_cr, rend_val_cr = ajusta_OvR(Xe, ye, Xv, yv, REJILLA_OVR, n_epochs=100)
print("Crédito - mejor combinación:", mejores_cr, "rend. validación:", rend_val_cr)

# # 4) Modelo final (entrenamiento+validación) y evaluación sobre prueba.
modelo_cr = RL_OvR(**mejores_cr)
modelo_cr.entrena(Xev, yev, n_epochs=100)
print("Crédito - rendimiento entrenamiento:", rendimiento(modelo_cr, Xev, yev))
print("Crédito - rendimiento PRUEBA:       ", rendimiento(modelo_cr, Xp, yp))

# ============================================================================
# 7.3) CLASIFICACIÓN DE IMÁGENES DE DÍGITOS ESCRITOS A MANO
# ============================================================================

# Cada imagen son 28x28 píxeles representados con caracteres:
#   ' ' (espacio) -> píxel blanco -> 0
#   '+' (borde) o '#' (interior) -> píxel negro -> 1  (no distinguimos borde/interior)
# Las imágenes vienen seguidas en un fichero (28 líneas por imagen) y las etiquetas en otro fichero aparte, en el mismo orden.

ALTO_DIGITO = 28   # nº de filas de píxeles por imagen
ANCHO_DIGITO = 28  # nº de columnas de píxeles por imagen


def carga_imagenes_digitos(ruta_imagenes, alto=ALTO_DIGITO, ancho=ANCHO_DIGITO):
    """
    Lee un fichero de imágenes de dígitos.
    Devuelve un array de numpy (n_imagenes, alto*ancho)
    Cada imagen aplanada en un vector de 0s y 1s.
    """
    with open(ruta_imagenes) as f:
        # No usamos strip() sobre cada línea: los espacios son píxeles blancos y no deben eliminarse. 
        # Solo quitamos el salto de línea final.
        lineas = [linea.rstrip("\n").rstrip("\r") for linea in f]

    imagenes = []
    # las imágenes vienen una detrás de otra sin separador, y cada imagen ocupa 'alto' líneas seguidas
    for inicio in range(0, len(lineas), alto):
        bloque = lineas[inicio:inicio + alto] # cada bloque es una imagen
        if len(bloque) < alto:
            break  # por si el fichero termina con líneas incompletas

        filas_pixeles = []
        for fila in bloque:
            # Rellenamos/recortamos cada línea a 'ancho' caracteres por seguridad
            # (algunas líneas pueden venir con espacios finales recortados).
            fila = fila.ljust(ancho)[:ancho]
            # ' ' -> 0 (blanco); '+' y '#' -> 1 (negro).
            filas_pixeles.append([0 if c == " " else 1 for c in fila])

        # Aplanamos la imagen 28x28 en un vector de 784 componentes.
        imagenes.append(np.array(filas_pixeles, dtype=float).flatten())

    return np.array(imagenes)


def carga_etiquetas_digitos(ruta_etiquetas):
    """
    Lee un fichero de etiquetas (un dígito por línea)
    Devuelve un array numpy de enteros.
    """
    with open(ruta_etiquetas) as f:
        return np.array([int(linea.strip()) for linea in f if linea.strip() != ""])


# --- Carga de las variables del dataset de dígitos

RUTA = "datos/digitdata/"
X_entr_dg = carga_imagenes_digitos(RUTA + "trainingimages")
y_entr_dg = carga_etiquetas_digitos(RUTA + "traininglabels")
X_val_dg  = carga_imagenes_digitos(RUTA + "validationimages")
y_val_dg  = carga_etiquetas_digitos(RUTA + "validationlabels")
X_test_dg = carga_imagenes_digitos(RUTA + "testimages")
y_test_dg = carga_etiquetas_digitos(RUTA + "testlabels")

# Ajuste de hiperparámetros usando el conjunto de validación YA dado:
mejores_dg, rend_val_dg = ajusta_OvR(X_entr_dg, y_entr_dg, X_val_dg, y_val_dg,
                                      REJILLA_OVR, n_epochs=100)
print("Dígitos - mejor combinación:", mejores_dg, "rend. validación:", rend_val_dg)

# Modelo final: entrenar con entrenamiento (+validación) y medir sobre test.
modelo_dg = RL_OvR(**mejores_dg)
modelo_dg.entrena(X_entr_dg, y_entr_dg, n_epochs=100)
print("Dígitos - rendimiento entrenamiento:", rendimiento(modelo_dg, X_entr_dg, y_entr_dg))
print("Dígitos - rendimiento validación:   ", rendimiento(modelo_dg, X_val_dg, y_val_dg))
print("Dígitos - rendimiento TEST:         ", rendimiento(modelo_dg, X_test_dg, y_test_dg))


# ============================================================================
# DESCRIPCIÓN DEL PROCESO Y RENDIMIENTOS
# ============================================================================

# CRÉDITO:
#   - Se aplica one-hot porque los atributos son categóricos (tema 6, diap. 25).
#   - Búsqueda en rejilla sobre validación (tema 9, diap. 20) y evaluación final
#     sobre prueba reservada (holdout; tema 4, diap. 2).
#   - Mejor combinación: rate=___, rate_decay=___, batch_tam=___, reg=___
#   - Rendimiento prueba: ___
#
# DÍGITOS:
#   - Cada imagen 28x28 se aplana en un vector binario de 784 píxewles.
#   - El dataset ya viene partido en entrenamiento/validación/prueba: se ajustan
#     hiperparámetros con validación y se da el rendimiento final sobre test.
#   - Mejor combinación: rate=___, rate_decay=___, batch_tam=___, reg=___
#   - Rendimiento test: ___  (objetivo: > 75%)