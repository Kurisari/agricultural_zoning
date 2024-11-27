import numpy as np
import random
from scipy.ndimage import label

# Leer archivo y cargar la matriz
def cargar_datos(archivo):
    with open(archivo, 'r') as file:
        lines = file.readlines()
        dimensiones = tuple(map(int, lines[0].split()))
        matriz = np.array([list(map(float, line.split())) for line in lines[1:]])
    return dimensiones, matriz

# Función para etiquetar regiones conexas
def etiquetar_regiones(asignaciones):
    estructura = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])  # Adyacencia solo ortogonal (no diagonal)
    etiquetas, _ = label(asignaciones, structure=estructura)
    return etiquetas

# Verificar homogeneidad dentro de cada sub-región
def verificar_homogeneidad(matriz, etiquetas, umbral):
    regiones = np.unique(etiquetas)
    for region in regiones:
        if region == 0:  # Ignorar fondo
            continue
        valores = matriz[etiquetas == region]
        if max(valores) - min(valores) > umbral:
            return False
    return True

# Función de costo: número de sub-regiones válidas
def calcular_costo(matriz, etiquetas, umbral):
    if not verificar_homogeneidad(matriz, etiquetas, umbral):
        return float('inf')  # Penalizar soluciones no homogéneas
    return len(np.unique(etiquetas)) - 1  # Número de regiones (ignorar fondo)

# Generar una solución inicial conectada
def generar_solucion_inicial(matriz, umbral):
    filas, columnas = matriz.shape
    asignaciones = np.zeros_like(matriz, dtype=int)
    region_id = 1
    for i in range(filas):
        for j in range(columnas):
            if asignaciones[i, j] == 0:  # Si no está asignado
                # Expandir región usando una cola (BFS)
                cola = [(i, j)]
                while cola:
                    x, y = cola.pop(0)
                    if asignaciones[x, y] == 0:
                        asignaciones[x, y] = region_id
                        # Verificar vecinos adyacentes
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < filas and 0 <= ny < columnas:
                                if asignaciones[nx, ny] == 0 and abs(matriz[x, y] - matriz[nx, ny]) <= umbral:
                                    cola.append((nx, ny))
                region_id += 1
    return asignaciones

# Movimiento: intenta cambiar la región de una celda
def generar_vecino(matriz, etiquetas, umbral):
    vecino = etiquetas.copy()
    i, j = random.choice(list(np.ndindex(etiquetas.shape)))
    region_actual = etiquetas[i, j]
    
    # Cambiar a una región adyacente
    posibles_regiones = set()
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + dx, j + dy
        if 0 <= ni < matriz.shape[0] and 0 <= nj < matriz.shape[1]:
            posibles_regiones.add(etiquetas[ni, nj])
    
    posibles_regiones.discard(region_actual)
    if posibles_regiones:
        vecino[i, j] = random.choice(list(posibles_regiones))
    
    # Reetiquetar regiones después de modificar
    return etiquetar_regiones(vecino)

# Búsqueda tabú
def busqueda_tabu(matriz, umbral, iteraciones=1000, tamaño_lista_tabu=50):
    solucion_actual = generar_solucion_inicial(matriz, umbral)
    mejor_solucion = solucion_actual
    mejor_costo = calcular_costo(matriz, mejor_solucion, umbral)
    
    lista_tabu = []
    for _ in range(iteraciones):
        vecinos = [generar_vecino(matriz, solucion_actual, umbral) for _ in range(10)]
        vecino_costos = [(vecino, calcular_costo(matriz, vecino, umbral)) for vecino in vecinos]
        vecino_costos = sorted(vecino_costos, key=lambda x: x[1])  # Ordenar por costo
        
        for vecino, costo in vecino_costos:
            if vecino.tolist() not in lista_tabu:
                solucion_actual = vecino
                if costo < mejor_costo:
                    mejor_solucion = vecino
                    mejor_costo = costo
                lista_tabu.append(vecino.tolist())
                if len(lista_tabu) > tamaño_lista_tabu:
                    lista_tabu.pop(0)
                break

    return mejor_solucion, mejor_costo

# Parámetros
archivo = "MO.txt"
umbral = 3.0  # Diferencia máxima permitida dentro de una sub-región
dimensiones, matriz = cargar_datos(archivo)

# Ejecutar búsqueda tabú
mejor_solucion, mejor_costo = busqueda_tabu(matriz, umbral)
print("Mejor solución encontrada:")
print(mejor_solucion)
print("Costo de la mejor solución:", mejor_costo)