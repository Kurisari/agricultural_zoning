import numpy as np
import random
from collections import deque

def calculate_homogeneity(region_values, total_variance, total_size, alpha):
    region_size = len(region_values)
    if region_size == 1:  # Una sola celda es siempre homogénea
        return True
    region_variance = np.var(region_values, ddof=1)
    H = 1 - (region_size * region_variance) / (total_variance * total_size)
    return H >= alpha

def inicializar_solucion(grid):
    """Inicializa la solución asignando cada celda a su propia región."""
    filas, columnas = grid.shape
    return np.arange(filas * columnas).reshape(filas, columnas)

def obtener_vecindad(solucion, grid):
    """
    Genera vecinos al modificar asignaciones de celdas a regiones.
    """
    vecinos = []
    filas, columnas = solucion.shape
    for _ in range(10):  # Generamos 10 vecinos aleatorios
        vecino = solucion.copy()
        f, c = random.randint(0, filas - 1), random.randint(0, columnas - 1)
        vecinos_regiones = set()
        # Buscar vecinos ortogonales
        for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nf, nc = f + df, c + dc
            if 0 <= nf < filas and 0 <= nc < columnas:
                vecinos_regiones.add(vecino[nf, nc])
        if vecinos_regiones:
            vecino[f, c] = random.choice(list(vecinos_regiones))
        vecinos.append(vecino)
    return vecinos

def calcular_costo(solucion, grid, alpha):
    """
    Calcula el costo de una solución basado en el número de subregiones y su homogeneidad.
    """
    filas, columnas = grid.shape
    total_variance = np.var(grid, ddof=1)
    total_size = filas * columnas
    regiones = np.unique(solucion)
    costo = 0
    for region in regiones:
        indices = np.argwhere(solucion == region)
        valores_region = [grid[x, y] for x, y in indices]
        if not calculate_homogeneity(np.array(valores_region), total_variance, total_size, alpha):
            return float('inf')  # Penalización para soluciones no válidas
        costo += 1
    return costo

def busqueda_tabu(grid, alpha, max_iter=100, tabu_size=10):
    """
    Implementa la búsqueda tabú para encontrar una zonificación agrícola óptima.
    """
    solucion_actual = inicializar_solucion(grid)
    mejor_solucion = solucion_actual.copy()
    mejor_costo = calcular_costo(mejor_solucion, grid, alpha)

    tabu_list = deque(maxlen=tabu_size)
    iteracion = 0

    while iteracion < max_iter:
        vecinos = obtener_vecindad(solucion_actual, grid)
        mejores_vecinos = []
        mejor_costo_vecino = float('inf')

        for vecino in vecinos:
            if vecino.tobytes() not in tabu_list:
                costo_vecino = calcular_costo(vecino, grid, alpha)
                if costo_vecino < mejor_costo_vecino:
                    mejor_costo_vecino = costo_vecino
                    mejores_vecinos = [vecino]
                elif costo_vecino == mejor_costo_vecino:
                    mejores_vecinos.append(vecino)

        if mejores_vecinos:
            mejor_vecino = random.choice(mejores_vecinos)
            solucion_actual = mejor_vecino

            # Actualizar lista tabú
            tabu_list.append(solucion_actual.tobytes())

            # Actualizar mejor solución global si es necesario
            if mejor_costo_vecino < mejor_costo:
                mejor_solucion = solucion_actual.copy()
                mejor_costo = mejor_costo_vecino

        iteracion += 1

    return mejor_solucion, mejor_costo

# Ejemplo de uso
grid = np.array([[10.7, 15.4, 13.9, 16.1, 15.6, 16.6, 16.6], 
                 [11.7, 16.0, 13.8, 12.6, 14.4, 15.4, 11.2], 
                 [15.1, 12.6, 14.8, 16.8, 10.5, 18.7, 10.4], 
                 [16.3, 12.7, 14.2, 11.4, 11.5, 16.7, 13.5], 
                 [14.1, 11.1, 14.5, 15.0, 14.13, 9.6, 12.5], 
                 [11.8, 12.8, 14.9, 14.0, 11.2, 14.7, 14.7]])
alpha = 0.7  # Umbral de homogeneidad

mejor_solucion, mejor_costo = busqueda_tabu(grid, alpha, max_iter=100, tabu_size=10)

print("Mejor solución encontrada:")
print(mejor_solucion)
print("Costo de la solución:", mejor_costo)