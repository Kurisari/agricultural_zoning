import numpy as np
import random
from collections import deque
import os

def read_txt_files(folder_path):
    """
    Lee todos los archivos .txt de una carpeta y devuelve una lista de matrices con sus nombres de archivo.
    """
    matrices = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                # Leer las dimensiones de la matriz
                rows, cols = map(int, lines[0].split())
                # Leer los valores numéricos
                data = [float(value) for line in lines[1:] for value in line.split()]
                # Convertir a numpy array con la forma especificada
                grid = np.array(data).reshape(rows, cols)
                matrices.append(grid)
                filenames.append(filename)
    return matrices, filenames

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

def busqueda_tabu(grid, alpha, max_iter=1000, tabu_size=15):
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

os.system('cls')

# Ruta de la carpeta con los archivos .txt
folder_path = r'C:\Users\geome\OneDrive\Documentos\SEMESTRE 5\OPTIMIZACION Y META 1\PROYECTO FINAL\zonificacion_agricola\Reales'

# Leer matrices y sus nombres desde los archivos .txt
matrices, filenames = read_txt_files(folder_path)

alpha1 = 0.5  # Umbral de homogeneidad
alpha2 = 0.7  # Umbral de homogeneidad
alpha3 = 0.9  # Umbral de homogeneidad

print(f"\nUmbral de homogeneidad: {alpha1}\n")
# Ejecutar búsqueda tabú para cada matriz
for i, (grid, filename) in enumerate(zip(matrices, filenames)):
    print(f"Procesando archivo: {filename}")
    mejor_solucion, mejor_costo = busqueda_tabu(grid, alpha1, max_iter=1000, tabu_size=15)
    print("Mejor solución encontrada:")
    print(mejor_solucion)
    print("\nCosto de la solución:", mejor_costo)
    print("-" * 50)

print(f"\nUmbral de homogeneidad: {alpha2}\n")
# Ejecutar búsqueda tabú para cada matriz
for i, (grid, filename) in enumerate(zip(matrices, filenames)):

    print(f"Procesando archivo: {filename}")
    mejor_solucion, mejor_costo = busqueda_tabu(grid, alpha2, max_iter=1000, tabu_size=15)
    print("Mejor solución encontrada:")
    print(mejor_solucion)
    print("\nCosto de la solución:", mejor_costo)
    print("-" * 50)
    
print(f"\nUmbral de homogeneidad: {alpha3}\n")
# Ejecutar búsqueda tabú para cada matriz
for i, (grid, filename) in enumerate(zip(matrices, filenames)):
    print(f"Procesando archivo: {filename}")
    mejor_solucion, mejor_costo = busqueda_tabu(grid, alpha3, max_iter=1000, tabu_size=15)
    print("Mejor solución encontrada:")
    print(mejor_solucion)
    print("\nCosto de la solución:", mejor_costo)
    print("-" * 50)
