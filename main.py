import os
import numpy as np
import random
import warnings
from skimage import measure
from scipy.ndimage import label
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from collections import deque
import pandas as pd

def read_txt_files(folder_path):
    """
    Lee todos los archivos .txt de una carpeta y devuelve una lista de matrices con sus nombres de archivo.
    """
    matrices, filenames = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    rows, cols = map(int, lines[0].split())
                    data = [float(val) for line in lines[1:] for val in line.split()]
                    matrices.append(np.array(data).reshape(rows, cols))
                    filenames.append(filename)
            except Exception as e:
                print(f"Error al leer {filename}: {e}")
    return matrices, filenames

def contar_subdivisiones(matriz):
    """
    Cuenta el número de subdivisiones distintas (regiones conexas) en la matriz,
    considerando que las subdivisiones están formadas por valores iguales y son
    ortogonalmente conexas.
    """
    labeled_array, num_features = measure.label(matriz, connectivity=1, return_num=True)
    return num_features

def inicializar_solucion_espectral(grid, n_clusters):
    """
    Inicializa la solución utilizando agrupamiento espectral basado en los valores de las celdas.
    """
    reshaped_grid = grid.flatten().reshape(-1, 1)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    labels = spectral.fit_predict(reshaped_grid)
    return labels.reshape(grid.shape)

def calcular_homogeneidad(grid, region, total_variance, total_size):
    """
    Calcula la homogeneidad de una región utilizando la fórmula de la imagen.
    """
    region_size = len(region)
    region_variance = np.var(region)

    if region_size > 1:  # Evitar divisiones por 0
        h_q_s = (1 - ((region_size - 1) * region_variance) / (total_variance * (total_size - region_size)))
    else:
        h_q_s = 0  # Homogeneidad máxima si solo hay un elemento

    return h_q_s

def calcular_costo(solucion, grid, alpha, penalizacion_subdivisiones=0.1):
    """
    Calcula el costo de una solución basado en la fórmula actualizada de homogeneidad.
    """
    regiones = np.unique(solucion)
    total_variance = np.var(grid)
    total_size = grid.size
    costo = 0

    for region in regiones:
        valores_region = grid[solucion == region]
        h_q_s = calcular_homogeneidad(grid, valores_region, total_variance, total_size)

        if h_q_s < alpha:
            return float('inf')  # Penalización para soluciones no válidas
        costo += (1 - h_q_s)  # Coste proporcional a la falta de homogeneidad

    subdivisiones = contar_subdivisiones(solucion)
    costo += penalizacion_subdivisiones * subdivisiones
    return costo

def obtener_vecindad(solucion, grid, suavizado_factor=0.5):
    """
    Genera vecinos basados en la similitud de los valores de las celdas, considerando vecinos ortogonales.
    """
    vecinos = []
    filas, columnas = solucion.shape
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(10):  # Generar 10 vecinos aleatorios
        vecino = solucion.copy()
        f, c = random.randint(0, filas - 1), random.randint(0, columnas - 1)
        vecinos_cercanos = []

        for df, dc in direcciones:
            nf, nc = f + df, c + dc
            if 0 <= nf < filas and 0 <= nc < columnas:
                diferencia = abs(grid[f, c] - grid[nf, nc])
                vecinos_cercanos.append(diferencia)

        if vecinos_cercanos:
            umbral_similitud = max(np.mean(vecinos_cercanos) * suavizado_factor, 0.1)
            vecinos_regiones = {
                vecino[nf, nc]
                for df, dc in direcciones
                if 0 <= (nf := f + df) < filas and 0 <= (nc := c + dc) < columnas and abs(grid[f, c] - grid[nf, nc]) <= umbral_similitud
            }
            if vecinos_regiones:
                vecino[f, c] = random.choice(list(vecinos_regiones))

        vecinos.append(vecino)
    return vecinos

def busqueda_tabu(grid, alpha, max_iter=10000, tabu_size=60, n_clusters=4):
    """
    Implementa la búsqueda tabú para encontrar una zonificación agrícola óptima.
    """
    solucion_actual = inicializar_solucion_espectral(grid, n_clusters)
    mejor_solucion = solucion_actual.copy()
    mejor_costo = calcular_costo(mejor_solucion, grid, alpha)
    tabu_list = deque(maxlen=tabu_size)

    for iteracion in range(max_iter):
        vecinos = obtener_vecindad(solucion_actual, grid)
        mejor_costo_vecino = float('inf')
        mejor_vecino = None

        for vecino in vecinos:
            if vecino.tobytes() not in tabu_list:
                costo_vecino = calcular_costo(vecino, grid, alpha)
                if costo_vecino < mejor_costo_vecino:
                    mejor_costo_vecino = costo_vecino
                    mejor_vecino = vecino

        if mejor_vecino is not None:
            solucion_actual = mejor_vecino
            tabu_list.append(solucion_actual.tobytes())
            if mejor_costo_vecino < mejor_costo:
                mejor_solucion = solucion_actual.copy()
                mejor_costo = mejor_costo_vecino

    return mejor_solucion, mejor_costo

def plot_solution(grid, solution, filename, alpha):
    """
    Genera dos gráficos de las subdivisiones en la matriz: uno con colores difuminados y otro sin difuminado,
    pero con el mismo esquema de colores.
    """
    vmin, vmax = np.min(solution), np.max(solution)
    # Definimos un mapa de colores personalizado
    cmap_diffuse = LinearSegmentedColormap.from_list("custom", ["blue", "cyan", "lime", "yellow", "orange", "red"])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Crear directorio para guardar imágenes por alpha
    alpha_dir = f"zonificacion_alpha{alpha}"
    if not os.path.exists(alpha_dir):
        os.makedirs(alpha_dir)

    # Imagen con difuminado (bilinear)
    plt.figure(figsize=(8, 6))
    img = plt.imshow(solution, cmap=cmap_diffuse, interpolation='bilinear', norm=norm)
    contours = measure.find_contours(solution, level=vmin - 0.5)

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color='black', linewidth=2)

    filas, columnas = solution.shape
    for f in range(filas):
        for c in range(columnas):
            plt.text(c, f, f"{solution[f, c]:.2f}", color="black", ha="center", va="center", fontsize=8)

    plt.title(f"Mapa de la Parcela: {filename} | alpha: {alpha}")
    plt.xlabel("Columnas")
    plt.ylabel("Filas")
    cbar = plt.colorbar(img, label="Valor")
    cbar.ax.set_ylabel('Valor')

    # Guardar imagen con difuminado
    image_filename_diffuse = os.path.join(alpha_dir, f"reales_diffuse_{filename.replace('.txt', f'_alpha{alpha}.png')}")
    plt.savefig(image_filename_diffuse)
    plt.close()

    # Imagen sin difuminado (nearest)
    plt.figure(figsize=(8, 6))
    img_no_diffuse = plt.imshow(solution, cmap=cmap_diffuse, interpolation='nearest', norm=norm)
    contours_no_diffuse = measure.find_contours(solution, level=vmin - 0.5)

    for contour in contours_no_diffuse:
        plt.plot(contour[:, 1], contour[:, 0], color='black', linewidth=2)

    for f in range(filas):
        for c in range(columnas):
            plt.text(c, f, f"{solution[f, c]:.2f}", color="black", ha="center", va="center", fontsize=8)

    plt.title(f"Mapa de la Parcela: {filename} | alpha: {alpha}")
    plt.xlabel("Columnas")
    plt.ylabel("Filas")
    cbar_no_diffuse = plt.colorbar(img_no_diffuse, label="Valor")
    cbar_no_diffuse.ax.set_ylabel('Valor')

    # Guardar imagen sin difuminado
    image_filename_no_diffuse = os.path.join(alpha_dir, f"reales_no_diffuse_{filename.replace('.txt', f'_alpha{alpha}.png')}")
    plt.savefig(image_filename_no_diffuse)
    plt.close()

    return image_filename_diffuse, image_filename_no_diffuse

def guardar_resultados_en_excel(results, best_results, alpha):
    with pd.ExcelWriter(f"resultados_reales_alpha{alpha}.xlsx") as writer:
        # Guardar las 30 iteraciones por cada archivo
        for filename, data in results.items():
            df_results = pd.DataFrame(data['results'])
            df_results.to_excel(writer, sheet_name=filename, index=False)

        # Guardar los mejores resultados en una hoja separada
        df_best = pd.DataFrame(best_results)
        df_best.to_excel(writer, sheet_name='Mejores_Resultados', index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Graph is not fully connected")

    folder_path = r'./zonificacion_agricola/Reales/'
    matrices, filenames = read_txt_files(folder_path)
    
    alpha_values = [0.5, 0.7, 0.9]
    all_results = {}
    best_results = []

    # Ejecutar el proceso 30 veces por cada alpha
    for alpha in alpha_values:
        # Limpiar best_results al inicio de cada alpha
        best_results = []
        
        for i in range(5):  # Ejecutar 30 veces para cada alpha
            for j, (grid, filename) in enumerate(zip(matrices, filenames)):
                print(f"Procesando archivo: {filename} | Iteración: {i+1} | alpha: {alpha}")
                mejor_solucion, mejor_costo = busqueda_tabu(grid, alpha, max_iter=10000, tabu_size=60, n_clusters=4)
                subdivisiones = contar_subdivisiones(mejor_solucion)

                # Guardar imagen de la mejor solución
                image_filename_diffuse, image_filename_no_diffuse = plot_solution(grid, mejor_solucion, filename, alpha)

                # Guardar los resultados
                if filename not in all_results:
                    all_results[filename] = {'results': []}

                all_results[filename]['results'].append({
                    'Iteración': i + 1,
                    'Costo': mejor_costo,
                    'Subdivisiones': subdivisiones,
                })

                # Almacenar el mejor resultado de cada archivo
                if i == 0 or mejor_costo < all_results[filename]['best_result']['Mejor Costo']:
                    all_results[filename]['best_result'] = {
                        'Mejor Costo': mejor_costo,
                        'Mejor Subdivisiones': subdivisiones,
                    }

        # Añadir los mejores resultados de todos los archivos al resumen
        for filename, data in all_results.items():
            best_results.append({
                'Archivo': filename,
                'Mejor Costo': data['best_result']['Mejor Costo'],
                'Mejor Subdivisiones': data['best_result']['Mejor Subdivisiones'],
            })

        # Guardar los resultados en un Excel
        guardar_resultados_en_excel(all_results, best_results, alpha)
