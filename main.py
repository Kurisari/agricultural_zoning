import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Cargar la matriz desde el archivo de texto
def cargar_matriz_desde_txt(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lines = archivo.readlines()
        filas, columnas = map(int, lines[0].split())  # Leer las dimensiones de la matriz
        matriz = np.array([list(map(float, linea.split())) for linea in lines[1:]])
        print(matriz)   
    return matriz


# Calcular homogeneidad de las zonas
def calcular_homogeneidad(matriz, zonas, sigma_T):
    homogeneidad = 0
    for zona in zonas:
        valores = [matriz[x, y] for x, y in zona]  # Acceder a los valores de la matriz usando las coordenadas de las zonas
        media = np.mean(valores)
        varianza = np.var(valores)
        homogeneidad += (1 - (varianza / sigma_T)) * len(zona)
    return homogeneidad / len(matriz)


# Función objetivo para evaluar la solución
def funcion_objetivo(matriz, zonas, sigma_T, alpha):
    h = calcular_homogeneidad(matriz, zonas, sigma_T)
    if h >= alpha:
        return 0  # Si cumple el umbral de homogeneidad
    return 1 - h  # Penalizar la falta de homogeneidad

# Generar vecindad asegurando el formato correcto
def generar_vecindad(zonas, matriz_shape):
    nuevas_zonas = []
    for zona in zonas:
        nueva_zona = zona.copy()
        if random.random() > 0.5:  # Aleatoriamente añadir o quitar celdas
            x, y = random.randint(0, matriz_shape[0] - 1), random.randint(0, matriz_shape[1] - 1)
            if (x, y) not in nueva_zona:
                nueva_zona.append((x, y))
        elif len(nueva_zona) > 1:
            nueva_zona.pop(random.randint(0, len(nueva_zona) - 1))  # Quitar una celda aleatoria
        nuevas_zonas.append(nueva_zona)
    return nuevas_zonas

# Búsqueda Tabú
def busqueda_tabu(matriz, num_iteraciones, lista_tabu_max, sigma_T, alpha):
    # Inicializar solución inicial
    zonas = [[(i, j)] for i in range(matriz.shape[0]) for j in range(matriz.shape[1])]  # Cada celda es su propia zona
    mejor_solucion = zonas
    mejor_valor = funcion_objetivo(matriz, mejor_solucion, sigma_T, alpha)
    lista_tabu = []

    for _ in range(num_iteraciones):
        vecindad = generar_vecindad(zonas, matriz.shape)
        mejor_vecino = min(vecindad, key=lambda z: funcion_objetivo(matriz, z, sigma_T, alpha))

        # Evaluar la mejor solución de la vecindad
        if mejor_vecino not in lista_tabu:
            zonas = mejor_vecino
            valor = funcion_objetivo(matriz, zonas, sigma_T, alpha)

            if valor < mejor_valor:
                mejor_solucion = zonas
                mejor_valor = valor

            lista_tabu.append(mejor_vecino)
            if len(lista_tabu) > lista_tabu_max:
                lista_tabu.pop(0)

    return mejor_solucion, mejor_valor

# Visualización de la matriz y las zonas
def plot_zonas(matriz, zonas):
    plt.imshow(matriz, cmap='viridis', interpolation='nearest')
    for zona in zonas:
        for (x, y) in zona:
            plt.text(y, x, f'X', ha='center', va='center', color='red', fontsize=8)
    plt.colorbar(label='Valores de MO')
    plt.title("Zonificación agrícola")
    plt.show()

# Parámetros y ejecución
ruta_archivo = r"./zonificacion_agricola/Reales/MO.txt" # Ruta al archivo con los datos
matriz = cargar_matriz_desde_txt(ruta_archivo)


num_iteraciones = 100
lista_tabu_max = 10
sigma_T = 5  # Ajustar según los datos
alpha = 0.7  # Umbral de homogeneidad

mejor_solucion, mejor_valor = busqueda_tabu(matriz, num_iteraciones, lista_tabu_max, sigma_T, alpha)

# Visualizar resultado
print("Mejor valor de la función objetivo:", mejor_valor)
plot_zonas(matriz, mejor_solucion)