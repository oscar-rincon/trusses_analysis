# -*- coding: utf-8 -*-
"""
Funciones para analizar mapa de elevación.
"""

import numpy as np
import matplotlib.pyplot as plt

def cargar_datos_elevacion(xs_path, ys_path, zs_path):
    """
    Carga los datos de elevación desde archivos de texto y muestra el tamaño de las matrices.

    Args:
        xs_path (str): Ruta al archivo de texto que contiene los datos de xs.
        ys_path (str): Ruta al archivo de texto que contiene los datos de ys.
        zs_path (str): Ruta al archivo de texto que contiene los datos de zs.

    Returns:
        tuple: Una tupla que contiene las matrices xs, ys y zs.
    """
    #-------------------------------------------------------------------
    # TODO: Cargar xs, ys y zs desde los archivos de texto usando NumPy
    xs = # <-- Completar aquí
    ys = # <-- Completar aquí
    zs = # <-- Completar aquí
    #-------------------------------------------------------------------
    
    # Mostrar el tamaño de las matrices
    print(xs.shape, ys.shape, zs.shape)

    return xs, ys, zs


def graficar_elevacion(xs, ys, zs):
    """
    Crea una figura con dos subplots: una gráfica 3D y un mapa de calor de los datos de elevación.

    Args:
        xs (numpy.ndarray): Matriz de coordenadas x.
        ys (numpy.ndarray): Matriz de coordenadas y.
        zs (numpy.ndarray): Matriz de elevaciones.

    Returns:
        None
    """
    # Crear la figura y los subplots
    #-------------------------------------------------------------------
    # TODO: Ajustar el tamaño de la figura
    fig = plt.figure(figsize=(,)) # <-- Completar aquí
    #-------------------------------------------------------------------

    # Subplot 1: Gráfica 3D
    ax1 = fig.add_subplot(121, projection='3d')
    #-------------------------------------------------------------------
    # TODO: Crear una gráfica 3D con los datos de elevación
    #-------------------------------------------------------------------   
    surf = ax1.plot_surface( , , ,  ) # <-- Completar aquí
    fig.colorbar(surf, ax=ax1, label='Altitud (km)')
    ax1.set_xlabel( ) # <-- Completar aquí
    ax1.set_ylabel( ) # <-- Completar aquí
    ax1.set_title( ) # <-- Completar aquí
    plt.tight_layout()

    # Subplot 2: Mapa de calor
    ax2 = fig.add_subplot(122)
    
    #-------------------------------------------------------------------
    # TODO: Crear un mapa de calor con los datos de elevación
    heatmap = ax2.imshow( ,  ,  ,  ) # <-- Completar aquí
    #-------------------------------------------------------------------
    fig.colorbar(heatmap, ax=ax2, label='Altitud (km)')
    #-------------------------------------------------------------------
    # TODO: Incorporar etiquetas y título
    # <Completar aquí>
    # <Completar aquí>
    # <Completar aquí>
    #-------------------------------------------------------------------

    # TODO: Ajustar el diseño de la figura
    plt.tight_layout()
    
    plt.show()
    return None


def calc_gradiente(zs, xs, ys):
    """
    Calcula los gradientes numéricos de un campo escalar (elevaciones) usando el método de diferencias finitas.

    Parameters:
    zs (np.ndarray): Una matriz 2D que representa el campo escalar (valores de elevación).
    xs (numpy.ndarray): Matriz de coordenadas x.
    ys (numpy.ndarray): Matriz de coordenadas y.

    Returns:
    tuple: Dos matrices 2D que representan el gradiente de z con respecto a x (dzdx) 
           y y (dzdy) (es decir, ∂z/∂x y ∂z/∂y).
    """
    # Tamaño del paso espacial en x y en y
    dx = xs[1, 0] - xs[0, 0]  # Se asume que es constante en x
    dy = ys[0, 1] - ys[0, 0]  # Se asume que es constante en y  
    
    rows, cols = zs.shape
    
    # Inicializar matrices de gradiente
    dzdx = np.zeros_like(zs)
    dzdy = np.zeros_like(zs)
    
    # Calcular el gradiente en la dirección x
    for i in range(rows):
        for j in range(1, cols - 1):
            #-------------------------------------------------------------------
            # TODO: Implementar cálculo de dzdx para puntos internos con diferencias central 
            # <Completar aquí>
            #-------------------------------------------------------------------            
            
    # Bordes para dzdx
    #-------------------------------------------------------------------
    # TODO: Implementar cálculo de dzdx para bodes con diferencias hacia adelante y hacia atrás
    # <Completar aquí>
    # <Completar aquí>
    #-------------------------------------------------------------------       
    
    # Calcular el gradiente en la dirección y
    for i in range(1, rows - 1):
        for j in range(cols):
            #-------------------------------------------------------------------
            # TODO: Implementar cálculo de dzdy para puntos internos con diferencias central 
            # <Completar aquí>
            #-------------------------------------------------------------------            
            
    # Bordes para dzdy
    #-------------------------------------------------------------------
    # TODO: Implementar cálculo de dzdx para bodes con diferencias hacia adelante y hacia atrás
    # <Completar aquí>
    # <Completar aquí>
    #-------------------------------------------------------------------     
    
    return dzdx, dzdy

def graficar_inclinacion(xs, ys, dzdx, dzdy):
    """
    Calcula y grafica el ángulo de inclinación del terreno en grados, categorizado en diferentes rangos.

    Args:
        xs (numpy.ndarray): Matriz de coordenadas x.
        ys (numpy.ndarray): Matriz de coordenadas y.
        dzdx (numpy.ndarray): Gradiente de la elevación en la dirección x.
        dzdy (numpy.ndarray): Gradiente de la elevación en la dirección y.

    Returns:
        None
    """
    #-------------------------------------------------------------------
    # TODO: Calcular la magnitud del gradiente con la función np.hypot   
    grad_magnitude = # <Completar aquí>
    #------------------------------------------------------------------- 

    #-------------------------------------------------------------------
    # TODO: Implementar el cálculo de theta con la función np.arctan
    theta = # <Completar aquí>
    #-------------------------------------------------------------------

    #-------------------------------------------------------------------
    # TODO: Convertir el ángulo de inclinación a grados con la función np.degrees
    theta_degrees = # <Completar aquí>
    #-------------------------------------------------------------------

    #-------------------------------------------------------------------
    # TODO: Definir los límites de las categorías de inclinación
    bounds =  # <Completar aquí>
    #-------------------------------------------------------------------
    cmap = plt.get_cmap( , len( ) - 1) # <Completar aquí>

    # Graficar el ángulo de inclinación con las categorías
    #-------------------------------------------------------------------
    # TODO: Graficar el ángulo de inclinación con las categorías    
    im = plt.imshow( , extent=( , ,  ,  ), origin='lower', cmap= , vmin= , vmax= ) # <Completar aquí>
    #-------------------------------------------------------------------
    
    # Añadir una barra de color con las categorías
    cbar = plt.colorbar( , ticks= ) # <Completar aquí>
    cbar.set_label( ) # <Completar aquí>

    #-------------------------------------------------------------------
    # TODO: Incorporar etiquetas y título
    # <Completar aquí>
    # <Completar aquí>
    # <Completar aquí>
    #-------------------------------------------------------------------

    # Mostrar la gráfica
    plt.show()
