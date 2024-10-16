
# -*- coding: utf-8 -*-
"""
Funciones para analizar mapa de elevación.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Configuración de estilo para las gráficas
plt.style.use('fast')
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 1.5
mpl.rcParams.update(
    {
        'interactive': False,
        "text.usetex": False,  # Use mathtext, not LaTeX
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    }
)

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
    # Cargar xs, ys y zs desde los archivos de texto usando NumPy
    xs = np.loadtxt(xs_path)
    ys = np.loadtxt(ys_path)
    zs = np.loadtxt(zs_path)

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
    # Definir los límites de niveles ías de elevación
    bounds = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]

    # Crear la figura y los subplots
    fig = plt.figure(figsize=(10, 3))

    # Subplot 1: Gráfica 3D
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(xs, ys, zs, cmap='terrain')
    # Añadir cuadrícula discontinua
    ax1.grid(True, linestyle='--')

    fig.colorbar(surf, ax=ax1, label='Altitud [km]')
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [km]')
    ax1.set_title('3D plot - Elevation')
    plt.tight_layout()

    # Subplot 2: Mapa de calor   
    ax2 = fig.add_subplot(122)
    heatmap = ax2.imshow(zs.T, extent=(np.min(xs), np.max(xs), np.min(ys), np.max(ys)), origin='lower', cmap='terrain')
    plt.contour(xs, ys, zs, levels=bounds, colors='gray', linewidths=0.5)
    fig.colorbar(heatmap, ax=ax2, label='Altitud [km]')
    ax2.set_xlabel('x (km)')
    ax2.set_ylabel('y (km)')
    ax2.set_title('Heat Map - Elevation')
    plt.tight_layout()
    plt.show()

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
    dx = xs[1, 0] - xs[0, 0] # Se asume que es constante en x
    dy = ys[0, 1] - ys[0, 0] # Se asume que es constante en y  
    
    rows, cols = zs.shape
    
    # Inicializar matrices de gradiente
    dzdx = np.zeros_like(zs)
    dzdy = np.zeros_like(zs)
    
    # Calcular el gradiente en la dirección x
    for i in range(rows):
        for j in range(1, cols - 1):
            dzdx[i, j] = (zs[i, j + 1] - zs[i, j - 1]) / (2 * dx)  # Diferencia central
            
    # Bordes para dzdx
    dzdx[:, 0] = (zs[:, 1] - zs[:, 0]) / dx  # Diferencia hacia adelante en el borde izquierdo
    dzdx[:, -1] = (zs[:, -1] - zs[:, -2]) / dx  # Diferencia hacia atrás en el borde derecho
    
    # Calcular el gradiente en la dirección y
    for i in range(1, rows - 1):
        for j in range(cols):
            dzdy[i, j] = (zs[i + 1, j] - zs[i - 1, j]) / (2 * dy)  # Diferencia central
            
    # Bordes para dzdy
    dzdy[0, :] = (zs[1, :] - zs[0, :]) / dy  # Diferencia hacia adelante en el borde superior
    dzdy[-1, :] = (zs[-1, :] - zs[-2, :]) / dy  # Diferencia hacia atrás en el borde inferior
    
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
    # Calcular la magnitud del gradiente
    grad_magnitude = np.hypot(dzdx, dzdy)

    # Calcular el ángulo de inclinación
    theta = np.arctan(grad_magnitude)

    # Convertir el ángulo de inclinación a grados
    theta_degrees = np.degrees(theta)

    # Definir los límites de las categorías de inclinación
    bounds = [0, 15, 30, 45, 60]
    cmap = plt.get_cmap('RdYlGn_r')

    # Graficar el ángulo de inclinación con las categorías
    im = plt.imshow(theta_degrees.T, extent=(np.min(xs), np.max(xs), np.min(ys), np.max(ys)), origin='lower', cmap=cmap, vmin=bounds[0], vmax=bounds[-1])

    # Añadir las líneas de contorno en los límites de las categorías
    contour = plt.contour(xs, ys, theta_degrees, levels=bounds, colors='gray', linewidths=0.5)

    # Añadir una barra de color con las categorías
    cbar = plt.colorbar(im, ticks=bounds)
    cbar.set_label('Inclination angle [degrees]')

    # Añadir una cuadrícula con líneas discontinuas
    #plt.grid(True, linestyle='--', linewidth=0.5)

    # Configurar etiquetas y título
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.title('Heatmap - Slopes')

    # Mostrar la gráfica
    plt.show()