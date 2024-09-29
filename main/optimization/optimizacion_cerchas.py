# -*- coding: utf-8 -*-
"""
Funciones para análisis de cerchas planas.

@author: Nicolás Guarín-Zapata
@date: Agosto 2024
"""

import numpy as np
from scipy.optimize import minimize
from plane_trusses import analysis
from analisis_cerchas import calc_esfuerzos_int
import matplotlib.pyplot as plt

def calc_masa(coords, eles, secciones, densidad):
    """
    Calcula la masa de una estructura de cercha basada en sus coordenadas, 
    elementos, áreas de sección transversal y densidad del material.

    Parámetros:
    coords (numpy.ndarray): Array de coordenadas de los nodos con forma (ncoords, 2).
    eles (numpy.ndarray): Array de elementos con forma (neles, 3), donde cada fila 
                          contiene [element_id, nodo_inicial, nodo_final].
    secciones (numpy.ndarray): Array de áreas de sección transversal para cada elemento.
    densidad (float): Densidad del material.

    Retorna:
    float: La masa total de la estructura de cercha.
    """
    neles = eles.shape[0]
    vol = 0
    for cont in range(neles):
        ini = eles[cont, 1]
        fin = eles[cont, 2]
        longitud = np.linalg.norm(coords[fin, :] - coords[ini, :])
        seccion = secciones[cont]
        vol += seccion * longitud
    masa = densidad * vol   
    return masa

def calc_esfuerzos_masa(data):
    """
    Calcula los esfuerzos internos y la masa total de una cercha.

    Parámetros:
    data (dict): Un diccionario que contiene la información de la cercha, incluyendo:
        - "nodes": Un array de coordenadas de los nodos.
        - "elements": Un array de elementos que define las conexiones entre nodos.
        - "mats": Un array de materiales, donde se asume que las secciones transversales están en la columna 1.

    Retorna:
    tuple: Una tupla que contiene:
        - esfuerzos (numpy.ndarray): Un array con los esfuerzos internos en los elementos en MPa.
        - masa (float): La masa total de la cercha en toneladas.
    """
    # Extraer los datos necesarios del diccionario
    nodes = data["nodes"]
    elements = data["elements"]
    mats = data["mats"]

    # Realizar el análisis para obtener los desplazamientos
    disp = analysis(data, verbose=False)
    
    # Calcular las fuerzas internas en los elementos
    esfuerzos = calc_esfuerzos_int(nodes, elements, mats, disp) / 1e6  # Convertir a MPa

    # Calcular la masa total de la cercha
    masa = calc_masa(nodes, elements, mats[elements[:, 0], 1], densidad=7800) / 1e3  # Convertir a toneladas

    return esfuerzos, masa

def optimizar_cercha(data):
    """
    Optimiza las secciones transversales de una cercha para minimizar la masa total,
    asegurando que no se superen los límites de esfuerzo y masa.

    Parámetros:
    data (dict): Un diccionario que contiene la información de la cercha, incluyendo:
        - "nodes": Un array de coordenadas de los nodos.
        - "elements": Un array de elementos que define las conexiones entre nodos.
        - "mats": Un array de materiales, donde se asume que las secciones transversales están en la columna 1.

    Retorna:
    tuple: Una tupla que contiene:
        - dict: El diccionario de entrada actualizado con las secciones optimizadas.
        - float: La masa mínima de la cercha en toneladas.
    """
    # Inicialización de secciones
    secciones_iniciales = data["mats"][:, 1]  # Suponiendo que las secciones están en la columna 1

    # Definir los límites de masa y esfuerzo
    max_masa = 10  # 10 toneladas
    max_esfuerzo = 330  # 330 MPa 
    epsilon = 1e-3  # Pequeño valor para simular la desigualdad estricta

    # Función objetivo: minimizar la masa total
    def objetivo(secciones):
        data["mats"][:, 1] = secciones       
        _, masa = calc_esfuerzos_masa(data)  # Usar el diccionario para obtener esfuerzos y masa
        return masa  # Queremos minimizar la masa

    # Función de restricciones
    def restricciones(secciones):
        # Actualizar las secciones en el diccionario
        data["mats"][:, 1] = secciones
        esfuerzos, masa = calc_esfuerzos_masa(data)
        # Restricción de esfuerzos y de masa
        return np.concatenate([
            (max_esfuerzo - esfuerzos - epsilon),  # No más de 330 MPa - epsilon
            [max_masa - masa - epsilon]             # No más de 10 toneladas - epsilon
        ])

    # Definir las restricciones para la optimización
    rest = {'type': 'ineq', 'fun': restricciones}

    # Definir límites para las secciones
    bounds = [(0.001, 0.02)] * len(secciones_iniciales)

    # Realizar la optimización
    resultado = minimize(objetivo, secciones_iniciales, constraints=rest, bounds=bounds)

    # Obtener las secciones óptimas
    secciones_optimas = resultado.x
    
    # Extraer masa mínima
    masa_minima = resultado.fun
    
    # Actualizar el diccionario data con las secciones óptimas
    data["mats"][:, 1] = secciones_optimas

    return data, masa_minima  


def vis_esfuerzos_secciones(coords, eles, esfuerzos, mats, desp=None):
    """
    Visualiza los esfuerzos de la cercha.

    El color azul representa un elemento sometido a compresión
    y el color rojo, uno sometido a tracción. La intensidad
    del color representa la magnitud del esfuerzo.

    Parámetros
    ----------
    coords : ndarray (float)
        Arreglo con coordenadas de los nodos.
    eles : ndarray (int)
        Arreglo con información de los elementos: propiedades
        y conectividades.
    esfuerzos : ndarray (float)
        Arreglo con los esfuerzos de cada elemento.
    mats : ndarray (float)
        Arreglo con los materiales, donde se asume que las secciones transversales están en la columna 1.
    desp : ndarray (float), opcional
        Arreglo con los desplazamientos de cada nodo, por defecto None.
    """
    # Obtener las secciones óptimas y la sección máxima
    secciones_optimas = mats[:, 1]
    max_seccion = secciones_optimas.max()
    
    # Calcular el esfuerzo máximo y escalar los esfuerzos
    esfuerzo_max = max(-esfuerzos.min(), esfuerzos.max())
    esfuerzo_escalado = 0.5 * (esfuerzos + esfuerzo_max) / esfuerzo_max
    
    # Extraer las coordenadas x e y de los nodos
    x, y = coords.T
    
    # Si no se proporcionan desplazamientos, inicializar con ceros
    if desp is None:
        desp = np.zeros_like(coords)

    # Dibujar cada barra con el color y grosor correspondiente
    for cont, barra in enumerate(eles[:, 1:]):
        color = plt.cm.seismic(esfuerzo_escalado[cont])
        
        # Escalar el grosor proporcionalmente a la sección
        grosor = 5 * (secciones_optimas[cont]) / max_seccion
        plt.plot(x[barra] + desp[barra, 0], y[barra] + desp[barra, 1],
                 color=color, lw=grosor)

    # Dibujar los nodos
    plt.plot(x + desp[:, 0], y + desp[:, 1], lw=0, marker="o",
             mec="#3c3c3c", mfc="#ffffff", markersize=3.5)
    
    # Ajustar los límites de los ejes para que no quede apretada
    plt.axis("image")
    padding = 0.1 * max(x.max() - x.min(), y.max() - y.min())
    plt.xlim(x.min() - padding, x.max() + padding)
    plt.ylim(y.min() - padding, y.max() + padding)