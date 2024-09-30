# -*- coding: utf-8 -*-
"""
Funciones para análisis de cerchas planas.

@author: Nicolás Guarín-Zapata
@date: Agosto 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plane_trusses import analysis

integrantes = {"solución": "Nicolás Guarín-Zapata"}

plt.style.use("ggplot")
# Configurar el tamaño de fuente predeterminado para los ticks
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

def calc_esfuerzos_int(coords, eles, mats, desp):
    """Calcula los esfuerzos internos para una cercha plana

    Parámetros
    ----------
    coords : ndarray (float)
        Arreglo con coordenadas de los nodos.
    eles : ndarray (int)
        Arreglo con información de los elementos: propiedades
        y conectividades.
    mats : ndarray (float)
        Arreglo con información de las propiedades de los elementos:
        módulo de Young y sección.

    Retorna
    -------
    esfuerzos : ndarray (float)
        Arreglo con los esfuerzos axiales de cada elemento
    """
    neles = eles.shape[0]
    esfuerzos = np.zeros((neles))
    for cont in range(neles):
        ini = eles[cont, 1]
        fin = eles[cont, 2]
        longitud = np.linalg.norm(coords[fin, :] - coords[ini, :])
        mat_id = eles[cont, 0]
        young, _ = mats[mat_id] 
        long_nueva = np.linalg.norm((coords[fin, :] + desp[fin, :]) - 
                                    (coords[ini, :] + desp[ini, :]))
        elongacion = long_nueva - longitud
        esfuerzos[cont] = young * elongacion / longitud
    return esfuerzos

def graficar(coords, eles, desp=None, alpha=1.0):
    """Grafica la cercha en su configuración original o deformada.

    Parámetros
    ----------
    coords : ndarray (float)
        Arreglo con coordenadas de los nodos.
    eles : ndarray (int)
        Arreglo con información de los elementos: propiedades
        y conectividades.
    desp : ndarray (float), opcional
        Arreglo con los desplazamientos de cada nodo, por defecto None.
    """
    x, y = coords.T
    if desp is None:
        desp = np.zeros_like(coords)
    for barra in eles[:, 1:]:
        plt.plot(x[barra] + desp[barra, 0],
                 y[barra] + desp[barra, 1], color="#3c3c3c", alpha=alpha)
    plt.plot(x + desp[:, 0], y + desp[:, 1], lw=0, marker="o",
             mec="#3c3c3c", mfc="#ffffff", alpha=alpha)
    plt.axis("image")
    return None

def vis_esfuerzos(coords, eles, esfuerzos, desp=None):
    """Visualiza los esfuerzos de la cercha.

    El color azul representa un elemento sometido a compresión
    y el color rojo, uno sometida a tracción. La intensidad
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
    desp : ndarray (float), opcional
        Arreglo con los desplazamientos de cada nodo, por defecto None.
    """
     
    esfuerzo_max = max(-esfuerzos.min(), esfuerzos.max())
    esfuerzo_escalado =  0.5*(esfuerzos + esfuerzo_max)/esfuerzo_max
    x, y = coords.T
    if desp is None:
        desp = np.zeros_like(coords)

    
    graficar(coords, eles, desp=None, alpha=0.2)
    
        
    for cont, barra in enumerate(eles[:, 1:]):
        
        color = plt.cm.seismic(esfuerzo_escalado[cont])
        plt.plot(x[barra] + desp[barra, 0], y[barra] + desp[barra, 1],
                color=color, lw=3)
        plt.plot(x + desp[:, 0], y + desp[:, 1], lw=0, marker="o",
             mec="#3c3c3c", mfc="#ffffff", markersize=3.5)
    plt.axis("image")
    # Ajustar los límites de los ejes para que no quede apretada
    padding = 0.1 * max(x.max() - x.min(), y.max() - y.min())
    plt.xlim(x.min() - padding, x.max() + padding)
    plt.ylim(y.min() - padding, y.max() + padding)

def calc_peso(coords, eles, secciones, densidades):
    """Calcula las cargas debidas al peso de la estructura

    Parámetros
    ----------
    coords : ndarray (float)
        Arreglo con coordenadas de los nodos.
    eles : ndarray (int)
        Arreglo con información de los elementos: propiedades
        y conectividades.
    secciones : ndarray (float)
        Arreglo con secciones transversales de las barras.
    densidades : ndarray (float)
        Arreglo con densidades de los materiales de las barras.

    Retorna
    -------
    cargas : ndarray (float)
        Arreglo con las cargas nodales debidas al peso de la
        estructura.
    """
    neles = eles.shape[0]
    ncoords = coords.shape[0]
    cargas = np.zeros((ncoords))
    for cont in range(neles):
        ini = eles[cont, 1]
        fin = eles[cont, 2]
        longitud = np.linalg.norm(coords[fin, :] - coords[ini, :])
        seccion = secciones[cont]
        densidad = densidades[cont]
        peso = 9.81 * seccion * densidad * longitud
        cargas[ini] = cargas[ini] + 0.5 * peso
        cargas[fin] = cargas[fin] + 0.5 * peso
    return cargas

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

def optimizar_cercha(data, max_masa=10, max_esfuerzo=330, min_seccion=0.001, max_seccion=0.2):
    """
    Optimiza las secciones transversales de una cercha para minimizar la masa total,
    asegurando que no se superen los límites de esfuerzo y masa.

    Parámetros:
    data (dict): Un diccionario que contiene la información de la cercha, incluyendo:
        - "nodes": Un array de coordenadas de los nodos.
        - "elements": Un array de elementos que define las conexiones entre nodos.
        - "mats": Un array de materiales, donde se asume que las secciones transversales están en la columna 1.
    max_masa (float): La masa máxima permitida en toneladas.
    max_esfuerzo (float): El esfuerzo máximo permitido en MPa.
    min_seccion (float): La sección transversal mínima permitida.
    max_seccion (float): La sección transversal máxima permitida.

    Retorna:
    tuple: Una tupla que contiene:
        - dict: El diccionario de entrada actualizado con las secciones optimizadas.
        - float: La masa mínima de la cercha en toneladas.
    """
    # Inicialización de secciones
    secciones_iniciales = data["mats"][:, 1]  # Suponiendo que las secciones están en la columna 1

    def objetivo(secciones):
        data["mats"][:, 1] = secciones
        nodes = data["nodes"]
        elements = data["elements"]
        masa = calc_masa(nodes, elements, secciones, densidad=7800) / 1e3  # Convertir a toneladas 
        return masa  # Queremos minimizar la masa

    # Función de restricciones
    def restricciones(secciones):
        # Actualizar las secciones en el diccionario
        data["mats"][:, 1] = secciones
        nodes = data["nodes"]
        elements = data["elements"]
        mats = data["mats"]

        # Realizar el análisis para obtener los desplazamientos
        disp = analysis(data, verbose=False)  # desplazamientos en los nodos en x y y [m]
        esfuerzos = calc_esfuerzos_int(nodes, elements, mats, disp) / 1e6  # Convertir a MPa

        masa = calc_masa(nodes, elements, secciones, densidad=7800) / 1e3  # Convertir a toneladas
        # Restricción de esfuerzos y de masa
        return np.concatenate([
            (max_esfuerzo - esfuerzos - 1),  # No más de 330 MPa - epsilon
            [max_masa - masa - 1]            # No más de 10 toneladas - epsilon
        ])

    # Definir las restricciones para la optimización
    rest = {'type': 'ineq', 'fun': restricciones}

    # Definir límites para las secciones
    bounds = [(min_seccion, max_seccion)] * len(secciones_iniciales)

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
    del color representa la magnitud del esfuerzo. El grosor
    de las barras es proporcional a la sección transversal.

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
    
if __name__ == "__main__":
    from plane_trusses import analysis
    nodes = np.array([
        [8.66, 0.0],
        [0.0, 5.0],
        [0.0, 0.0]])
    cons = np.array([
        [0, 0],
        [-1, 0],
        [-1, -1]], dtype=int)
    elements = np.array([
        [0, 2, 0],
        [0, 0, 1],
        [0, 1, 2]], dtype=int)
    loads = np.array([
        [0.0, -1e6],
        [0.0, 0.0],
        [0.0, 0.0]])
    mats = np.array([
        [1e9, 0.1]])

    data = {
        "nodes": nodes,
        "cons": cons,
        "elements": elements,
        "loads": loads,
        "mats": mats}
    disp = analysis(data, verbose=True)
    
    esfuerzos = calc_esfuerzos_int(nodes, elements, mats, disp)

    plt.figure(figsize=(8, 6))
    graficar(nodes, elements, alpha=0.4)
    vis_esfuerzos(nodes, elements, esfuerzos, desp=disp)
    plt.show()