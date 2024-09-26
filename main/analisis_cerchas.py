# -*- coding: utf-8 -*-
"""
Funciones para análisis de cerchas planas.

@author: Nicolás Guarín-Zapata
@date: Agosto 2024
"""
import numpy as np
import matplotlib.pyplot as plt

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
             mec="#3c3c3c", mfc="#ffffff")
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
    
    
