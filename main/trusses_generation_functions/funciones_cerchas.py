# -*- coding: utf-8 -*-
"""
Funciones para crear análisis de cerchas planas.

"""

import numpy as np

def warren_bridge_data(largo=24.0, alto=6, n=5, seccion=0.02, young=200e9, carga_total=400e3):
    """
    Genera los datos estructurales para una cercha tipo puente Warren.

    Parámetros:
    largo: Longitud total del puente (en metros).
    alto: Altura del puente (en metros).
    n: Número de segmentos horizontales en la mitad de la cercha.
    seccion: Área de la sección transversal de los elementos (en metros cuadrados).
    young: Módulo de Young del material (en Pascales).
    carga_total: Carga total distribuida en los nodos inferiores (en Newtons).

    Retorno:
    Un diccionario con los datos de los nodos, elementos, restricciones, cargas y materiales.
    """

    # Duplicar el número de segmentos debido a la simetría del puente
    n = n * 2

    # Crear lista para almacenar los nodos
    nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        nodes.append([i * (largo / n), 0.0])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n):
        nodes.append([i * (largo / n) + (largo / (2 * n)), alto])

    # Convertir lista de nodos a un array de numpy
    nodes = np.array(nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    elements = []

    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        elements.append([i, i, i + 1])

    # Diagonales: conectan nodos inferiores y superiores alternadamente
    for i in range(1, n + 1):
        # Diagonal ascendente
        elements.append([n + i, i - 1, n + i])  
        # Diagonal descendente
        elements.append([n + i + 1, n + i, i])

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(2, n + 1):
        elements.append([n + n + i, n + i - 1, n + i])

    # Convertir la lista de elementos a un array de numpy
    elements = np.array(elements)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    cons = np.zeros((nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    loads = np.zeros((nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        loads[i, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    sections = [seccion] * np.shape(elements)[0]  # Secciones iniciales
    
    # Crear un array con los módulos de Young de los elementos
    modulus_of_elasticity = np.ones_like(sections) * young

    # Crear un array con los módulos de Young y las secciones transversales
    materials = np.array([modulus_of_elasticity, sections]).T

    # Preparar el diccionario de datos como se espera en la función de análisis
    data = {
        "nodes": nodes,
        "cons": cons,
        "elements": elements,
        "loads": loads,
        "mats": materials
    }

    return data

def pratt_bridge_data(largo=24.0, alto=6, n=3, seccion=0.02, young=200e9, carga_total=400e3):
    """
    Genera los datos estructurales para una cercha tipo puente Pratt.

    Parámetros:
    largo: Longitud total del puente (en metros).
    alto: Altura del puente (en metros).
    n: Número de segmentos horizontales en la mitad de la cercha.
    seccion: Área de la sección transversal de los elementos (en metros cuadrados).
    young: Módulo de Young del material (en Pascales).
    carga_total: Carga total distribuida en los nodos inferiores (en Newtons).

    Retorno:
    Un diccionario con los datos de los nodos, elementos, restricciones, cargas y materiales.
    """

    # Duplicar el número de segmentos debido a la simetría del puente
    n = n * 2

    # Crear lista para almacenar los nodos
    nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        nodes.append([i * (largo / n), 0.0])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n -1):
        nodes.append([i * (largo / n) + (largo / (n)), alto])

    # Convertir lista de nodos a un array de numpy
    nodes = np.array(nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    eles = []
    ele_num = 0
    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        eles.append([ele_num, i, i + 1])
        ele_num += 1

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(2, n):
        eles.append([ele_num, n + i - 1, n + i])
        ele_num += 1

    # Diagonales: conectan nodos inferiores y superiores alternadamente
    for i in range(1, n):
        # straigth ascendente
        eles.append([ele_num, i, n + i])
        ele_num += 1

    for i in range(1, int(n/2)):
        # diagonal ascendente
        eles.append([ele_num, i+1, n + i])
        ele_num += 1

    for i in range(int(n/2), n-1):
        # diagonal descendente
        eles.append([ele_num,  i, n+i+1])
        ele_num += 1

    eles.append([ele_num,  0, n+1])
    ele_num += 1

    eles.append([ele_num,  n, 2*n-1])
    ele_num += 1

    eles = np.array(eles)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    cons = np.zeros((nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    loads = np.zeros((nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        loads[i, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    sections = [seccion] * np.shape(eles)[0]  # Secciones iniciales

    # Crear un array con los módulos de Young de los elementos
    modulus_of_elasticity = np.ones_like(sections) * young

    # Crear un array con los módulos de Young y las secciones transversales
    materials = np.array([modulus_of_elasticity, sections]).T

    # Preparar el diccionario de datos como se espera en la función de análisis
    data = {
        "nodes": nodes,
        "eles": eles,
        "cons": cons,
        "elements": eles,
        "loads": loads,
        "mats": materials    
    }

    return data

def howe_bridge_data(largo=24.0, alto=6, n=2, seccion=0.02, young=200e9, carga_total=400e3):
    """
    Genera los datos estructurales para una cercha tipo puente en Howe.

    Parámetros:
    largo: Longitud total del puente (en metros).
    alto: Altura del puente (en metros).
    n: Número de segmentos horizontales en la mitad de la cercha.
    seccion: Área de la sección transversal de los elementos (en metros cuadrados).
    young: Módulo de Young del material (en Pascales).
    carga_total: Carga total distribuida en los nodos inferiores (en Newtons).

    Retorno:
    Un diccionario con los datos de los nodos, elementos, restricciones, cargas y materiales.
    """
    # Duplicar el número de segmentos debido a la simetría del puente
    n = n * 2

    # Crear lista para almacenar los nodos
    nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        nodes.append([i * (largo / n), 0.0])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n -1):
        nodes.append([i * (largo / n) + (largo / (n)), alto])

    # Convertir lista de nodos a un array de numpy
    nodes = np.array(nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    eles = []
    ele_num = 0
    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        eles.append([ele_num, i, i + 1])
        ele_num += 1

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(2, n):
        eles.append([ele_num, n + i - 1, n + i])
        ele_num += 1

    # Diagonales: conectan nodos inferiores y superiores alternadamente
    for i in range(1, n):
        # straigth ascendente
        eles.append([ele_num, i, n + i])
        ele_num += 1

    for i in range(1, int(n/2)):
        # diagonal ascendente
        eles.append([ele_num, i, n + i +1])
        ele_num += 1

    for i in range(int(n/2), n-1):
        # diagonal descendente
        eles.append([ele_num,  n + i, i+1])
        ele_num += 1

    eles.append([ele_num,  0, n+1])
    ele_num += 1

    eles.append([ele_num,  n, 2*n-1])
    ele_num += 1

    eles = np.array(eles)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    cons = np.zeros((nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    loads = np.zeros((nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        loads[i, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    sections = [seccion] * np.shape(eles)[0]  # Secciones iniciales

    # Crear un array con los módulos de Young de los elementos
    modulus_of_elasticity = np.ones_like(sections) * young

    # Crear un array con los módulos de Young y las secciones transversales
    materials = np.array([modulus_of_elasticity, sections]).T

    # Preparar el diccionario de datos como se espera en la función de análisis
    data = {
        "nodes": nodes,
        "eles": eles,
        "cons": cons,
        "elements": eles,
        "loads": loads,
        "mats": materials    
    }

    return data

def double_warren_bridge_data(largo=24.0, alto=6, n=3, seccion=0.02, young=200e9, carga_total=400e3):
    """
    Genera los datos estructurales para una cercha tipo puente de Warren doble.

    Parámetros:
    largo: Longitud total del puente (en metros).
    alto: Altura del puente (en metros).
    n: Número de segmentos en la mitad de la cercha.
    seccion: Área de la sección transversal de los elementos (en metros cuadrados).
    young: Módulo de Young del material (en Pascales).
    carga_total: Carga total distribuida en los nodos inferiores (en Newtons).

    Retorno:
    Un diccionario con los datos de los nodos, elementos, restricciones, cargas y materiales.
    """

    # Duplicar el número de segmentos debido a la simetría del puente
    n = n * 2

    # Crear lista para almacenar los nodos
    nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        nodes.append([i * (largo / n), 0.0])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n -1):
        nodes.append([i * (largo / n) + (largo / (n)), alto])

    # Convertir lista de nodos a un array de numpy
    nodes = np.array(nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    eles = []
    ele_num = 0
    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        eles.append([ele_num, i, i + 1])
        ele_num += 1

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(2, n):
        eles.append([ele_num, n + i - 1, n + i])
        ele_num += 1
 
    for i in range(1, int(n/2)):
        # diagonal ascendente
        eles.append([ele_num, i+1, n + i])
        ele_num += 1

    for i in range(int(n/2), n-1):
        # diagonal descendente
        eles.append([ele_num,  i, n+i+1])
        ele_num += 1

    for i in range(1, int(n/2)):
        # diagonal ascendente
        eles.append([ele_num, i, n + i +1])
        ele_num += 1

    for i in range(int(n/2), n-1):
        # diagonal descendente
        eles.append([ele_num,  n + i, i+1])
        ele_num += 1


    eles.append([ele_num,  0, n+1])
    ele_num += 1

    eles.append([ele_num,  n, 2*n-1])
    ele_num += 1

    eles = np.array(eles)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    cons = np.zeros((nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    loads = np.zeros((nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        loads[2, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    sections = [seccion] * np.shape(eles)[0]  # Secciones iniciales

    # Crear un array con los módulos de Young de los elementos
    modulus_of_elasticity = np.ones_like(sections) * young

    # Crear un array con los módulos de Young y las secciones transversales
    materials = np.array([modulus_of_elasticity, sections]).T

    # Preparar el diccionario de datos como se espera en la función de análisis
    data = {
        "nodes": nodes,
        "eles": eles,
        "cons": cons,
        "elements": eles,
        "loads": loads,
        "mats": materials    
    }

    return data

def x_bridge_data(largo=24.0, alto=6, n=3, seccion=0.02, young=200e9, carga_total=400e3):

    """
    Genera los datos estructurales para una cercha tipo puente en X.

    Parámetros:
    largo: Longitud total del puente (en metros).
    alto: Altura del puente (en metros).
    n: Número de segmentos horizontales en la mitad de la cercha.
    seccion: Área de la sección transversal de los elementos (en metros cuadrados).
    young: Módulo de Young del material (en Pascales).
    carga_total: Carga total distribuida en los nodos inferiores (en Newtons).

    Retorno:
    Un diccionario con los datos de los nodos, elementos, restricciones, cargas y materiales.
    """

    # Duplicar el número de segmentos debido a la simetría del puente
    n = n * 2

    # Crear lista para almacenar los nodos
    nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        nodes.append([i * (largo / n), 0.0])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n -1):
        nodes.append([i * (largo / n) + (largo / (n)), alto])

    # Convertir lista de nodos a un array de numpy
    nodes = np.array(nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    eles = []
    ele_num = 0
    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        eles.append([ele_num, i, i + 1])
        ele_num += 1

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(2, n):
        eles.append([ele_num, n + i - 1, n + i])
        ele_num += 1

    # Diagonales: conectan nodos inferiores y superiores alternadamente
    for i in range(1, n):
        # straigth ascendente
        eles.append([ele_num, i, n + i])
        ele_num += 1

    for i in range(1, int(n/2)):
        # diagonal ascendente
        eles.append([ele_num, i+1, n + i])
        ele_num += 1

    for i in range(int(n/2), n-1):
        # diagonal descendente
        eles.append([ele_num,  i, n+i+1])
        ele_num += 1

    for i in range(1, int(n/2)):
        # diagonal ascendente
        eles.append([ele_num, i, n + i +1])
        ele_num += 1

    for i in range(int(n/2), n-1):
        # diagonal descendente
        eles.append([ele_num,  n + i, i+1])
        ele_num += 1


    eles.append([ele_num,  0, n+1])
    ele_num += 1

    eles.append([ele_num,  n, 2*n-1])
    ele_num += 1

    eles = np.array(eles)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    cons = np.zeros((nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    loads = np.zeros((nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        loads[i, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    sections = [seccion] * np.shape(eles)[0]  # Secciones iniciales

    # Crear un array con los módulos de Young de los elementos
    modulus_of_elasticity = np.ones_like(sections) * young

    # Crear un array con los módulos de Young y las secciones transversales
    materials = np.array([modulus_of_elasticity, sections]).T

    # Preparar el diccionario de datos como se espera en la función de análisis
    data = {
        "nodes": nodes,
        "eles": eles,
        "cons": cons,
        "elements": eles,
        "loads": loads,
        "mats": materials    
    }

    return data


def k_bridge_data(largo=24.0, alto=6, n=3, seccion=0.02, young=200e9, carga_total=400e3):
    """
    Genera los datos estructurales para una cercha tipo puente en K.

    Parámetros:
    largo: Longitud total del puente (en metros).
    alto: Altura del puente (en metros).
    n: Número de segmentos horizontales en la mitad de la cercha.
    seccion: Área de la sección transversal de los elementos (en metros cuadrados).
    young: Módulo de Young del material (en Pascales).
    carga_total: Carga total distribuida en los nodos inferiores (en Newtons).

    Retorno:
    Un diccionario con los datos de los nodos, elementos, restricciones, cargas y materiales.
    """

    # Duplicar el número de segmentos debido a la simetría del puente
    n = n * 2
 
    # Crear lista para almacenar los nodos
    nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        nodes.append([i * (largo / n), 0.0])

    # Nodos intermedios (desplazados a mitad del paso y a la altura definida)
    for i in range(0,int(n/2)-1):
        nodes.append([i * (largo / n) + (largo / (n)), alto/2])

    # Nodos intermedios (desplazados a mitad del paso y a la altura definida)
    for i in range(int(n/2),n -1):
        nodes.append([i * (largo / n) + (largo / (n)), alto/2])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n -1):
        nodes.append([i * (largo / n) + (largo / (n)), alto])

    # Convertir lista de nodos a un array de numpy
    nodes = np.array(nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    eles = []
    ele_num = 0
    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        eles.append([ele_num, i, i + 1])
        ele_num += 1

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(0, n-2):
        eles.append([ele_num, 2*n-1+i, 2*n+i])
        ele_num += 1

    for i in range(1, int(n/2)):
         
        eles.append([ele_num, i, i+n])
        ele_num += 1

    for i in range(int(n/2), n-1):
        
        eles.append([ele_num, i+1, i+n])
        ele_num += 1

    for i in range(0, int(n/2)-1):
         
        eles.append([ele_num, i+n+1, i+2*n-1])
        ele_num += 1        

    for i in range(int(n/2), n-1):
         
        eles.append([ele_num, i+n, i+2*n-1])
        ele_num += 1   

    for i in range(1, int(n/2)):
         
        eles.append([ele_num, i+1, i+n])
        ele_num += 1 

    for i in range(int(n/2), n-1):
        
        eles.append([ele_num, i, i+n])
        ele_num += 1

    for i in range(0, int(n/2)-1):
         
        eles.append([ele_num, i+n+1, i+2*n])
        ele_num += 1   

    for i in range(int(n/2), n-1):
         
        eles.append([ele_num, i+n, i+2*n-2])
        ele_num += 1 

    eles.append([ele_num,  0, 2*n+1-2])
    ele_num += 1

    eles.append([ele_num,  int(n/2), (n)+(n-2)+int(n/2)])
    ele_num += 1

    eles.append([ele_num,  n, 3*n-3])
    ele_num += 1

    eles = np.array(eles)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    cons = np.zeros((nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    loads = np.zeros((nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        loads[i, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    sections = [seccion] * np.shape(eles)[0]  # Secciones iniciales

    # Crear un array con los módulos de Young de los elementos
    modulus_of_elasticity = np.ones_like(sections) * young

    # Crear un array con los módulos de Young y las secciones transversales
    materials = np.array([modulus_of_elasticity, sections]).T

    # Preparar el diccionario de datos como se espera en la función de análisis
    data = {
        "nodes": nodes,
        "eles": eles,
        "cons": cons,
        "elements": eles,
        "loads": loads,
        "mats": materials    
    }

    return data