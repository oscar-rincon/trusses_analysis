
# -*- coding: utf-8 -*-
"""
Este módulo define los datos estructurales del puentes, incluyendo nodos, restricciones,
cargas, elementos y materiales. Estos datos se utilizan para realizar análisis y optimización
de la estructura del puente.

Briges

1 - Warren Bridge
2 - Pratt Bridge
3 - Howe Bridge
4 - Double Warren Bridge
5 - X-Truss Bridge
6 - K-Truss Bridge

Roofs

1 - Howe Roof
2 - Pratt Roof
3 - Double Fink Roof
4 - Compund Fink Roof
5 - Fan Roof
6 - Modified Queen Roof
"""

import numpy as np


# Definir nodos (coordenadas de cada nodo en la cercha)
warren_bridge_nodes = np.array([
    [0.0, 0.0],    # Nodo 0: Soporte izquierdo
    [4.0, 0.0],    # Nodo 1
    [8.0, 0.0],    # Nodo 2
    [12.0, 0.0],   # Nodo 3
    [16.0, 0.0],   # Nodo 4
    [20.0, 0.0],   # Nodo 5
    [24.0, 0.0],   # Nodo 6: Soporte derecho
    [2.0, 6.0],    # Nodo 7: Nodo superior entre 0 y 1
    [6.0, 6.0],    # Nodo 8: Nodo superior entre 1 y 2
    [10.0, 6.0],   # Nodo 9: Nodo superior entre 2 y 3
    [14.0, 6.0],   # Nodo 10: Nodo superior entre 3 y 4
    [18.0, 6.0],   # Nodo 11: Nodo superior entre 4 y 5
    [22.0, 6.0],   # Nodo 12: Nodo superior entre 5 y 6
])

# Definir elementos (conectividad e índice de propiedades del material)
# [material_index, start_node, end_node]
warren_bridge_elements = np.array([
    [0, 0, 1],  # Chord inferior entre Nodo 0 y Nodo 1
    [1, 1, 2],  # Chord inferior entre Nodo 1 y Nodo 2
    [2, 2, 3],  # Chord inferior entre Nodo 2 y Nodo 3
    [3, 3, 4],  # Chord inferior entre Nodo 3 y Nodo 4
    [4, 4, 5],  # Chord inferior entre Nodo 4 y Nodo 5
    [5, 5, 6],  # Chord inferior entre Nodo 5 y Nodo 6
    [6, 0, 7],  # Diagonal entre Nodo 0 y Nodo 7
    [7, 7, 1],  # Diagonal entre Nodo 7 y Nodo 1
    [8, 1, 8],  # Diagonal entre Nodo 1 y Nodo 8
    [9, 8, 2],  # Diagonal entre Nodo 8 y Nodo 2
    [10, 2, 9],  # Diagonal entre Nodo 2 y Nodo 9
    [11, 9, 3],  # Diagonal entre Nodo 9 y Nodo 3
    [12, 3, 10], # Diagonal entre Nodo 3 y Nodo 10
    [13, 10, 4], # Diagonal entre Nodo 10 y Nodo 4
    [14, 4, 11], # Diagonal entre Nodo 4 y Nodo 11
    [15, 11, 5], # Diagonal entre Nodo 11 y Nodo 5
    [16, 5, 12], # Diagonal entre Nodo 5 y Nodo 12
    [17, 12, 6], # Diagonal entre Nodo 12 y Nodo 6
    [18, 7, 8],  # Chord superior entre Nodo 7 y Nodo 8
    [19, 8, 9],  # Chord superior entre Nodo 8 y Nodo 9
    [20, 9, 10], # Chord superior entre Nodo 9 y Nodo 10
    [21, 10, 11],# Chord superior entre Nodo 10 y Nodo 11
    [22, 11, 12],# Chord superior entre Nodo 11 y Nodo 12
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
# [x_constraint, y_constraint] para cada nodo
warren_bridge_cons = np.array([
    [0, -1],    # Nodo 0: Fijo en y (Soporte izquierdo)
    [0, 0],      # Nodo 1: Libre
    [0, 0],      # Nodo 2: Libre
    [0, 0],      # Nodo 3: Libre
    [0, 0],      # Nodo 4: Libre
    [0, 0],      # Nodo 5: Libre
    [-1, -1],     # Nodo 6: Fijo en y (Soporte derecho)
    [0, 0],      # Nodo 7: Libre
    [0, 0],      # Nodo 8: Libre
    [0, 0],      # Nodo 9: Libre
    [0, 0],      # Nodo 10: Libre
    [0, 0],      # Nodo 11: Libre
    [0, 0],      # Nodo 12: Libre
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
# [force_x, force_y] para cada nodo
warren_bridge_loads = np.array([
    [0.0, 0.0],   # Nodo 0: Sin carga (soporte)
    [0.0, -400e3],  # Nodo 1: Carga aplicada hacia abajo
    [0.0, -400e3],  # Nodo 2: Carga aplicada hacia abajo
    [0.0, -400e3],  # Nodo 3: Carga aplicada hacia abajo
    [0.0, -400e3],  # Nodo 4: Carga aplicada hacia abajo
    [0.0, -400e3],  # Nodo 5: Carga aplicada hacia abajo
    [0.0, 0.0],   # Nodo 6: Sin carga (soporte)
    [0.0, 0.0],   # Nodo 7: Sin carga
    [0.0, 0.0],   # Nodo 8: Sin carga
    [0.0, 0.0],   # Nodo 9: Sin carga
    [0.0, 0.0],   # Nodo 10: Sin carga
    [0.0, 0.0],   # Nodo 11: Sin carga
    [0.0, 0.0],   # Nodo 12: Sin carga
])

# Inicialización de secciones con el caso más costoso
warren_bridge_sections = [0.02] * np.shape(warren_bridge_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
warren_bridge_modulus_of_elasticity = np.ones_like(warren_bridge_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
warren_bridge_materials = np.array([warren_bridge_modulus_of_elasticity, warren_bridge_sections]).T


# Preparar el diccionario de datos como se espera en la función de análisis
warren_bridge_data = {
    "nodes": warren_bridge_nodes,
    "cons": warren_bridge_cons,
    "elements": warren_bridge_elements,
    "loads": warren_bridge_loads,
    "mats": warren_bridge_materials
}

"""
2 - Pratt Bridge
"""

pratt_bridge_nodes = np.array([
    [0.0, 0.0],    # Nodo 0: Soporte izquierdo
    [4.0, 0.0],    # Nodo 1
    [8.0, 0.0],    # Nodo 2
    [12.0, 0.0],   # Nodo 3
    [16.0, 0.0],   # Nodo 4
    [20.0, 0.0],   # Nodo 5
    [24.0, 0.0],   # Nodo 6: Soporte derecho
    [4.0, 6.0],    # Nodo 7: Nodo superior entre 0 y 1
    [8.0, 6.0],    # Nodo 8: Nodo superior entre 1 y 2
    [12.0, 6.0],   # Nodo 9: Nodo superior entre 2 y 3
    [16.0, 6.0],   # Nodo 10: Nodo superior entre 3 y 4
    [20.0, 6.0]    # Nodo 11: Nodo superior entre 4 y 5
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
# [x_constraint, y_constraint] para cada nodo
pratt_bridge_constraints = np.array([
    [0, -1],    # Nodo 0: Fijo en x e y (Soporte izquierdo)
    [0, 0],      # Nodo 1: Libre
    [0, 0],      # Nodo 2: Libre
    [0, 0],      # Nodo 3: Libre
    [0, 0],      # Nodo 4: Libre
    [0, 0],      # Nodo 5: Libre    
    [-1, -1],     # Nodo 6: Fijo en x e y (Soporte derecho)
    [0, 0],      # Nodo 7: Libre
    [0, 0],      # Nodo 8: Libre
    [0, 0],      # Nodo 9: Libre
    [0, 0],      # Nodo 10: Libre
    [0, 0]       # Nodo 11: Libre
], dtype=int)

# Definir elementos (conectividad e índice de propiedades del material)
# [material_index, start_node, end_node]
pratt_bridge_elements = np.array([
    # Chord inferior (entre nodos inferiores)
    [0, 0, 1],  # Entre Nodo 0 y Nodo 1
    [1, 1, 2],  # Entre Nodo 1 y Nodo 2
    [2, 2, 3],  # Entre Nodo 2 y Nodo 3
    [3, 3, 4],  # Entre Nodo 3 y Nodo 4
    [4, 4, 5],  # Entre Nodo 4 y Nodo 5
    [5, 5, 6],  # Entre Nodo 5 y Nodo 6
    
    # Chord superior (entre nodos superiores)
    [6, 7, 8],  # Entre Nodo 7 y Nodo 8
    [7, 8, 9],  # Entre Nodo 8 y Nodo 9
    [8, 9, 10], # Entre Nodo 9 y Nodo 10
    [9, 10, 11],# Entre Nodo 10 y Nodo 11

    # Montantes verticales (conectan nodos superiores con los inferiores)
    [10, 1, 7],  # Montante entre Nodo 1 y Nodo 7
    [11, 2, 8],  # Montante entre Nodo 2 y Nodo 8
    [12, 3, 9],  # Montante entre Nodo 3 y Nodo 9
    [13, 4, 10], # Montante entre Nodo 4 y Nodo 10
    [14, 5, 11], # Montante entre Nodo 5 y Nodo 11

    # Diagonales del tipo Pratt (diagonales hacia el centro del puente)
    [15, 0, 7],  # Diagonal entre Nodo 0 y Nodo 7
    [16, 7, 2],  # Diagonal entre Nodo 7 y Nodo 2
    [17, 8, 3],  # Diagonal entre Nodo 2 y Nodo 9
    [18, 3, 10], # Diagonal entre Nodo 9 y Nodo 4
    [19, 4, 11], # Diagonal entre Nodo 4 y Nodo 11
    [20, 11, 6]  # Diagonal entre Nodo 11 y Nodo 6
])

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
# [force_x, force_y] para cada nodo
pratt_bridge_loads = np.array([
    [0.0, 0.0],  # Nodo 0: Sin carga (soporte izquierdo)
    [0.0, -400e3], # Nodo 1: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 2: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 3: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 4: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 5: Carga aplicada hacia abajo (-1e3 N)
    [0.0, 0.0],  # Nodo 6: Sin carga (soporte derecho)
    [0.0, 0.0],  # Nodo 7: Sin carga (nodo superior)
    [0.0, 0.0],  # Nodo 8: Sin carga (nodo superior)
    [0.0, 0.0],  # Nodo 9: Sin carga (nodo superior)
    [0.0, 0.0],  # Nodo 10: Sin carga (nodo superior)
    [0.0, 0.0]   # Nodo 11: Sin carga (nodo superior)
])

# Inicialización de secciones con el caso más costoso
pratt_bridge_sections = [0.02] * np.shape(pratt_bridge_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
pratt_bridge_modulus_of_elasticity = np.ones_like(pratt_bridge_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
pratt_bridge_materials = np.array([pratt_bridge_modulus_of_elasticity, pratt_bridge_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
pratt_bridge_data = {
    "nodes": pratt_bridge_nodes,
    "cons": pratt_bridge_constraints,
    "elements": pratt_bridge_elements,
    "loads": pratt_bridge_loads,
    "mats": pratt_bridge_materials
}

"""
Howe Bridge
"""

howe_bridge_nodes = np.array([
    [0.0, 0.0],    # Nodo 0: Soporte izquierdo
    [4.0, 0.0],    # Nodo 1
    [8.0, 0.0],    # Nodo 2
    [12.0, 0.0],   # Nodo 3
    [16.0, 0.0],   # Nodo 4
    [20.0, 0.0],   # Nodo 5
    [24.0, 0.0],   # Nodo 6: Soporte derecho
    [4.0, 6.0],    # Nodo 7: Nodo superior entre 0 y 1
    [8.0, 6.0],    # Nodo 8: Nodo superior entre 1 y 2
    [12.0, 6.0],   # Nodo 9: Nodo superior entre 2 y 3
    [16.0, 6.0],   # Nodo 10: Nodo superior entre 3 y 4
    [20.0, 6.0]    # Nodo 11: Nodo superior entre 4 y 5
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
# [x_constraint, y_constraint] para cada nodo
howe_bridge_constraints = np.array([
    [0, -1],    # Nodo 0: Fijo en x e y (Soporte izquierdo)
    [0, 0],      # Nodo 1: Libre
    [0, 0],      # Nodo 2: Libre
    [0, 0],      # Nodo 3: Libre
    [0, 0],      # Nodo 4: Libre
    [0, 0],      # Nodo 5: Libre    
    [-1, -1],     # Nodo 6: Fijo en x e y (Soporte derecho)
    [0, 0],      # Nodo 7: Libre
    [0, 0],      # Nodo 8: Libre
    [0, 0],      # Nodo 9: Libre
    [0, 0],      # Nodo 10: Libre
    [0, 0]       # Nodo 11: Libre
], dtype=int)

# Definir elementos (conectividad e índice de propiedades del material)
# [material_index, start_node, end_node]
howe_bridge_elements = np.array([
    # Chord inferior (entre nodos inferiores)
    [0, 0, 1],  # Entre Nodo 0 y Nodo 1
    [1, 1, 2],  # Entre Nodo 1 y Nodo 2
    [2, 2, 3],  # Entre Nodo 2 y Nodo 3
    [3, 3, 4],  # Entre Nodo 3 y Nodo 4
    [4, 4, 5],  # Entre Nodo 4 y Nodo 5
    [5, 5, 6],  # Entre Nodo 5 y Nodo 6
    
    # Chord superior (entre nodos superiores)
    [6, 7, 8],  # Entre Nodo 7 y Nodo 8
    [7, 8, 9],  # Entre Nodo 8 y Nodo 9
    [8, 9, 10], # Entre Nodo 9 y Nodo 10
    [9, 10, 11],# Entre Nodo 10 y Nodo 11

    # Montantes verticales (conectan nodos superiores con los inferiores)
    [10, 1, 7],  # Montante entre Nodo 1 y Nodo 7
    [11, 2, 8],  # Montante entre Nodo 2 y Nodo 8
    [12, 3, 9],  # Montante entre Nodo 3 y Nodo 9
    [13, 4, 10], # Montante entre Nodo 4 y Nodo 10
    [14, 5, 11], # Montante entre Nodo 5 y Nodo 11

    # Diagonales del tipo Pratt (diagonales hacia el centro del puente)
    [15, 0, 7],  # Diagonal entre Nodo 0 y Nodo 7
    [16, 1, 8],  # Diagonal entre Nodo 7 y Nodo 2
    [17, 2, 9],  # Diagonal entre Nodo 2 y Nodo 9
    [18, 9, 4], # Diagonal entre Nodo 9 y Nodo 4
    [19, 10, 5], # Diagonal entre Nodo 4 y Nodo 11
    [20, 11, 6]  # Diagonal entre Nodo 11 y Nodo 6
])

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
# [force_x, force_y] para cada nodo
howe_bridge_loads = np.array([
    [0.0, 0.0],  # Nodo 0: Sin carga (soporte izquierdo)
    [0.0, -400e3], # Nodo 1: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 2: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 3: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 4: Carga aplicada hacia abajo (-1e3 N)
    [0.0, -400e3], # Nodo 5: Carga aplicada hacia abajo (-1e3 N)
    [0.0, 0.0],  # Nodo 6: Sin carga (soporte derecho)
    [0.0, 0.0],  # Nodo 7: Sin carga (nodo superior)
    [0.0, 0.0],  # Nodo 8: Sin carga (nodo superior)
    [0.0, 0.0],  # Nodo 9: Sin carga (nodo superior)
    [0.0, 0.0],  # Nodo 10: Sin carga (nodo superior)
    [0.0, 0.0]   # Nodo 11: Sin carga (nodo superior)
])

# Inicialización de secciones con el caso más costoso
howe_bridge_sections = [0.02] * np.shape(howe_bridge_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
howe_bridge_modulus_of_elasticity = np.ones_like(howe_bridge_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
howe_bridge_materials = np.array([howe_bridge_modulus_of_elasticity, howe_bridge_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
howe_bridge_data = {
    "nodes": howe_bridge_nodes,
    "cons": howe_bridge_constraints,
    "elements": howe_bridge_elements,
    "loads": howe_bridge_loads,
    "mats": howe_bridge_materials
}

"""
4 - Double Warren Bridge
"""

# Definir nodos (coordenadas de cada nodo en la cercha)
double_warren_bridge_nodes = np.array([
    [0.0, 0.0],    # Nodo 0: Soporte izquierdo
    [4.0, 0.0],    # Nodo 1
    [8.0, 0.0],    # Nodo 2
    [12.0, 0.0],   # Nodo 3
    [16.0, 0.0],   # Nodo 4
    [20.0, 0.0],   # Nodo 5
    [24.0, 0.0],   # Nodo 6: Soporte derecho
    [4.0, 6.0],    # Nodo 7: Nodo superior entre 0 y 1
    [8.0, 6.0],    # Nodo 8: Nodo superior entre 1 y 2
    [12.0, 6.0],   # Nodo 9: Nodo superior entre 2 y 3
    [16.0, 6.0],   # Nodo 10: Nodo superior entre 3 y 4
    [20.0, 6.0]    # Nodo 11: Nodo superior entre 4 y 5
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
# [x_constraint, y_constraint] para cada nodo
double_warren_bridge_constraints = np.array([
    [0, -1],    # Nodo 0: Fijo en y (Soporte izquierdo)
    [0, 0],      # Nodo 1: Libre
    [0, 0],      # Nodo 2: Libre
    [0, 0],      # Nodo 3: Libre
    [0, 0],      # Nodo 4: Libre
    [0, 0],      # Nodo 5: Libre
    [-1, -1],     # Nodo 6: Fijo en y (Soporte derecho)
    [0, 0],      # Nodo 7: Libre
    [0, 0],      # Nodo 8: Libre
    [0, 0],      # Nodo 9: Libre
    [0, 0],      # Nodo 10: Libre
    [0, 0],      # Nodo 11: Libre
], dtype=int)

# Definir elementos (conectividad e índice de propiedades del material)
# [material_index, start_node, end_node]
double_warren_bridge_elements = np.array([
    [0, 0, 1],  # Chord inferior entre Nodo 0 y Nodo 1
    [1, 1, 2],  # Chord inferior entre Nodo 1 y Nodo 2
    [2, 2, 3],  # Chord inferior entre Nodo 2 y Nodo 3
    [3, 3, 4],  # Chord inferior entre Nodo 3 y Nodo 4
    [4, 4, 5],  # Chord inferior entre Nodo 4 y Nodo 5
    [5, 5, 6],  # Chord inferior entre Nodo 5 y Nodo 6

    [6, 7, 8],  # Chord superior entre Nodo 7 y Nodo 8
    [7, 8, 9],  # Chord superior entre Nodo 8 y Nodo 9
    [8, 9, 10], # Chord superior entre Nodo 9 y Nodo 10
    [9, 10, 11],# Chord superior entre Nodo 10 y Nodo 11

    # Verticales
    [10, 1, 7],  # Diagonal entre Nodo 0 y Nodo 7
    [11, 5, 11], # Diagonal entre Nodo 4 y Nodo 11

    # Diagonales
    [12, 0, 7],  # Diagonal entre Nodo 0 y Nodo 8
    [13, 7, 2],  # Diagonal entre Nodo 1 y Nodo 9
    [14, 1, 8],  # Diagonal entre Nodo 2 y Nodo 10
    [15, 8, 3],  # Diagonal entre Nodo 3 y Nodo 11
    [16, 2, 9],  # Diagonal entre Nodo 4 y Nodo 12
    [17, 9, 4],  # Diagonal entre Nodo 5 y Nodo 13
    [18, 3, 10], # Diagonal entre Nodo 6 y Nodo 14
    [19, 10, 5], # Diagonal entre Nodo 7 y Nodo 15
    [20, 4, 11], # Diagonal entre Nodo 8 y Nodo 16
    [21, 11, 6], # Diagonal entre Nodo 9 y Nodo 17
])

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
# [force_x, force_y] para cada nodo
double_warren_bridge_loads = np.array([
    [0.0, 0.0],  # Nodo 0: Sin carga (soporte)
    [0.0, -400e3], # Nodo 1: Carga aplicada hacia abajo
    [0.0, -400e3], # Nodo 2: Carga aplicada hacia abajo
    [0.0, -400e3], # Nodo 3: Carga aplicada hacia abajo
    [0.0, -400e3], # Nodo 4: Carga aplicada hacia abajo
    [0.0, -400e3], # Nodo 5: Carga aplicada hacia abajo
    [0.0, 0.0],  # Nodo 6: Sin carga (soporte)
    [0.0, 0.0],  # Nodo 7: Sin carga
    [0.0, 0.0],  # Nodo 8: Sin carga
    [0.0, 0.0],  # Nodo 9: Sin carga
    [0.0, 0.0],  # Nodo 10: Sin carga
    [0.0, 0.0],  # Nodo 11: Sin carga
])

# Inicialización de secciones con el caso más costoso
double_warren_bridge_sections = [0.02] * np.shape(double_warren_bridge_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
double_warren_bridge_modulus_of_elasticity = np.ones_like(double_warren_bridge_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
double_warren_bridge_materials = np.array([double_warren_bridge_modulus_of_elasticity, double_warren_bridge_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
double_warren_bridge_data = {
    "nodes": double_warren_bridge_nodes,
    "cons": double_warren_bridge_constraints,
    "elements": double_warren_bridge_elements,
    "loads": double_warren_bridge_loads,
    "mats": double_warren_bridge_materials
}

"""
5 - X-Truss Bridge
"""
# Cada fila representa un nodo con sus coordenadas (x, y)
x_truss_bridge_nodes = np.array([
    [0.0, 0.0],   # Nodo 0: Soporte izquierdo
    [4.0, 0.0],   # Nodo 1
    [8.0, 0.0],   # Nodo 2
    [12.0, 0.0],  # Nodo 3
    [16.0, 0.0],  # Nodo 4
    [20.0, 0.0],  # Nodo 5
    [24.0, 0.0],  # Nodo 6: Soporte derecho
    [4.0, 6.0],   # Nodo 7: Nodo superior entre 0 y 1
    [8.0, 6.0],   # Nodo 8: Nodo superior entre 1 y 2
    [12.0, 6.0],  # Nodo 9: Nodo superior entre 2 y 3
    [16.0, 6.0],  # Nodo 10: Nodo superior entre 3 y 4
    [20.0, 6.0],  # Nodo 11: Nodo superior entre 4 y 5
])

# -------------------------------
# Definición de restricciones
# -------------------------------
# Las restricciones se definen para cada nodo como [x_constraint, y_constraint]
# Donde -1 = fijo, 0 = libre
x_truss_bridge_constraints = np.array([
    [0, -1],  # Nodo 0: Fijo en x e y (Soporte izquierdo)
    [0, 0],    # Nodo 1: Libre en ambas direcciones
    [0, 0],    # Nodo 2: Libre en ambas direcciones
    [0, 0],    # Nodo 3: Libre en ambas direcciones
    [0, 0],    # Nodo 4: Libre en ambas direcciones
    [0, 0],    # Nodo 5: Libre en ambas direcciones
    [-1, -1],   # Nodo 6: Fijo en y, libre en x (Soporte derecho)
    [0, 0],    # Nodo 7: Libre
    [0, 0],    # Nodo 8: Libre
    [0, 0],    # Nodo 9: Libre
    [0, 0],    # Nodo 10: Libre
    [0, 0],    # Nodo 11: Libre
], dtype=int)

# -------------------------------
# Definición de elementos
# -------------------------------
# Cada fila representa un elemento en formato [material_index, start_node, end_node]
x_truss_bridge_elements = np.array([
    [0, 0, 1],  # Chord inferior entre Nodo 0 y Nodo 1
    [1, 1, 2],  # Chord inferior entre Nodo 1 y Nodo 2
    [2, 2, 3],  # Chord inferior entre Nodo 2 y Nodo 3
    [3, 3, 4],  # Chord inferior entre Nodo 3 y Nodo 4
    [4, 4, 5],  # Chord inferior entre Nodo 4 y Nodo 5
    [5, 5, 6],  # Chord inferior entre Nodo 5 y Nodo 6

    [6, 7, 8],  # Chord superior entre Nodo 7 y Nodo 8
    [7, 8, 9],  # Chord superior entre Nodo 8 y Nodo 9
    [8, 9, 10], # Chord superior entre Nodo 9 y Nodo 10
    [9, 10, 11],# Chord superior entre Nodo 10 y Nodo 11

    # Elementos verticales
    [10, 1, 7],  # Vertical entre Nodo 1 y Nodo 7
    [11, 2, 8],  # Vertical entre Nodo 2 y Nodo 8
    [12, 3, 9],  # Vertical entre Nodo 3 y Nodo 9
    [13, 4, 10], # Vertical entre Nodo 4 y Nodo 10
    [14, 5, 11], # Vertical entre Nodo 5 y Nodo 11

    # Elementos diagonales
    [15, 0, 7],  # Diagonal entre Nodo 0 y Nodo 7
    [16, 7, 2],  # Diagonal entre Nodo 7 y Nodo 2
    [17, 1, 8],  # Diagonal entre Nodo 1 y Nodo 8
    [18, 8, 3],  # Diagonal entre Nodo 8 y Nodo 3
    [19, 2, 9],  # Diagonal entre Nodo 2 y Nodo 9
    [20, 9, 4],  # Diagonal entre Nodo 9 y Nodo 4
    [21, 3, 10], # Diagonal entre Nodo 3 y Nodo 10
    [22, 10, 5], # Diagonal entre Nodo 10 y Nodo 5
    [23, 4, 11], # Diagonal entre Nodo 4 y Nodo 11
    [24, 11, 6], # Diagonal entre Nodo 11 y Nodo 6
])

# -------------------------------
# Definición de cargas
# -------------------------------
# Fuerzas aplicadas en cada nodo como [force_x, force_y]
x_truss_bridge_loads = np.array([
    [0.0, 0.0],   # Nodo 0: Sin carga (soporte)
    [0.0, -400e3], # Nodo 1: Carga de -2 kN en y
    [0.0, -400e3], # Nodo 2: Carga de -2 kN en y
    [0.0, -400e3], # Nodo 3: Carga de -2 kN en y
    [0.0, -400e3], # Nodo 4: Carga de -2 kN en y
    [0.0, -400e3], # Nodo 5: Carga de -2 kN en y
    [0.0, 0.0],   # Nodo 6: Sin carga (soporte)
    [0.0, 0.0],   # Nodo 7: Sin carga
    [0.0, 0.0],   # Nodo 8: Sin carga
    [0.0, 0.0],   # Nodo 9: Sin carga
    [0.0, 0.0],   # Nodo 10: Sin carga
    [0.0, 0.0],   # Nodo 11: Sin carga
])

# Inicialización de secciones con el caso más costoso
x_truss_bridge_sections = [0.02] * np.shape(x_truss_bridge_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
x_truss_bridge_modulus_of_elasticity = np.ones_like(x_truss_bridge_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
x_truss_bridge_materials = np.array([x_truss_bridge_modulus_of_elasticity, x_truss_bridge_sections]).T


# -------------------------------
# Preparación de datos para el análisis
# -------------------------------
x_truss_bridge_data = {
    "nodes": x_truss_bridge_nodes,
    "cons": x_truss_bridge_constraints,
    "elements": x_truss_bridge_elements,
    "loads": x_truss_bridge_loads,
    "mats": x_truss_bridge_materials
}

"""
6 - K-Truss Bridge
"""

k_truss_bridge_nodes = np.array([
    [0,  0],
    [4,  0],
    [8,  0],
    [12, 0],
    [16, 0],
    [20, 0],
    [24, 0],
    [4,  6],
    [8,  6],
    [12, 6],
    [16, 6],
    [20, 6],
    [4,  3],
    [8,  3],
    [16, 3],
    [20, 3]
])

k_truss_bridge_constraints = np.array([
    [0, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [-1, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
], dtype=int)

k_truss_bridge_loads = np.array([
    [0, 0],
    [0, -400e3],
    [0, -400e3],
    [0, -400e3],
    [0, -400e3],
    [0, -400e3],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])

k_truss_bridge_elements = np.array([
    [0, 0,  1],
    [1, 1,  2],
    [2, 2,  3],
    [3, 3,  4],
    [4, 4,  5],
    [5, 5,  6],
    [6, 7,  8],
    [7, 8,  9],
    [8, 9,  10],
    [9, 10, 11],
    [10, 0,  7],
    [11, 6,  11],
    [12, 1,  12],
    [13, 2,  13],
    [14, 4,  14],
    [15, 5,  15],
    [16, 12, 7],
    [17, 13, 8],
    [18, 14, 10],
    [19, 15, 11],
    [20, 2,  12],
    [21, 3,  13],
    [22, 3,  14],
    [23, 4,  15],
    [24, 12, 8],
    [25, 13, 9],
    [26, 9,  14],
    [27, 10, 15],
    [28, 3,  9]
], dtype=int)

# Inicialización de secciones con el caso más costoso
k_truss_bridge_sections = [0.02] * np.shape(k_truss_bridge_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
k_truss_bridge_modulus_of_elasticity = np.ones_like(k_truss_bridge_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
k_truss_bridge_materials = np.array([k_truss_bridge_modulus_of_elasticity, k_truss_bridge_sections]).T

k_truss_bridge_data = {
    "nodes": k_truss_bridge_nodes,
    "cons": k_truss_bridge_constraints,
    "elements": k_truss_bridge_elements,
    "loads": k_truss_bridge_loads,
    "mats": k_truss_bridge_materials
}


"""
1 - Howe Roof Truss
"""

# Definir nodos (coordenadas de cada nodo en la cercha)
howe_roof_nodes = np.array([
    [0,  0],
    [2,  0],
    [4,  0],
    [6,  0],
    [8,  0],
    [10, 0],
    [12, 0],
    [2,  1],
    [10, 1],
    [4,  2],
    [8,  2],
    [6,  3]
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
# [x_constraint, y_constraint] para cada nodo
howe_roof_cons = np.array([
    [0, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [-1, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
# [force_x, force_y] para cada nodo
howe_roof_loads = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, -10_000e3/5],
    [0, -10_000e3/5],
    [0, -10_000e3/5],
    [0, -10_000e3/5],
    [0, -10_000e3/5]
])

# Definir elementos (conectividad e índice de propiedades del material)
# [material_index, start_node, end_node]
howe_roof_elements = np.array([
    [0, 0,  1],
    [1, 1,  7],
    [2, 7,  0],
    [3, 1,  2],
    [4, 2,  7],
    [5, 2,  3],
    [6, 3,  9],
    [7, 9,  2],
    [8, 3,  4],
    [9, 4,  10],
    [10, 10, 3],
    [11, 3,  11],
    [12, 4,  5],
    [13, 5,  8],
    [14, 8,  4],
    [15, 8,  10],
    [16, 5,  6],
    [17, 6,  8],
    [18, 10, 11],
    [19, 11, 9],
    [20, 9,  7]
], dtype=int)

# Inicialización de secciones con el caso más costoso
howe_roof_sections = [0.02] * np.shape(howe_roof_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
howe_roof_modulus_of_elasticity = np.ones_like(howe_roof_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
howe_roof_materials = np.array([howe_roof_modulus_of_elasticity, howe_roof_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
howe_roof_data = {
    "nodes": howe_roof_nodes,
    "cons": howe_roof_cons,
    "elements": howe_roof_elements,
    "loads": howe_roof_loads,
    "mats": howe_roof_materials
}

"""
2 - Pratt Roof Truss
"""

# Definir nodos (coordenadas de cada nodo en la cercha)
pratt_roof_nodes = np.array([
    [0,  0], # Nodo 0
    [2,  0], # Nodo 1
    [4,  0], # Nodo 2
    [6,  0], # Nodo 3
    [8,  0], # Nodo 4
    [10, 0], # Nodo 5
    [12, 0], # Nodo 6
    [2,  1], # Nodo 7
    [10, 1], # Nodo 8
    [4,  2], # Nodo 9
    [8,  2], # Nodo 10
    [6,  3] # Nodo 11
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
# [x_constraint, y_constraint] para cada nodo
pratt_roof_cons = np.array([
    [0, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [-1, -1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
# [force_x, force_y] para cada nodo
pratt_roof_loads = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, -10_000e3/5],
    [0, -10_000e3/5],
    [0, -10_000e3/5],
    [0, -10_000e3/5],
    [0, -10_000e3/5]
])

# Definir elementos (conectividad e índice de propiedades del material)
# [material_index, start_node, end_node]
pratt_roof_elements = np.array([
    [0, 0,  1],   # Chord inferior
    [1, 1,  7],   # Diagonal
    [2, 7,  0],   # Diagonal
    [3, 1,  2],   # Chord inferior
    [4, 1,  9],   # Diagonal
    [5, 2,  3],   # Chord inferior
    [6, 2,  11],   # Diagonal
    [7, 9,  2],   # Diagonal
    [8, 3,  4],   # Chord inferior
    [9, 4,  10],  # Diagonal
    [10, 11, 4],  # Diagonal
    [11, 3,  11], # Diagonal
    [12, 4,  5],  # Chord inferior
    [13, 5,  8],  # Diagonal
    [14, 10,  5],  # Diagonal
    [15, 8,  10], # Diagonal
    [16, 5,  6],  # Chord inferior
    [17, 6,  8],  # Diagonal
    [18, 10, 11], # Diagonal
    [19, 11, 9],  # Diagonal
    [20, 9,  7]   # Diagonal
], dtype=int)

# Inicialización de secciones con el caso más costoso
pratt_roof_sections = [0.02] * np.shape(pratt_roof_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
pratt_roof_modulus_of_elasticity = np.ones_like(pratt_roof_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
pratt_roof_materials = np.array([pratt_roof_modulus_of_elasticity, pratt_roof_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
pratt_roof_data = {
    "nodes": pratt_roof_nodes,
    "cons": pratt_roof_cons,
    "elements": pratt_roof_elements,
    "loads": pratt_roof_loads,
    "mats": pratt_roof_materials
}

"""
Double Fink Roof Truss
"""
# Definir nodos (coordenadas de cada nodo en la cercha)
double_fink_roof_nodes = np.array([
    [0,  0],  # Nodo 0
    [2.4,  0],  # Nodo 1
    [2*2.4,  0],  # Nodo 2
    [3*2.4,  0],  # Nodo 3
    [4*2.4,  0],  # Nodo 4
    [5*2.4, 0],  # Nodo 5
    [2,  1],  # Nodo 6
    [4,  2],  # Nodo 7
    [6,  3],  # Nodo 8
    [8,  2],  # Nodo 9
    [10,  1],  # Nodo 10
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
double_fink_roof_cons = np.array([
    [0, -1], # Nodo 0
    [0, 0], # Nodo 1
    [0, 0], # Nodo 2
    [0, 0], # Nodo 3
    [0, 0], # Nodo 4
    [-1, -1], # Nodo 5
    [0, 0], # Nodo 6
    [0, 0], # Nodo 7
    [0, 0], # Nodo 8
    [0, 0], # Nodo 9
    [0, 0], # Nodo 10
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
double_fink_roof_loads = np.array([
    [0, 0], # Nodo 0: Carga de -2 kN en y
    [0, 0], # Nodo 1: Sin carga
    [0, 0], # Nodo 2: Sin carga
    [0, 0], # Nodo 3: Sin carga
    [0, 0], # Nodo 4: Sin carga
    [0, 0], # Nodo 5: Sin carga
    [0, -10_000e3/5], # Nodo 6: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 7: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 8: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 9: Carga de -2 kN en y
    [0, -10_000e3/5],  # Nodo 10: Carga de -2 kN en y
])

# Definir elementos (conectividad e índice de propiedades del material)
double_fink_roof_elements = np.array([
    [0, 0,  1],   # elemento 0  
    [1, 1,  2],   # elemento 1 
    [2, 2,  3],   # elemento 2 
    [3, 3,  4],   # elemento 3 
    [4, 4,  5],   # elemento 4 
    [5, 0,  6],   # elemento 5
    [6, 6,  7],   # elemento 6
    [7, 7,  8],   # elemento 7
    [8, 8,  9],   # elemento 8
    [9, 9,  10],   # elemento 9  
    [10, 10,  5],   # elemento 10  
    [11, 6,  1],   # elemento 11
    [12, 1,  7],   # elemento 12
    [13, 7,  2],   # elemento 13
    [14, 2,  8],   # elemento 14
    [15, 8,  3],   # elemento 15 
    [16, 3,  9],   # elemento 16
    [17, 9,  4],   # elemento 17
    [18, 4,  10],   # elemento 18
], dtype=int)

# Inicialización de secciones con el caso más costoso
double_fink_roof_sections = [0.02] * np.shape(double_fink_roof_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
double_fink_roof_modulus_of_elasticity = np.ones_like(double_fink_roof_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
double_fink_roof_materials = np.array([double_fink_roof_modulus_of_elasticity, double_fink_roof_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
double_fink_roof_data = {
    "nodes": double_fink_roof_nodes,
    "cons": double_fink_roof_cons,
    "elements": double_fink_roof_elements,
    "loads": double_fink_roof_loads,
    "mats": double_fink_roof_materials
}

"""
Compound Fink Roof Truss
"""
# Definir nodos (coordenadas de cada nodo en la cercha)
compund_fink_roof_nodes = np.array([
    [0,  0],  # Nodo 0
    [2,  0],  # Nodo 1
    [4,  0],  # Nodo 2
    [8,  0],  # Nodo 3
    [10,  0],  # Nodo 4
    [12, 0],  # Nodo 5
    [1.5, 0.75],  # Nodo 6
    [3,  2*0.75],  # Nodo 7
    [4.5,  3*0.75],  # Nodo 8
    [6,  4*0.75],  # Nodo 9
    [7.5,  3*0.75],  # Nodo 10
    [9.0,  2*0.75],  # Nodo 11
    [10.5,  0.75],  # Nodo 12
    [5,  1.5],  # Nodo 13
    [7,  1.5],  # Nodo 14
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
compund_fink_roof_cons = np.array([
    [0, -1], # Nodo 0
    [0, 0], # Nodo 1
    [0, 0], # Nodo 2
    [0, 0], # Nodo 3
    [0, 0], # Nodo 4
    [-1, -1], # Nodo 5
    [0, 0], # Nodo 6
    [0, 0], # Nodo 7
    [0, 0], # Nodo 8
    [0, 0], # Nodo 9
    [0, 0], # Nodo 10
    [0, 0], # Nodo 11
    [0, 0], # Nodo 12
    [0, 0], # Nodo 13
    [0, 0], # Nodo 14
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
compund_fink_roof_loads = np.array([
    [0, 0], # Nodo 0: Carga de -2 kN en y
    [0, 0], # Nodo 1: Sin carga
    [0, 0], # Nodo 2: Sin carga
    [0, 0], # Nodo 3: Sin carga
    [0, 0], # Nodo 4: Sin carga
    [0, 0], # Nodo 5: Sin carga
    [0, -10_000e3/7], # Nodo 6: Carga de -2 kN en y
    [0, -10_000e3/7], # Nodo 7: Carga de -2 kN en y
    [0, -10_000e3/7], # Nodo 8: Carga de -2 kN en y
    [0, -10_000e3/7], # Nodo 9: Carga de -2 kN en y
    [0, -10_000e3/7],  # Nodo 10: Carga de -2 kN en y
    [0, -10_000e3/7],  # Nodo 11: Carga de -2 kN en y
    [0, -10_000e3/7],  # Nodo 12: Carga de -2 kN en y
    [0, 0],  # Nodo 13: Carga de -2 kN en y
    [0, 0],  # Nodo 14: Carga de -2 kN en y
])

# Definir elementos (conectividad e índice de propiedades del material)
compund_fink_roof_elements = np.array([
    [0, 0,  1],   # elemento 0  
    [1, 1,  2],   # elemento 1 
    [2, 2,  3],   # elemento 2 
    [3, 3,  4],   # elemento 3 
    [4, 4,  5],   # elemento 4 
    [5, 0,  6],   # elemento 5
    [6, 6,  7],   # elemento 6
    [7, 7,  8],   # elemento 7
    [8, 8,  9],   # elemento 8
    [9, 9,  10],   # elemento 9  
    [10, 10,  11],   # elemento 10  
    [11, 11,  12],   # elemento 11 
    [12, 12, 5],   # elemento 12 
    [13, 1, 6],   # elemento 13 
    [14, 1, 7],   # elemento 14
    [15, 2, 7],   # elemento 15 
    [16, 7, 13],   # elemento 16
    [17, 13, 8],   # elemento 17
    [18, 13, 2],   # elemento 18
    [19, 13, 9],   # elemento 19
    [20, 9, 14],   # elemento 20
    [21, 14, 10],   # elemento 21
    [22, 14, 3],   # elemento 22
    [23, 14, 11],   # elemento 23
    [24, 11, 3],   # elemento 24
    [25, 4, 11],   # elemento 25
    [26, 4, 12],   # elemento 26
], dtype=int)

# Inicialización de secciones con el caso más costoso
compund_fink_roof_sections = [0.02] * np.shape(compund_fink_roof_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
compund_fink_roof_modulus_of_elasticity = np.ones_like(compund_fink_roof_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
compund_fink_roof_materials = np.array([compund_fink_roof_modulus_of_elasticity, compund_fink_roof_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
compund_fink_roof_data = {
    "nodes": compund_fink_roof_nodes,
    "cons": compund_fink_roof_cons,
    "elements": compund_fink_roof_elements,
    "loads": compund_fink_roof_loads,
    "mats": compund_fink_roof_materials
}

"""
Fan Roof Truss
"""
# Definir nodos (coordenadas de cada nodo en la cercha)
fan_roof_nodes = np.array([
    [0,  0],  # Nodo 0
    [4,  0],  # Nodo 1
    [8,  0],  # Nodo 2
    [12,  0],  # Nodo 3
    [2,  1],  # Nodo 4
    [4,  2],  # Nodo 5
    [6,  3],  # Nodo 6
    [8,  2],  # Nodo 7
    [10,  1],  # Nodo 8
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
fan_roof_cons = np.array([
    [0, -1], # Nodo 0
    [0, 0], # Nodo 1
    [0, 0], # Nodo 2
    [-1, -1], # Nodo 3
    [0, 0], # Nodo 4
    [0, 0], # Nodo 5
    [0, 0], # Nodo 6
    [0, 0], # Nodo 7
    [0, 0], # Nodo 8
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
fan_roof_loads = np.array([
    [0, 0], # Nodo 0: Carga de -2 kN en y
    [0, 0], # Nodo 1: Sin carga
    [0, 0], # Nodo 2: Sin carga
    [0, 0], # Nodo 3: Sin carga
    [0, -10_000e3/5], # Nodo 4: Sin carga
    [0, -10_000e3/5], # Nodo 5: Sin carga
    [0, -10_000e3/5], # Nodo 6: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 7: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 8: Carga de -2 kN en y
])

# Definir elementos (conectividad e índice de propiedades del material)
fan_roof_elements = np.array([
    [0, 0,  1],   # elemento 0  
    [1, 1,  2],   # elemento 1 
    [2, 2,  3],   # elemento 2 
    [3, 0,  4],   # elemento 3
    [4, 4,  5],   # elemento 4
    [5, 5,  6],   # elemento 5
    [6, 6,  7],   # elemento 6
    [7, 7,  8],   # elemento 7
    [8, 8,  3],   # elemento 8
    [9, 1,  4],   # elemento 9
    [10, 1,  5],   # elemento 10
    [11, 1,  6],   # elemento 11
    [12, 2,  6],   # elemento 12
    [13, 2,  7],   # elemento 13
    [14, 2,  8],   # elemento 14
    ], dtype=int)

# Inicialización de secciones con el caso más costoso
fan_roof_sections = [0.02] * np.shape(fan_roof_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
fan_roof_modulus_of_elasticity = np.ones_like(fan_roof_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
fan_roof_materials = np.array([fan_roof_modulus_of_elasticity, fan_roof_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
fan_roof_data = {
    "nodes": fan_roof_nodes,
    "cons": fan_roof_cons,
    "elements": fan_roof_elements,
    "loads": fan_roof_loads,
    "mats": fan_roof_materials
}


"""
Modified Queen Roof Truss
"""
# Definir nodos (coordenadas de cada nodo en la cercha)
modified_queen_roof_nodes = np.array([
    [0,  0],  # Nodo 0
    [3,  0],  # Nodo 1
    [6,  0],  # Nodo 2
    [9,  0],  # Nodo 3
    [12,  0],  # Nodo 4
    [2,  1],  # Nodo 5
    [4,  2],  # Nodo 6
    [6,  3],  # Nodo 7
    [8,  2],  # Nodo 8
    [10,  1],  # Nodo 9
])

# Definir restricciones (0 = libre, -1 = fijo en esa dirección)
modified_queen_roof_cons = np.array([
    [0, -1], # Nodo 0
    [0, 0], # Nodo 1
    [0, 0], # Nodo 2
    [0, 0], # Nodo 3
    [-1, -1], # Nodo 4
    [0, 0], # Nodo 5
    [0, 0], # Nodo 6
    [0, 0], # Nodo 7
    [0, 0], # Nodo 8
    [0, 0], # Nodo 9
], dtype=int)

# Definir cargas (fuerzas aplicadas en cada nodo en direcciones x e y)
modified_queen_roof_loads = np.array([
    [0, 0], # Nodo 0: Carga de -2 kN en y
    [0, 0], # Nodo 1: Sin carga
    [0, 0], # Nodo 2: Sin carga
    [0, 0], # Nodo 3: Sin carga
    [0, 0], # Nodo 4: Sin carga
    [0, -10_000e3/5], # Nodo 5: Sin carga
    [0, -10_000e3/5], # Nodo 6: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 7: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 8: Carga de -2 kN en y
    [0, -10_000e3/5], # Nodo 9
])

# Definir elementos (conectividad e índice de propiedades del material)
modified_queen_roof_elements = np.array([
    [0, 0,  1],   # elemento 0  
    [1, 1,  2],   # elemento 1 
    [2, 2,  3],   # elemento 2 
    [3, 3,  4],   # elemento 3
    [4, 0,  5],   # elemento 4
    [5, 5,  6],   # elemento 5
    [6, 6,  7],   # elemento 6
    [7, 7,  8],   # elemento 7
    [8, 8,  9],   # elemento 8
    [9, 9,  4],   # elemento 9
    [10, 1,  5],   # elemento 10
    [11, 1,  6],   # elemento 11
    [12, 2,  6],   # elemento 12
    [13, 2,  7],   # elemento 13
    [14, 2,  8],   # elemento 14
    [15, 3,  8],   # elemento 15   
    [16, 3,  9],   # elemento 16  
    ], dtype=int)

# Inicialización de secciones con el caso más costoso
modified_queen_roof_sections = [0.02] * np.shape(modified_queen_roof_elements)[0]  # Secciones iniciales
    
# Crear un array con los módulos de Young de los elementos
modified_queen_roof_modulus_of_elasticity = np.ones_like(modified_queen_roof_sections) * 200e9

# Crear un array con los módulos de Young y las secciones transversales
modified_queen_roof_materials = np.array([modified_queen_roof_modulus_of_elasticity, modified_queen_roof_sections]).T

# Preparar el diccionario de datos como se espera en la función de análisis
modified_queen_roof_data = {
    "nodes": modified_queen_roof_nodes,
    "cons": modified_queen_roof_cons,
    "elements": modified_queen_roof_elements,
    "loads": modified_queen_roof_loads,
    "mats": modified_queen_roof_materials
}
