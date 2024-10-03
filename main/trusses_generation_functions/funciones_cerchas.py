import numpy as np

def crear_warren_bridge_data(largo=24.0, alto=6, n=20, seccion=0.02, young=200e9, carga_total=400e3):
    # Crear lista para almacenar los nodos
    warren_bridge_nodes = []

    # Nodos inferiores (sobre el eje horizontal)
    for i in range(n + 1):
        warren_bridge_nodes.append([i * (largo / n), 0.0])

    # Nodos superiores (desplazados a mitad del paso y a la altura definida)
    for i in range(n):
        warren_bridge_nodes.append([i * (largo / n) + (largo / (2 * n)), alto])

    # Convertir lista de nodos a un array de numpy
    warren_bridge_nodes = np.array(warren_bridge_nodes)

    # Definir elementos (conectividad e índice de propiedades del material)
    warren_bridge_elements = []

    # Chords inferiores: conectan nodos consecutivos de la parte inferior
    for i in range(n):
        warren_bridge_elements.append([i, i, i + 1])

    # Diagonales: conectan nodos inferiores y superiores alternadamente
    for i in range(1, n + 1):
        # Diagonal ascendente
        warren_bridge_elements.append([n + i, i - 1, n + i])  
        # Diagonal descendente
        warren_bridge_elements.append([n + i + 1, n + i, i])

    # Chords superiores: conectan nodos consecutivos de la parte superior
    for i in range(2, n + 1):
        warren_bridge_elements.append([n + n + i, n + i - 1, n + i])

    # Convertir la lista de elementos a un array de numpy
    warren_bridge_elements = np.array(warren_bridge_elements)

    # Crear el arreglo de restricciones lleno de ceros con la misma forma de los nodos
    warren_bridge_cons = np.zeros((warren_bridge_nodes.shape[0], 2), dtype=int)

    # Modificar restricciones específicas según el problema
    warren_bridge_cons[0] = [0, -1]   # Nodo 0: Fijo en y (Soporte izquierdo)
    warren_bridge_cons[n] = [-1, -1]  # Nodo n: Fijo en x e y (Soporte derecho)

    # Crear el arreglo de fuerzas lleno de ceros con la misma forma de los nodos
    warren_bridge_loads = np.zeros((warren_bridge_nodes.shape[0], 2), dtype=float)

    # Aplicar fuerzas verticales de -carga_total/n en los nodos inferiores (excepto los extremos)
    for i in range(1, n):  # Desde el nodo 1 hasta el nodo n-1 (excluyendo extremos)
        warren_bridge_loads[i, 1] = -carga_total / n  # Fuerza en la dirección y

    # Inicialización de secciones con el caso más costoso
    warren_bridge_sections = [seccion] * np.shape(warren_bridge_elements)[0]  # Secciones iniciales
    
    # Crear un array con los módulos de Young de los elementos
    warren_bridge_modulus_of_elasticity = np.ones_like(warren_bridge_sections) * young

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

    return warren_bridge_data
