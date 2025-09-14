# SIA TP2 - Algoritmos Genéticos

Este proyecto implementa un **Algoritmo Genético (AG) para aproximar imágenes mediante triángulos**.  
El objetivo es generar una imagen lo más parecida posible a una imagen de entrada utilizando un número limitado de triángulos, mediante evolución de poblaciones, selección, cruza y mutación.  

## Prerrequisitos
- [Python](https://www.python.org/downloads/) instalado en el sistema.
- `pip` disponible en la terminal (`pip --version` para verificar).

## Construcción

Para construir el proyecto por completo y contar con el entorno necesario, ejecute de manera secuencial los siguientes comandos desde la raíz:

### Windows:

    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

### Linux/MacOS

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Ejecucion

```bash
python main.py [opciones]
```

## Opciones disponibles

### Imagen
- **`--image`** *(str, default: `images/blue_square.png`)*  
  Ruta a la imagen objetivo que se quiere aproximar.

- **`--size`** *(int int, default: `128 128`)*  
  Tamaño de la imagen en formato `(ancho alto)`.

---

### Población y evolución
- **`--population_size`** *(int, default: `40`)*  
  Número de individuos en cada generación.

- **`--generation_amount`** *(int, default: `100000`)*  
  Número máximo de generaciones que se ejecutará el algoritmo.

- **`--replacement_method`** *(str, default: `traditional`)*  
  Estrategia de reemplazo para formar la nueva generación:  
  - `traditional`: mezcla padres + hijos, se queda con los mejores.  
  - `youth_bias`: da preferencia a los hijos (mayor diversidad).

---

### Polígonos
- **`--max_polygons`** *(int, default: `40`)*  
  Máximo número de polígonos permitidos en cada individuo.

- **`--polygon_sides`** *(int, default: `3`)*  
  Número de lados de cada polígono (ej: 3 = triángulos, 4 = cuadrados).

---

### Mutación
- **`--mutation_method`** *(str, default: `uniform_multi_gen`)*  
  Estrategia de mutación:  
  - `gen`: muta un único polígono.  
  - `limited_multi_gen`: muta entre 1 y 10 polígonos aleatorios.  
  - `uniform_multi_gen`: cada polígono tiene probabilidad independiente de mutar.  
  - `complete`: muta todos los polígonos.  

- **`--mutate_structure`** *(flag, default: `False`)*  
  Permite agregar nuevos polígonos como parte de la mutación (siempre que no supere `max_polygons`).

---

### Selección
- **`--selection_method`** *(str, default: `boltzmann`)*  
  Estrategia de selección de padres:  
  - `elite`  
  - `roulette` → selección por ruleta proporcional al fitness.  
  - `universal` → selección universal estocástica.  
  - `boltzmann` → probabilidad suavizada con temperatura.  
  - `tournament_deterministic`  
  - `tournament_probabilistic`  
  - `ranking`

---

### Cruce (crossover)
- **`--crossover`** *(str, default: `uniform_crossover`)*  
  Estrategia de cruce:  
  - `one_point`: un punto de corte.  
  - `two_point`: dos puntos de corte.  
  - `uniform_crossover`: cada gen tiene probabilidad de venir de uno u otro padre.  

---

### Error objetivo
- **`--target_error`** *(float, default: `0.01`)*  
  Umbral de error mínimo. Si el mejor individuo alcanza un error menor o igual, el algoritmo se detiene aunque no se hayan agotado todas las generaciones.
