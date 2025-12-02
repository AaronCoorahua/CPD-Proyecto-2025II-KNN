# Paralelización del Algoritmo KNN con MPI

**Autores:** Ruben Aaron Coorahua Peña, Isaac Vera Romero, Angel Ulises Tito Berrocal  
**Institución:** UTEC — Universidad de Ingeniería y Tecnología, Lima, Perú  
**Fecha:** Noviembre 2025

## Resumen

Este proyecto presenta una implementación paralela del algoritmo de clasificación **k-Nearest Neighbors (KNN)** utilizando el estándar **Message Passing Interface (MPI)**. El objetivo principal es reducir significativamente el tiempo de ejecución respecto a su versión secuencial mediante estrategias de computación distribuida.

La estrategia propuesta distribuye el conjunto de datos de prueba entre múltiples procesos, permitiendo que el cálculo de distancias (la parte más costosa computacionalmente) se realice de manera local y paralela. Posteriormente, los resultados se consolidan mediante operaciones colectivas de comunicación.

## Estructura del Repositorio

El proyecto contiene los siguientes archivos principales:

*   `knn-secuencial.py`: Implementación secuencial del algoritmo KNN (Línea base).
*   `knn-paralelo.py`: Implementación paralela optimizada utilizando `mpi4py` (Corresponde a la Versión C del informe).
*   `generate_data.py`: Script para generar datasets sintéticos utilizando `scikit-learn`.
*   `run_knn.sh`: Script para la ejecución automatizada de experimentos.
*   `datasets/`: Directorio que almacena los conjuntos de datos en formato `.npz`.

## Requisitos Previos

Para ejecutar este proyecto, se requiere un entorno con soporte para MPI y Python.

*   **Sistema Operativo:** Linux / Windows (con soporte MPI)
*   **Lenguaje:** Python 3.x
*   **Librerías:**
    *   `mpi4py`
    *   `numpy`
    *   `scikit-learn`
*   **Implementación MPI:** OpenMPI, MPICH o MS-MPI.

Instalación de dependencias de Python:
```bash
pip install mpi4py numpy scikit-learn
```

## Metodología de Paralelización

El diseño paralelo se basa en el modelo **PRAM** y la metodología **PCAM** de Foster, aprovechando que la clasificación de cada muestra es independiente.

1.  **Carga y Preprocesamiento (Root):** El proceso raíz carga el dataset, lo baraja y lo divide en entrenamiento (80%) y prueba (20%).
2.  **Broadcast (`MPI_Bcast`):** Se difunde el conjunto de entrenamiento completo ($X_{train}, y_{train}$) a todos los procesos.
3.  **Scatter (`MPI_Scatter`):** Se divide el conjunto de prueba ($X_{test}$) en $p$ bloques y se distribuye uno a cada proceso.
4.  **Cómputo Local:** Cada proceso calcula las distancias euclidianas y determina los $k$ vecinos más cercanos para su subconjunto de datos.
5.  **Gather (`MPI_Gather`):** El proceso raíz recolecta las predicciones parciales y reconstruye el vector final de resultados.

## Instrucciones de Uso

### 1. Generación de Datasets
Antes de ejecutar los algoritmos, genere los datos sintéticos:
```bash
python generate_data.py
```
Esto creará archivos `.npz` en la carpeta `datasets/` con diferentes tamaños de muestras (ej. 1000, 2000, ..., 40000).

### 2. Ejecución Secuencial
Para obtener una línea base de rendimiento:
```bash
python knn-secuencial.py
```

### 3. Ejecución Paralela
Para ejecutar la versión paralela, utilice el comando `mpirun` o `mpiexec` especificando el número de procesos (`-np`).

**Sintaxis:**
```bash
mpirun -np <num_procesos> python knn-paralelo.py <archivo_dataset> <k>
```

**Ejemplo:** Ejecutar con 4 procesos sobre el dataset de 16,000 muestras buscando 3 vecinos:
```bash
mpirun -np 4 python knn-paralelo.py dataset_16000.npz 3
```

## Resultados y Rendimiento

Los experimentos realizados demuestran:

*   **Speedup:** Se alcanza una aceleración casi lineal hasta aproximadamente 16-20 procesos.
*   **Eficiencia:** Alta eficiencia en escalabilidad fuerte para datasets grandes.
*   **Exactitud:** La versión paralela mantiene exactamente la misma precisión (accuracy) que la versión secuencial, ya que el algoritmo es determinista y opera sobre los mismos datos.
*   **Comunicación:** El overhead de comunicación se mantiene bajo y estable, permitiendo que el tiempo de cómputo domine la ejecución en la mayoría de los casos.

## Referencias

1.  MPI for Python Documentation (mpi4py v3.1.6).
2.  Scikit-Learn Datasets Documentation.
3.  Wang, Y., et al. "A split–merge clustering algorithm based on the k nearest neighbor graph".
4.  PyData London, "MPI for Data Science - Parallel Processing with mpi4py".