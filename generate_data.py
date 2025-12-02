import numpy as np
from sklearn.datasets import make_classification
import os

# tamaños de dataset a generar
sizes = [1000, 2000, 4000, 8000, 16000, 20000, 30000, 40000]

output_dir = "datasets"

if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
else:
    os.mkdir(output_dir)

print("\n================ GENERANDO DATASETS =================\n")

for n in sizes:
    print(f"- Generando dataset con n = {n} muestras...")

    X, y = make_classification(
        n_samples=n,
        n_features=64,
        n_informative=30,
        n_redundant=10,
        n_repeated=0,
        n_classes=10,
        n_clusters_per_class=1,
        class_sep=2.5,      
        flip_y=0.0,         
        random_state=42
    )

    filename = f"{output_dir}/dataset_{n}.npz"
    np.savez(filename, X=X, y=y)
    print(f"  > guardado en {filename}")

print("\n================ LISTO (✓) =================\n")
print("Datasets creados correctamente y reemplazados.")