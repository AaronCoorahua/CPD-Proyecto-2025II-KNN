from mpi4py import MPI
import numpy as np
import sys
from collections import Counter


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


# ============================================
#  MPI INIT
# ============================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================
#  ARGUMENTOS
# ============================================
if len(sys.argv) < 2:
    if rank == 0:
        print("Uso: mpirun -np <p> python knn_parallel.py <dataset.npz> [k]")
    sys.exit(0)

dataset_name = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dataset_path = f"dataset/{dataset_name}"

# ============================================
#  SOLO RANK 0 LEE
# ============================================
if rank == 0:
    data = np.load(dataset_path)
    X = data["X"]
    y = data["y"]

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    split = int(0.8 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    print("\n===== DATASET INFO =====")
    print("Dataset:", dataset_name)
    print("Train:", len(X_train), "| Test:", len(X_test))
    print("========================\n")

else:
    X_train = None
    y_train = None
    X_test = None
    y_test = None


# ============================================
#  MEDICIÓN TOTAL 
# ============================================
comm.Barrier()
t_start = MPI.Wtime()


# ============================================
#  BROADCAST TRAIN DATA 
# ============================================
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)


# ============================================
#  SPLIT & SCATTER TEST DATA  
# ============================================
local_X_test = comm.scatter(
    np.array_split(X_test, size) if rank == 0 else None,
    root=0
)
local_y_test = comm.scatter(
    np.array_split(y_test, size) if rank == 0 else None,
    root=0
)


# ============================================
#  CÓMPUTO LOCAL
# ============================================
t_compute_start = MPI.Wtime()
local_y_pred = [knn_predict(x, X_train, y_train, k) for x in local_X_test]
t_compute_end = MPI.Wtime()

local_compute = t_compute_end - t_compute_start


# ============================================
#  GATHER RESULTS 
# ============================================
gathered_pred = comm.gather(local_y_pred, root=0)
gathered_compute = comm.gather(local_compute, root=0)


comm.Barrier()
t_end = MPI.Wtime()

local_total = t_end - t_start
local_comm = local_total - local_compute


# ============================================
#  ROOT PRINTS RESULTS
# ============================================
gathered_total = comm.gather(local_total, root=0)
gathered_comm = comm.gather(local_comm, root=0)

if rank == 0:

    y_pred = np.concatenate(gathered_pred)
    accuracy = np.mean(y_pred == y_test)

    t_total = max(gathered_total)
    t_compute = max(gathered_compute)
    t_comm = max(gathered_comm)

    print("\n=============== RESULTADOS MPI KNN ================")
    print(f"dataset = {dataset_name}")
    print(f"n = {len(X_test)} | p = {size} | k = {k}")
    print(f"t_total   = {t_total:.6f} sec")
    print(f"t_compute = {t_compute:.6f} sec")
    print(f"t_comm    = {t_comm:.6f} sec")
    print(f"accuracy  = {accuracy:.4f}")
    print("===================================================\n")
