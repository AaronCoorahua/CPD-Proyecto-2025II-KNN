#!/bin/bash

SCRIPT="knn_large.py"                  # nombre del archivo python
OUTPUT="results_knn.csv"               # archivo CSV de salida

echo "p,dataset,n,avg_t_total,avg_t_compute,avg_t_comm,avg_accuracy" > $OUTPUT

DATASETS=("dataset_1000.npz" "dataset_2000.npz" "dataset_4000.npz" "dataset_8000.npz" "dataset_16000.npz" "dataset_20000.npz" "dataset_30000.npz" "dataset_40000.npz")
PROCS=(32 24 20 16 8 4 2 1)
REPS=1

for dataset in "${DATASETS[@]}"
do
    for p in "${PROCS[@]}"
    do
        sum_total=0
        sum_compute=0
        sum_comm=0
        sum_acc=0
        N=0

        echo ""
        echo "==============================================="
        echo " DATASET = $dataset | p = $p | $REPS reps"
        echo "==============================================="

        for i in $(seq 1 $REPS)
        do
            echo "[dataset=$dataset | p=$p] iteración $i/$REPS"

            mpirun -np $p python3 $SCRIPT "$dataset" 3 \
            | grep -E "dataset =|n =|t_total|t_compute|t_comm|accuracy" > tmp_run.log

            N=$(grep "n ="            tmp_run.log | awk '{print $3}')
            TTOTAL=$(grep "t_total"   tmp_run.log | awk '{print $3}')
            TCOMPUTE=$(grep "t_compute" tmp_run.log | awk '{print $3}')
            TCOMM=$(grep "t_comm"     tmp_run.log | awk '{print $3}')
            ACC=$(grep "accuracy"     tmp_run.log | awk '{print $3}')

            sum_total=$(echo "$sum_total + $TTOTAL" | bc -l)
            sum_compute=$(echo "$sum_compute + $TCOMPUTE" | bc -l)
            sum_comm=$(echo "$sum_comm + $TCOMM" | bc -l)
            sum_acc=$(echo "$sum_acc + $ACC" | bc -l)
        done

        avg_total=$(echo "scale=6; $sum_total / $REPS" | bc -l)
        avg_compute=$(echo "scale=6; $sum_compute / $REPS" | bc -l)
        avg_comm=$(echo "scale=6; $sum_comm / $REPS" | bc -l)
        avg_acc=$(echo "scale=6; $sum_acc / $REPS" | bc -l)

        printf "%d,%s,%d,%.6f,%.6f,%.6f,%.4f\n" \
        $p $dataset $N $avg_total $avg_compute $avg_comm $avg_acc >> $OUTPUT

        rm tmp_run.log
    done
done

echo ""
echo "================ FIN DE EXPERIMENTOS ================"
echo "Resultados guardados en $OUTPUT"