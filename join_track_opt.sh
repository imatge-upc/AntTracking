#!/bin/bash

# Define los arreglos con los valores predefinidos para valor1 y valor2
thr_eu_dist=(0.5 0.5 0.9 0.9 0.3 0.3)
thr_sp_dist=(0.03 0.01 0.03 0.01 0.03 0.01)

# Asegúrate de que NUM_EJECUCIONES no sea mayor que la longitud de los arreglos
NUM_EJECUCIONES=${#thr_eu_dist[@]}

# Bucle para ejecutar script1.py
for (( i=0; i<NUM_EJECUCIONES; i++ ))
do
    valor1=${thr_eu_dist[$i]}
    valor2=${thr_sp_dist[$i]}

    # Construye el nombre del archivo de salida basado en los valores
    archivo_salida="./DATA/opt_params/all_ants_007_model_tr28_all_ants_augmented_split_tracks_fastReid_joined_thr${valor1}_dist${valor2}.txt"

    echo "Execution number $((i+1)) join_tracks.py with max_euclidean_dist=$valor1 and max_normalized_spatial_dist=$valor2"
    python3 join_tracks.py ./DATA/all_ants_007_best_model_all_ants_augmented_split_tracks.txt /home/usuaris/imatge/pol.serra.i.montes/TFG/AntTracking/DATA/all_ants_007_best_model_all_ants_augmented_split_tracks_fastReid.txt $archivo_salida --thr=$valor1 --dist=$valor2
    python3 scr_hota.py $archivo_salida
done

echo "Fully executions completed."
