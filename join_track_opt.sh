#!/bin/bash

# Define los arreglos con los valores predefinidos para valor1 y valor2
valores1=(0.5 0.5 0.9 0.9 0.3 0.3)
valores2=(0.03 0.01 0.03 0.01 0.03 0.01)

# Asegúrate de que NUM_EJECUCIONES no sea mayor que la longitud de los arreglos
NUM_EJECUCIONES=${#valores1[@]}

# Bucle para ejecutar script1.py
for (( i=0; i<NUM_EJECUCIONES; i++ ))
do
    valor1=${valores1[$i]}
    valor2=${valores2[$i]}

    # Construye el nombre del archivo de salida basado en los valores
    archivo_salida="./DATA/opt_params/all_ants_007_model_tr28_all_ants_augmented_split_tracks_fastReid_joined_thr${valor1}_dist${valor2}.txt"

    echo "Ejecución $((i+1)) de script1.py con valor1=$valor1 y valor2=$valor2"
    python3 join_tracks.py ./DATA/all_ants_007_best_model_all_ants_augmented_split_tracks.txt /home/usuaris/imatge/pol.serra.i.montes/TFG/AntTracking/DATA/all_ants_007_best_model_all_ants_augmented_split_tracks_fastReid.txt $archivo_salida --thr=$valor1 --dist=$valor2
    python3 scr_hota.py $archivo_salida
done

echo "Ejecuciones completadas."
