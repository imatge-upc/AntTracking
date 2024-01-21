"""Mi Script
    Entrada: name to json archive

Usage:
  mi_script.py (-h | --help)
  mi_script.py paths <entrada>

Options:
  -h --help       Muestra esta ayuda.
  --opcional=<valor>  Valor opcional.
"""

import json
from docopt import docopt
import os
# Ruta al archivo JSON
root_dir = '/home/usuaris/imatge/pol.serra.i.montes/runs/detect'

def json_to_txt(data):
    # Crear un diccionario para almacenar datos por image_id
    data_by_image_id = {}

    # Organizar los datos por image_id
    for entry in data:
        image_id = entry['image_id']
        category_id = entry['category_id']
        bbox = entry['bbox']
        score = entry['score']

        # Verificar si ya existe la clave image_id en el diccionario
        if image_id not in data_by_image_id:
            data_by_image_id[image_id] = []

        # Agregar la información al diccionario
        data_by_image_id[image_id].append({'category_id': category_id, 'bbox': bbox, 'score': score})

    # Crear un archivo de texto para cada image_id
    for image_id, entries in data_by_image_id.items():
        output_filename = f"{image_id}.txt"

        with open(os.path.join(root_dir,arguments['<entrada>'].split('/')[0],'preds',output_filename), 'w') as file:
            # Escribir la información en el archivo
            for entry in entries:
                norm_pos = []
                file.write(f"{entry['category_id']} ") #{entry['bbox']}\n)
                for pos in entry['bbox']:
                    norm_pos.append(float(pos)/640)
                norm_pos_yolo = norm_pos
                norm_pos_yolo[0]=norm_pos[0]+(norm_pos[2]/2)
                norm_pos_yolo[1]=norm_pos[1]+(norm_pos[3]/2)
                norm_pos_yolo[2]=norm_pos[2]
                norm_pos_yolo[3]=norm_pos[3]
                for norm in norm_pos_yolo:
                    file.write(f"{norm} ")
                file.write("\n")

if __name__ == "__main__":

    arguments = docopt(__doc__)
    os.makedirs(os.path.join(root_dir,arguments['<entrada>'].split('/')[0],"preds"), exist_ok=True)
    with open(os.path.join(root_dir,arguments['<entrada>']), 'r') as file:
        data = json.load(file)
    json_to_txt(data)
