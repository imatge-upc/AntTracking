
from docopt import docopt
from openvino.runtime import Core, serialize
import os
import shutil
import sys
from ultralytics import YOLO


def export_model_to_openvino(model_path, output_dir):
    model = YOLO(model_path)
    model.export(format="openvino", nms=True) # dynamic=False

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    openvino_output_folder = os.path.join(os.path.dirname(model_path), f"{model_name}_openvino_model")

    for item in os.listdir(openvino_output_folder):
        shutil.move(os.path.join(openvino_output_folder, item), output_dir)
    os.rmdir(openvino_output_folder)


    print(f"Model exported to OpenVINO format in directory: {output_dir}")
    return os.path.join(output_dir, f"{model_name}.xml")


DOCTEXT = f"""
Usage:
  export_yolo_to_openvino.py <model_path> <output_dir>
  export_yolo_to_openvino.py -h | --help
"""

if __name__ == "__main__":
    args = docopt(DOCTEXT, sys.argv[1:], help=True)

    model_path = args["<model_path>"]
    output_dir = args["<output_dir>"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    model_xml_path = export_model_to_openvino(model_path, output_dir)
