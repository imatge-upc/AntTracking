
import cv2
from docopt import docopt
import numpy as np
from openvino.runtime import Core
import os
import sys

from ceab_ants.detection.process_video import process_video


QUEUE_GET_TIMEOUT = 2 # seconds
TQDM_INTERVAL = 10 # iterations


def preprocess_frames(frame, input_shape):
    # Ultralytics authomatically reshape input, openVINO needs the input preprocessed

    # TODO: pad into square before
    resized_frame = cv2.resize(frame, (input_shape[0], input_shape[1]))  # (W, H)
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    processed_frame = np.transpose(normalized_frame, (2, 0, 1)).astype(np.float32)  # (C, H, W)
    return [processed_frame]

def extract_obboxes(outputs, initial_frame, original_width=4000, original_height=3000, input_width=2240, input_height=2240, conf_threshold=0.5):

    processed_results = []
    frame_index = initial_frame

    for output in outputs:
        xywhr = output[:, (0, 1, 2, 3, 6)]  # Extract x, y, w, h, r
        confidences = output[:, 4].reshape(-1, 1)  # Extract scores

        mask = confidences.flatten() > conf_threshold

        xywhr = xywhr[mask]
        confidences = confidences[mask]

        xywhr[:, 4] = np.rad2deg(xywhr[:, 4])  # Convert radians to degrees

        scale_w = original_width / input_width
        scale_h = original_height / input_height

        xywhr[:, 0] *= scale_w
        xywhr[:, 1] *= scale_h
        xywhr[:, 2] *= scale_w
        xywhr[:, 3] *= scale_h

        obboxes = np.concatenate((xywhr, confidences), axis=1)
        current_frame = frame_index
        processed_results.append((current_frame, obboxes))

        frame_index += 1
    
    last_frame_batch = frame_index
    return processed_results, last_frame_batch


def main(video_source, model_path, output, max_preloaded, batch_size, original_w=4000, original_h=3000, model_w=2240, model_h=2240, initial_frame=0, num_frames=-1):

    class VINOModel():
        def __init__(self):
            core = Core()
            model = core.read_model(model=model_path) # .xml
            print("Compiling model")
            self.model = core.compile_model(model=model, device_name="CPU")
            print("Model compiled")
            self.infer_request = self.model.create_infer_request()
            self.input_blob = self.model.input(0)
            # input_shape = input_blob.shape  # [N, C, H, W]

        def apply(self, batch):
            input_data = np.stack(batch) # TODO: can I have a batch size smaller than the input_blob.shape? If not, add zeros frames
            result = self.infer_request.infer({self.input_blob: input_data})
            outputs = result[self.model.output(0)]
            return outputs
    
    build_model = VINOModel
    apply_model = lambda model, batch : model.apply(batch)

    preprocess_func = lambda frame : preprocess_frames(frame, (model_w, model_h))
    postprocess_func = lambda results, initial_frame : extract_obboxes(results, initial_frame, original_width=original_w, original_height=original_h, input_width=model_w, input_height=model_h)
    
    process_video(video_source, output, build_model, apply_model, preprocess_func, postprocess_func, max_preloaded, batch_size, batch_size, initial_frame=initial_frame, num_frames=num_frames, timeout_get=QUEUE_GET_TIMEOUT, tqdm_interval=TQDM_INTERVAL)    


DOCTEXT = """
Usage:
  ant_detection_yolo_resize_cpu.py <video_source> <output> <model_path> [--max_preloaded=<n>] [--batch_size=<b>]
  ant_detection_yolo_resize_cpu.py -h | --help

Options:
  --max_preloaded=<n>  Maximum number of preloaded frames [default: 4].
  --batch_size=<b>  Number of frames to batch for inference [default: 1].

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_source = args["<video_source>"]
    output = args["<output>"]
    model_path = args["<model_path>"]
    max_frames = int(args["--max_preloaded"])
    batch_size = int(args["--batch_size"])

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    main(video_source, model_path, output, max_frames, batch_size)
