
import queue
from threading import Event, Lock
from tqdm import tqdm

from ceab_ants.io.obb_writer import OBBPostProcessWriterContext
from ceab_ants.io.queued_frame_loading import QueuedVideoLoaderContext


QUEUE_GET_TIMEOUT = 2 # seconds


def process_video(video_source, output, build_model, apply_model, preprocess_func, postprocess_func, queue_size=8, batch_size=4, min_batch_size=None, initial_frame=0, num_frames=-1, timeout_get=QUEUE_GET_TIMEOUT, tqdm_interval=10):

    min_batch_size = min_batch_size or batch_size

    model = build_model()

    output_queue = queue.Queue()
    stop_event_write = Event()
    with OBBPostProcessWriterContext(
        initial_frame=initial_frame, 
        output_queue=output_queue, 
        stop_event=stop_event_write, 
        output_filename=output, 
        postprocess_func=postprocess_func
    ):
    
        frame_queue = queue.Queue(maxsize=queue_size)
        queue_lock = Lock()
        stop_event_load = Event()
        with QueuedVideoLoaderContext(
            video_source, 
            video_id=0, # This object just process 1 video so no video_id needed
            frame_queue=frame_queue,
            queue_lock=queue_lock, 
            stop_event=stop_event_load, 
            initial_frame=initial_frame,
            preprocess_func=preprocess_func
        ):

            def generator():
                fr = initial_frame
                while (num_frames <= 0 or fr <= (initial_frame + num_frames)) and not (stop_event_load.is_set() and frame_queue.empty()):
                    try:
                        _, frame = frame_queue.get(timeout=timeout_get) # This object just process 1 video so no video_id needed
                        yield frame
                        fr += 1
                    except queue.Empty:
                        if stop_event_load.is_set():
                            break
            
            batch = []
            for frame in tqdm(generator(), mininterval=tqdm_interval, maxinterval=tqdm_interval):

                batch.append(frame)

                if (len(batch) < batch_size) and (not frame_queue.empty() or (len(batch) < min_batch_size)):
                    continue
                    
                if batch:
                    results = apply_model(model, batch)

                    output_queue.put(results)
                    batch = []
            
            if batch:
                results = apply_model(model, batch)

                output_queue.put(results)
                batch = []
