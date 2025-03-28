
from contextlib import contextmanager
import queue
from threading import Thread
from tqdm import tqdm


QUEUE_GET_TIMEOUT = 0.5

class OBBPostProcessWriter():
    """
    For working with only 1 video, for multiple videos, a thread that separates by video id is needed
    """

    def __init__(self, output_queue, metadata_queue, stop_event, output_filename, postprocess_func=None, tqdm_interval=1):
        self.metadata_queue = metadata_queue # Metadata is a tuple with at least (video id, frame number, number of model outputs)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.output_filename = output_filename
        self.postprocess_func = postprocess_func if postprocess_func else (lambda x, metadata : ([(metadata[1], x[i]) for i in range(x)]))

        self.tqdm_interval = tqdm_interval

    def start(self):
        with open(self.output_filename, 'w') as output_file:
            with tqdm(mininterval=self.tqdm_interval, maxinterval=self.tqdm_interval, leave=False) as pbar:
                while not self.stop_event.is_set() or not self.output_queue.empty():
                    try:
                        metadata = self.metadata_queue.get(timeout=QUEUE_GET_TIMEOUT)
                        _, _, num_outputs = metadata[:3]

                        batch = []
                        for _ in range(num_outputs):
                            while True:
                                try:
                                    elem = self.output_queue.get(timeout=QUEUE_GET_TIMEOUT)
                                    batch.append(elem)
                                    break
                                except queue.Empty:
                                    if self.stop_event.is_set(): # If not, self.output_queue.get() may be stuck on ctrl+C
                                        pbar.clear()
                                        break
                                    continue
            
                        processed_batch = self.postprocess_func(batch, metadata)
                        self.write_output(processed_batch, output_file)

                        if not self.stop_event.is_set():
                            pbar.update(1)
                    except queue.Empty:
                        continue
            
                output_file.flush()

    def write_output(self, batch, output_file):
        frame, obboxes = batch
        if len(obboxes) > 0:
            MOTDet_line = lambda fr, obbox: (
                f'{fr:.0f},-1,{obbox[0]:.5f},{obbox[1]:.5f},'
                f'{obbox[2]:.5f},{obbox[3]:.5f},{obbox[5]:.5f},-1,-1,-1,{obbox[4]:.1f}'
            )
            detection_text = '\n'.join([MOTDet_line(frame, obbox) for obbox in obboxes])
            print(detection_text, end='\n', file=output_file)

@contextmanager
def OBBPostProcessWriterContext(output_queue, metadata_queue, stop_event, output_filename, postprocess_func=None, tqdm_interval=1):
    obb_writer = OBBPostProcessWriter(output_queue, metadata_queue, stop_event, output_filename, postprocess_func=postprocess_func, tqdm_interval=tqdm_interval)
    obb_writer_thread = Thread(target=obb_writer.start, daemon=True)
    try:
        obb_writer_thread.start()
        yield
    finally:
        stop_event.set()
        obb_writer_thread.join()
