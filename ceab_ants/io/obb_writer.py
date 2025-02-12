
from contextlib import contextmanager
import queue
from threading import Thread


QUEUE_GET_TIMEOUT = 0.5

class OBBPostProcessWriter():

    def __init__(self, initial_frame, output_queue, stop_event, output_filename, postprocess_func=None):
        self.frame = initial_frame
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.output_filename = output_filename
        self.postprocess_func = postprocess_func if postprocess_func else (lambda x, fr : ([(fr + i, x[i]) for i in range(x)], fr + len(x)))

    def start(self):
        with open(self.output_filename, 'w') as output_file:
            while not self.stop_event.is_set() or not self.output_queue.empty():
                try:
                    batch = self.output_queue.get(timeout=QUEUE_GET_TIMEOUT)
                    processed_batch, self.frame = self.postprocess_func(batch, self.frame)
                    self.write_output(processed_batch, output_file)
                except queue.Empty:
                    continue
        
            output_file.flush()

    def write_output(self, batch, output_file):
        for frame, obboxes in batch:
            if len(obboxes) > 0:
                MOTDet_line = lambda fr, obbox: (
                    f'{fr:.0f},-1,{obbox[0]:.5f},{obbox[1]:.5f},'
                    f'{obbox[2]:.5f},{obbox[3]:.5f},{obbox[5]:.5f},-1,-1,-1,{obbox[4]:.1f}'
                )
                detection_text = '\n'.join([MOTDet_line(frame, obbox) for obbox in obboxes])
                print(detection_text, end='\n', file=output_file)

@contextmanager
def OBBPostProcessWriterContext(initial_frame, output_queue, stop_event, output_filename, postprocess_func=None):
    obb_writer = OBBPostProcessWriter(initial_frame, output_queue, stop_event, output_filename, postprocess_func=postprocess_func)
    obb_writer_thread = Thread(target=obb_writer.start, daemon=True)
    try:
        obb_writer_thread.start()
        yield
    finally:
        stop_event.set()
        obb_writer_thread.join()
