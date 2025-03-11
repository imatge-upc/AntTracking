
import numpy as np
import subprocess
import sys
import warnings


MAXROWS = 5000000


def loadtxt(seq_path, skiprows, max_rows):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq_dets = np.loadtxt(seq_path, delimiter=',', skiprows=skiprows, max_rows=max_rows, comments=None)
    return seq_dets

class MOTLoader():
    
    def __init__(self, seq_path=None, min_score=0.1, first_frame=1, max_rows=MAXROWS, verbose=False):

        self.seq_path = seq_path
        self.skiprows = 0
        self.max_rows = max_rows
        self.seq_dets = loadtxt(self.seq_path, self.skiprows, self.max_rows)
        self.buf_dets = self.seq_dets.copy()

        self.min_score = min_score
        self.first_frame = first_frame
        
        self.current_frame = first_frame

        self.verbose = verbose
        self._last_frame = None
        self._last_row = None
    
    def reset(self):
        self.current_frame = self.first_frame
        self.skiprows = 0
        self._last_frame = None
        self._last_row = None
        self.seq_dets = loadtxt(self.seq_path, self.skiprows, self.max_rows)

    @property
    def last_frame(self):

        if self._last_frame : return self._last_frame
        self._compute_max_frame_row()
        return self._last_frame
    
    @property
    def system_last_frame(self):

        result = subprocess.run(['tail',  '-1', self.seq_path], stdout=subprocess.PIPE)
        result = subprocess.run(['cut', '-d,', '-f1'], input=result.stdout, stdout=subprocess.PIPE)
        
        return int(result.stdout)
    
    @property
    def last_row(self):

        if self._last_row : return self._last_row
        self._compute_max_frame_row()
        return self._last_row
    
    def _compute_max_frame_row(self):

        skiprows = 0
        seq_dets = loadtxt(self.seq_path, skiprows, self.max_rows)

        self._last_row = 0
        while seq_dets.size != 0:
            self._last_frame = int(seq_dets[:, 0].max())
            self._last_row += len(seq_dets)
            if self.verbose : print(f"Reading row {skiprows + self.max_rows} for last_frame")
            skiprows += self.max_rows
            seq_dets = loadtxt(self.seq_path, skiprows, self.max_rows)

    def __getitem__(self, key):
        dets = self.buf_dets[self.buf_dets[:, 0] == key]
        dets = dets[dets[:, -1] >= self.min_score] # filter out low score detections
        return dets
    
    def __call__(self, frame, override=False):

        self.current_frame = frame if override else self.current_frame

        if self.verbose and ((self.current_frame - 1) % 500 == 0) : print (f'Processing frame {self.current_frame - 1} / {self.last_frame}', file=sys.stderr)

        if self.current_frame < int(self.buf_dets[:, 0].min()):
            current_frame = self.current_frame
            self.reset()
            self.current_frame = current_frame
        
        if self.current_frame >= int(self.buf_dets[:, 0].max()):
            while int(self.seq_dets[:, 0].max()) <= self.current_frame:
                #print(f"Reading row {self.skiprows + self.max_rows}")
                self.skiprows += self.max_rows
                self.seq_dets = loadtxt(self.seq_path, self.skiprows, self.max_rows)
                if self.seq_dets.size == 0:
                    break

                self.buf_dets = np.vstack([self.buf_dets[self.buf_dets[:, 0] == self.current_frame, :], self.seq_dets.copy()])
                
        dets = self[self.current_frame]
        self.current_frame += 1

        return dets

class PrecomputedMOTDetector(MOTLoader):
    def __init__(self, seq_path=None, min_score=0.1, first_frame=1, max_rows=MAXROWS, verbose=False):
        super().__init__(seq_path, min_score, first_frame, max_rows, verbose)

    def __getitem__(self, key):
        dets = self.buf_dets[self.buf_dets[:, 0] == key, 2:7]
        dets[:, 2:4] += dets[:, 0:2] # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        dets = dets[dets[:, -1] >= self.min_score] # filter out low score detections

        return dets.reshape(-1, 5) # np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)

class PrecomputedOMOTDetector(MOTLoader):

    def __init__(self, seq_path=None, min_score=0.1, first_frame=1, max_rows=MAXROWS, verbose=False):
        super().__init__(seq_path, min_score, first_frame, max_rows, verbose)    

    def __getitem__(self, key):
        dets = self.buf_dets[self.buf_dets[:, 0] == key, :][:, [2, 3, 4, 5, 10, 6]]

        dets = dets[dets[:, -1] >= self.min_score] # filter out low score detections

        return dets.reshape(-1, 6) # np.array([[x, y, w, h, a, score], ...]).reshape(N, 6)
