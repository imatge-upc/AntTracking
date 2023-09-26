# "track to track" association problem
# TODO: Test it

import numpy as np


class MOTPostptocessReID():

    def __init__(self, apparence_aggregation_model, distance_funct, th=0.5):
        self.aggregation = apparence_aggregation_model
        self.distance_funct = distance_funct
        self.th = th

        self.current_tracks = []
        self.apparences = None

    def associate(self, new_track):

        if new_track.shape[0] == 0 or new_track.shape[1] <= 10:
            return self.current_tracks
        
        apparence = self.aggregation(new_track[:, 10:])

        if len(self.current_tracks) == 0:
            self.current_tracks.append(new_track[:, :10])
            self.apparences = apparence.reshape(1, *apparence.shape)
        
        else:
            distances = self.distance_funct(self.apparences, apparence)
            idx = np.argmin(distances)

            if distances[idx] < self.th:
                new_track[:, 1] = self.current_tracks[idx][0, 1]
                self.current_tracks[idx] = np.vstack((self.current_tracks[idx], new_track[:, :10]))
                new_apparence = self.aggregation(self.current_tracks[idx][:, 10:])
                self.apparences[idx, :] = new_apparence
            else:
                self.current_tracks.append(new_track[:, :10])
                self.apparences = np.vstack([self.apparences, apparence])
        
        return self.current_tracks

    __call__ = associate
