
# NOTE: build_obbox_ocsort and build_robbox_ocsort

import numpy as np

from ceab_ants.bbox_metrics.obbox_metrics import iou_obbox_batch, angle_score_obbox_batch, obbox_speed_direction
from ceab_ants.tracking.box_trackers.estimators.obbox_estimator import OBBoxEstimator
from ceab_ants.tracking.box_trackers.estimators.robbox_estimator import ROBBoxEstimator
from ceab_ants.tracking.box_trackers.utils.obbox_utils import obbox_to_observation, features_to_obbox
from ceab_ants.tracking.trackers.sort.sort import SORT #, SORTAssociator, SORTEstimator, SORTTrackManager
from ceab_ants.tracking.trackers.sort.ocsort_utils.associator import OCSORTAssociator
from ceab_ants.tracking.trackers.sort.ocsort_utils.estimator import OCSORTEstimator
from ceab_ants.tracking.trackers.sort.ocsort_utils.track_manager import OCSORTTrackManager


##### OBBOX UTILS #####

def ocsort_obbox_detections_decoder(bbox):
    # This prepare the input of the kalman filter, if observation includes additional data.
    return obbox_to_observation(bbox)

def ocsort_obbox_estimation_encoder(kalman_pred, momentum, last_obs, kth_obs):
    # This prepare the data that the associator as well as the track manager will recive.
    momentum = (np.asarray(momentum or np.array((0, 0)))).reshape((1, 2))
    last_obs = last_obs.reshape(1, 6) if last_obs is not None else np.full((1, 6), -1)
    kth_obs = kth_obs.reshape(1, 6) if kth_obs is not None else np.full((1, 6), -1)
    return np.concatenate([features_to_obbox(kalman_pred)[:, :5], momentum, last_obs, kth_obs], axis=1).reshape(1, 19)

def ocsort_obbox_estimation_decoder(estimations):
    # This answer how is the predicted data at the input of the associator
    estimations = np.vstack(estimations) 
    trackers = estimations[:, :5]
    velocities = estimations[:, 5 : 7]
    last_boxes = estimations[:, 7 : 13] # Obbox with confidence so 6 elements
    previous_obs = estimations[:, 13 : 19] # Obbox with confidence so 6 elements

    return trackers, velocities, last_boxes, previous_obs

def ocsort_process_obbox_output(out, trk, return_pred=False):
    # Prepare a row of the current frame output (tracks need a score, by instance 0)
    if return_pred:
        out = np.concatenate([out[:5], [0]]) # OCSORT Kalman output has a lot of extra data and no confidence score

    return np.concatenate((out[:6].reshape(-1), [trk.id])).reshape(1, -1)

def ocsort_robbox_unfreeze_update_generator(initial_obs, final_obs, time_gap):
    
    bbox1 = features_to_obbox(initial_obs).reshape(-1)
    bbox2 = features_to_obbox(final_obs).reshape(-1)
    
    increment = (bbox2 - bbox1) / time_gap
    for i in range(1, time_gap + 1):
        bbox_i = bbox1 + i * increment

        kalman = yield obbox_to_observation(bbox_i).reshape((5, 1))

        yield None

def ocsort_obbox_unfreeze_update_generator(initial_obs, final_obs, time_gap):
    
    bbox1 = features_to_obbox(initial_obs).reshape(-1)
    bbox2 = features_to_obbox(final_obs).reshape(-1)
    
    increment = (bbox2 - bbox1) / time_gap
    for i in range(1, time_gap + 1):
        bbox_i = bbox1 + i * increment

        kalman = yield obbox_to_observation(bbox_i).reshape((5, 1))

        if kalman.state[7] + kalman.state[2] <= 0 : kalman.state[7] = 0
        if kalman.state[8] + kalman.state[3] <= 0 : kalman.state[8] = 0
        
        yield None

##### OBBOX MODELS #####

def build_obbox_ocsort(
        detector, 
        second_score_function, 
        th_conf=0.5, 
        th_first_score=0.3, 
        th_second_score=0.3, 
        inertia_weight=0.2, 
        use_byte=False, 
        delta_t=3, 
        max_age=1, 
        min_hits=3, 
        max_last_update=1,
        reduced=False
):

    associator = OCSORTAssociator(
        iou_obbox_batch, 
        angle_score_obbox_batch, 
        ocsort_obbox_estimation_decoder, 
        second_score_function, 
        None, 
        th_conf, 
        th_first_score, 
        th_second_score, 
        inertia_weight, 
        use_byte
    )
    
    unfreeze_update_generator = ocsort_robbox_unfreeze_update_generator if reduced else ocsort_obbox_unfreeze_update_generator
    ocsort_estimator_builder = lambda nf, tm, sc, su, fu, process_obs, process_pred : OCSORTEstimator(
        nf, tm, sc, su, fu, 
        unfreeze_update_generator=unfreeze_update_generator, 
        momentum_function=obbox_speed_direction, 
        delta_t=delta_t,
        process_obs=process_obs, 
        process_pred=process_pred
    )

    estimator_cls = ROBBoxEstimator if reduced else OBBoxEstimator
    rbbox_estimator_builder = lambda : estimator_cls(
        estimator_cls=ocsort_estimator_builder, 
        postprocess_funct=ocsort_obbox_estimation_encoder, 
        preprocess_funct=ocsort_obbox_detections_decoder
    )

    track_manager = OCSORTTrackManager(
        rbbox_estimator_builder, 
        max_age=max_age, 
        min_hits=min_hits, 
        max_last_update=max_last_update, 
        return_pred=False, 
        process_output=ocsort_process_obbox_output,
        confidence_funct=lambda det : det[4],
        th_conf=th_conf
    )

    ocsort = SORT(detector, associator, track_manager)
    return ocsort

def build_robbox_ocsort(
        detector, 
        second_score_function, 
        th_conf=0.5, 
        th_first_score=0.3, 
        th_second_score=0.3, 
        inertia_weight=0.2, 
        use_byte=False, 
        delta_t=3, 
        max_age=1, 
        min_hits=3, 
        max_last_update=1):
    
    ocsort =  build_obbox_ocsort(
        detector, 
        second_score_function, 
        th_conf=th_conf, 
        th_first_score=th_first_score, 
        th_second_score=th_second_score, 
        inertia_weight=inertia_weight, 
        use_byte=use_byte, 
        delta_t=delta_t, 
        max_age=max_age, 
        min_hits=min_hits, 
        max_last_update=max_last_update,
        reduced=True)
    
    return ocsort
