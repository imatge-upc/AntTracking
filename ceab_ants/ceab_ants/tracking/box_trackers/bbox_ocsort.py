
# NOTE: build_bbox_ocsort and build_rbbox_ocsort

import numpy as np

from ceab_ants.bbox_metrics.bbox_metrics import iou_bbox_batch, angle_score_bbox_batch, bbox_speed_direction
from ceab_ants.tracking.box_trackers.estimators.bbox_estimator import BBoxEstimator
from ceab_ants.tracking.box_trackers.estimators.rbbox_estimator import RBBoxEstimator
from ceab_ants.tracking.box_trackers.utils.bbox_utils import features_to_ccwh, bbox_to_observation, features_to_bbox, ccwh_to_features
from ceab_ants.tracking.trackers.sort.sort import SORT #, SORTAssociator, SORTEstimator, SORTTrackManager
from ceab_ants.tracking.trackers.sort.ocsort_utils.associator import OCSORTAssociator
from ceab_ants.tracking.trackers.sort.ocsort_utils.estimator import OCSORTEstimator
from ceab_ants.tracking.trackers.sort.ocsort_utils.track_manager import OCSORTTrackManager


##### BBOX UTILS #####

def ocsort_bbox_detections_decoder(bbox):
    # This prepare the input of the kalman filter, if observation includes additional data.
    return bbox_to_observation(bbox)

def ocsort_bbox_estimation_encoder(kalman_pred, momentum, last_obs, kth_obs):
    # This prepare the data that the associator as well as the track manager will recive.
    momentum = (np.asarray(momentum or np.array((0, 0)))).reshape((1, 2))
    last_obs = last_obs.reshape(1, 5) if last_obs is not None else np.full((1, 5), -1)
    kth_obs = kth_obs.reshape(1, 5) if kth_obs is not None else np.full((1, 5), -1)
    return np.concatenate([features_to_bbox(kalman_pred)[:, :4], momentum, last_obs, kth_obs], axis=1).reshape(1, 16)

def ocsort_bbox_estimation_decoder(estimations):
    # This answer how is the predicted data at the input of the associator
    estimations = np.vstack(estimations) 
    trackers = estimations[:, :4]
    velocities = estimations[:, 4 : 6]
    last_boxes = estimations[:, 6 : 11] # Bbox with confidence so 5 elements
    previous_obs = estimations[:, 11 : 16] # Bbox with confidence so 5 elements

    return trackers, velocities, last_boxes, previous_obs

def ocsort_process_bbox_output(out, trk, return_pred=False):
    # Prepare a row of the current frame output (tracks need a score, by instance 0)
    if return_pred:
        out = np.concatenate([out[:4], [0]]) # OCSORT Kalman output has a lot of extra data and no confidence score

    return np.concatenate((out[:5].reshape(-1), [trk.id])).reshape(1, -1)

def ocsort_rbbox_unfreeze_update_generator(initial_obs, final_obs, time_gap):
    
    bbox1 = features_to_ccwh(initial_obs).reshape(-1)
    bbox2 = features_to_ccwh(final_obs).reshape(-1)
    
    increment = (bbox2 - bbox1) / time_gap
    for i in range(1, time_gap + 1):
        bbox_i = bbox1 + i * increment

        kalman = yield ccwh_to_features(bbox_i).reshape((4, 1))

        if kalman.state[6] + kalman.state[2] <= 0 : kalman.state[6] = 0
        yield None

def ocsort_bbox_unfreeze_update_generator(initial_obs, final_obs, time_gap):
    
    bbox1 = features_to_ccwh(initial_obs).reshape(-1)
    bbox2 = features_to_ccwh(final_obs).reshape(-1)
    
    increment = (bbox2 - bbox1) / time_gap
    for i in range(1, time_gap + 1):
        bbox_i = bbox1 + i * increment

        kalman = yield ccwh_to_features(bbox_i).reshape((4, 1))

        if kalman.state[6] + kalman.state[2] <= 0 : kalman.state[6] = 0
        if kalman.state[7] + kalman.state[3] <= 0 : kalman.state[7] = 0
        
        yield None

##### BBOX MODELS #####

def build_bbox_ocsort(
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
        iou_bbox_batch, 
        angle_score_bbox_batch, 
        ocsort_bbox_estimation_decoder, 
        second_score_function, 
        None, 
        th_conf, 
        th_first_score, 
        th_second_score, 
        inertia_weight, 
        use_byte
    )
    
    unfreeze_update_generator = ocsort_rbbox_unfreeze_update_generator if reduced else ocsort_bbox_unfreeze_update_generator
    ocsort_estimator_builder = lambda nf, tm, sc, su, fu, process_obs, process_pred : OCSORTEstimator(
        nf, tm, sc, su, fu, 
        unfreeze_update_generator=unfreeze_update_generator, 
        momentum_function=bbox_speed_direction, 
        delta_t=delta_t,
        process_obs=process_obs, 
        process_pred=process_pred
    )

    estimator_cls = RBBoxEstimator if reduced else BBoxEstimator
    rbbox_estimator_builder = lambda : estimator_cls(
        estimator_cls=ocsort_estimator_builder, 
        postprocess_funct=ocsort_bbox_estimation_encoder, 
        preprocess_funct=ocsort_bbox_detections_decoder
    )

    track_manager = OCSORTTrackManager(
        rbbox_estimator_builder, 
        max_age=max_age, 
        min_hits=min_hits, 
        max_last_update=max_last_update, 
        return_pred=False, 
        process_output=ocsort_process_bbox_output,
        confidence_funct=lambda det : det[4],
        th_conf=th_conf
    )

    ocsort = SORT(detector, associator, track_manager)
    return ocsort

def build_rbbox_ocsort(
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
    
    ocsort =  build_bbox_ocsort(
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
