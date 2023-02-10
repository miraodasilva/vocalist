#coding=utf-8
import cv2
import pickle
import numpy as np


def read_video(filename):
    cap = cv2.VideoCapture(filename)                                             
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR                                            
        if ret:                                                                  
            yield frame                                                          
        else:                                                                    
            break                                                                
    cap.release()


def pick_landmarks_fromRetinaFaceTracker(filename):
    """                                                                          
    Design for the pingchuan's landmark pkl files.                               
    """                                                                          
                                                                                 
    # -- for RetineFaceTracker                                                   
    with open(filename, "rb") as pkl_file:                                       
        multi_sub_landmarks = pickle.load(pkl_file)                              
    landmarks = [None] * len( multi_sub_landmarks["landmarks"])                  
    for frame_idx in range(len(landmarks)):                                      
        if len(multi_sub_landmarks["landmarks"][frame_idx]) == 0:
            continue                                                             
        else:                                                                    
            # -- decide person id using maximal bounding box  0: Left, 1: top, 2: right, 3: bottom, probability
            max_bbox_person_id = 0                                               
            max_bbox_len = multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][2] + \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][3] - \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][0] - \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][1]
            landmark_scores = multi_sub_landmarks["landmarks_scores"][frame_idx][max_bbox_person_id]
            for temp_person_id in range(1, len(multi_sub_landmarks["bbox"][frame_idx])):
                temp_bbox_len = multi_sub_landmarks["bbox"][frame_idx][temp_person_id][2] + \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][3] - \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][0] - \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][1]
                if temp_bbox_len > max_bbox_len:                                 
                    max_bbox_person_id = temp_person_id                          
                    max_bbox_len = temp_bbox_len                                 
                    landmark_scores = multi_sub_landmarks['landmarks_scores'][frame_idx][temp_person_id]
            landmarks[frame_idx] = multi_sub_landmarks["landmarks"][frame_idx][max_bbox_person_id] if landmark_scores[17:].min() >= 0.2 else None
    return landmarks, multi_sub_landmarks


def plot_landmarks(frame, landmarks, connection_colour, landmark_colour):
    for idx in range(len(landmarks) - 1):
        if idx < 48:
            continue
        if (idx != 16 and idx != 21 and idx != 26 and idx != 30 and
                idx != 35 and idx != 41 and idx != 47 and idx != 59):
            cv2.line(frame, tuple(landmarks[idx].astype(int).tolist()),
                     tuple(landmarks[idx + 1].astype(int).tolist()),
                     color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
        if idx == 30:
            cv2.line(frame, tuple(landmarks[30].astype(int).tolist()),
                     tuple(landmarks[33].astype(int).tolist()),
                     color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
        elif idx == 36:
            cv2.line(frame, tuple(landmarks[36].astype(int).tolist()),
                     tuple(landmarks[41].astype(int).tolist()),
                     color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
        elif idx == 42:
            cv2.line(frame, tuple(landmarks[42].astype(int).tolist()),
                     tuple(landmarks[47].astype(int).tolist()),
                     color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
        elif idx == 48:
            cv2.line(frame, tuple(landmarks[48].astype(int).tolist()),
                     tuple(landmarks[59].astype(int).tolist()),
                     color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
        elif idx == 60:
            cv2.line(frame, tuple(landmarks[60].astype(int).tolist()),
                     tuple(landmarks[67].astype(int).tolist()),
                     color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
    
    # border top
    xx1, xx2 = min(landmarks[:, 0]), max(landmarks[:, 0])
    yy1, yy2 = max(landmarks[:, 1]), min(landmarks[:, 1])
    cv2.rectangle(frame, (int(xx1), int(yy1)), (int(xx2), int(yy2)), color=connection_colour, thickness=1)

    for idx, landmark in enumerate(landmarks):
        if idx in [48, 51, 57, 54]:
            cv2.circle(frame, tuple(landmark.astype(int).tolist()), 2, landmark_colour, thickness=2)


def get_video_properties(filename):
    properties = {}
    vid =  cv2.VideoCapture(filename)
    properties['width'] = vid.get(cv2.CAP_PROP_FRAME_WIDTH )
    properties['height'] = vid.get(cv2.CAP_PROP_FRAME_HEIGHT )
    properties['fps'] =  vid.get(cv2.CAP_PROP_FPS)
    properties['frames'] = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.release()
    return properties


# -- Landmark interpolation.
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):

    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks
