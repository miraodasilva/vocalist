import os
import cv2
import glob
import random
import numpy as np
from utils import pick_landmarks_fromRetinaFaceTracker
src_dir = "/vol/paramonos2/projects/pm4115/IC_LRS3_Lang/IC_LRS3_Lang_RetinaFaceLandmark/es/"
res = []
filenames = glob.glob(os.path.join(src_dir, "**/*.pkl"), recursive=True)
random.shuffle(filenames)
for landmarks_pathname in filenames:
    # -- Step 2, extract landmarks from pkl files, exclude landmarks with low confidence.
    landmarks, multi_sub_landmarks = pick_landmarks_fromRetinaFaceTracker(landmarks_pathname)
    if not landmarks:
        continue
    for landmark in landmarks:
        if landmark is not None:
            res.append(landmark)
    if len(res) > 1000000:
        break
res = np.array(res)
res = np.stack(res, axis=0)
landmarks = np.mean(res, axis=0)
print(landmarks.shape)
height, width = 1080, 1920
frame = np.zeros((height, width, 3), np.uint8)
xx1, xx2 = min(landmarks[:, 0]), max(landmarks[:, 0])
yy1, yy2 = max(landmarks[:, 1]), min(landmarks[:, 1])
#cv2.rectangle(frame, (int(xx1), int(yy1)), (int(xx2), int(yy2)), color=(0, 0, 255), thickness=1)
print( (int(xx1), int(yy1)), (int(xx2), int(yy2)))
#for idx, landmark in enumerate(landmarks):
#    cv2.circle(frame, tuple(landmark.astype(int).tolist()), 2, (0, 255, 0), thickness=2)
#cv2.imwrite("/homes/pm4115/public_html/landmarks.jpg", frame)

'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#landmarks = np.load("/vol/paramonos2/projects/pm4115/for_Marija/step2_affine_transformation/20words_mean_face.npy")
xx1, xx2 = min(landmarks[:, 0]), max(landmarks[:, 0])
yy1, yy2 = max(landmarks[:, 1]), min(landmarks[:, 1])
print( (int(xx1), int(yy1)), (int(xx2), int(yy2)))
plt.figure(1)
plt.scatter(x=landmarks[:,0], y=1920-landmarks[:,1], c='r', s=40)
plt.savefig("/homes/pm4115/public_html/landmarks_lrw.jpg", dpi=640)
'''
