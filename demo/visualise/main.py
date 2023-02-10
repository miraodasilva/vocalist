import os
import cv2
import glob
import ffmpeg
import librosa
import numpy as np

from .utils import *


class LandmarkVisualiser(object):
    def __init__(self, connection_colour=(135, 27, 132)):
        """
        :type Tuple, connection_colour: the BGR color for landmark edges.
        """
        self._connection_colour = connection_colour

    def visualise(
        self,
        video_pathname,
        landmarks,
        dst_v_pathname,
        landmarks_interpolate=False,
        visualise_landmarks=False,
        sync_av=False,
    ):
        """video visualisation.
        :type str video_pathname: the filename for the raw video.
        :type str landmarks_pathname: the filename for the landmarks
        :type str dst_v_pathname: the filename for the saved video.
        :type bool landmarks_interpolate: the flag to use landmark interpolation.
        :type bool visualise_landmarks: the flag to visualise landmarks.
        :type bool sync_av: the flag to merge audio and visual stream.
        """

        # -- Step 1, get properties of videos
        properties = get_video_properties(video_pathname)
        # -- Step 2, extract landmarks from pkl files, exclude landmarks with low confidence.
        """
        landmarks, multi_sub_landmarks = pick_landmarks_fromRetinaFaceTracker(landmarks_pathname)
        if landmarks_interpolate:
            landmarks = landmarks_interpolate(landmarks)
        if not landmarks:
            return
        """
        # -- step 3, visualise landmarks
        writer = cv2.VideoWriter(
            dst_v_pathname[:-4] + ".mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            properties["fps"],
            (int(properties["width"]), int(properties["height"])),
            isColor=True,
        )
        vid_gen = read_video(video_pathname)
        idx = 0
        while True:
            try:
                frame = vid_gen.__next__()
            except StopIteration:
                break
            if visualise_landmarks and landmarks[idx] is not None:
                plot_landmarks(
                    frame, landmarks[idx], connection_colour=self._connection_colour, landmark_colour=(0, 0, 255)
                )
            writer.write(frame)
            idx += 1
        writer.release()

        # sync audio and visual stream.
        if sync_av:
            self.sync_av(
                src_a_pathname=video_pathname,
                dst_a_pathname=dst_v_pathname[:-4] + ".wav",
                dst_v_pathname=dst_v_pathname,
            )

    def sync_av(self, src_a_pathname, dst_a_pathname, dst_v_pathname):
        """
        :type str src_a_pathname: origianl audio filename
        :type str dst_a_pathname: saved audio filename
        :type str dst_v_pathname: saved video filename
        """

        y, sr = librosa.load(src_a_pathname, sr=None)
        dst_a_pathname = dst_v_pathname[:-4] + ".wav"
        librosa.output.write_wav(dst_a_pathname, y, sr, norm=False)
        video = ffmpeg.input(dst_v_pathname)
        audio = ffmpeg.input(dst_a_pathname)
        out = ffmpeg.output(
            video, audio, dst_v_pathname[:-4] + ".m.mp4", vcodec="copy", acodec="aac", strict="experimental"
        )
        ffmpeg.run(out)
        os.system("rm -rf {}".format(dst_a_pathname))
        os.system("rm -rf {}".format(dst_v_pathname))
        os.system("mv {} {}".format(dst_v_pathname[:-4] + ".m.mp4", dst_v_pathname))


landmark_visualiser = LandmarkVisualiser(connection_colour=(0, 255, 0))
src_dir = "./examples/videos"
video_filenames = glob.glob(os.path.join(src_dir, "**/*.mp4"), recursive=True)

for video_filename in video_filenames:
    landmark_filename = video_filename.replace("/videos/", "/landmarks/")[:-4] + ".pkl"
    dst_v_pathname = video_filename.replace("/videos/", "/outputs/")
    assert os.path.exists(landmark_filename), "{} does not exist."
    if not os.path.exists(os.path.dirname(dst_v_pathname)):
        os.makedirs(os.path.dirname(dst_v_pathname))
    landmark_visualiser.visualise(
        video_filename, landmark_filename, dst_v_pathname, visualise_landmarks=True, sync_av=False
    )
