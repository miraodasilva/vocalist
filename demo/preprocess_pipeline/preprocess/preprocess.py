#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import os
import time
from ..dataloader.dataloader import AVSRDataLoader


class MouthPreprocessor(object):
    """MouthPreprocessor."""

    def __init__(
        self,
        modality,
        face_track=False,
        device="cuda:0",
    ):
        """__init__.
        :param modality: str, the modality for loading. choices chosen from ["video", "raw_audio"]
        :param face_track: str, face tracker will be used if set it as True.
        :param device: str, contain the device on which a torch.Tensor is or will be allocated.
        """

        self._dataloader = AVSRDataLoader(
            convert_gray=False, start_idx=0
        )  # we set start_idx to 0 so that we crop the whole face
        self._device = device
        self._modality = modality
        if face_track:
            from ..tracker.face_tracker import FaceTracker

            self.face_tracker = FaceTracker(device=device)
        else:
            self.face_tracker = None

    def __call__(self, data_filename, landmarks_filename):
        """__call__.

        :param data_filename: str, the filename of the input sequence.
        :param landmarks_filename: str, the filename of the corresponding landmarks.
        """
        # Step 1, track face in the input video or read landmarks from the file.
        assert os.path.isfile(data_filename), "{} does not exist.".format(data_filename)
        if os.path.isfile(landmarks_filename):
            landmarks = landmarks_filename
        else:
            assert self.face_tracker is not None, "Face tracker is not enabled."
            end = time.time()
            landmarks = self.face_tracker.tracker(data_filename)
            print("Detection speed: {:.2f} fps.".format(len(landmarks) / (time.time() - end)))

        # Step 2, extract mouth patches from segments.
        sequence = self._dataloader.load_data(
            self._modality,
            data_filename,
            landmarks,
        )

        return landmarks, sequence
