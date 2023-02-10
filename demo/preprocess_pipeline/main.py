from preprocess.preprocess import MouthPreprocessor

face_track = True
modality = "video"
pre = MouthPreprocessor(
    modality=modality,
    face_track=face_track,
)

data_filename = "/vol/paramonos/projects/Stavros/lipread_mp4/ABOUT/test/ABOUT_00001.mp4"
landmarks_filename = ""

assert pre(data_filename, landmarks_filename)[1].shape == (29, 96, 96)
