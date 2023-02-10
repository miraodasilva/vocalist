from models.model import SyncTransformer
import torchaudio
from torchaudio.transforms import MelScale
from torchvision.transforms import Compose
from hparams import hparams
import torch
import torch.nn as nn
import numpy as np
import cv2
from demo.preprocess_pipeline.preprocess.preprocess import MouthPreprocessor


def write_lossless_video(video, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25, (96, 96), 3)
    for frame in list(video):
        writer.write(frame)
    writer.release()  # close the writer


class AudioLoader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, audio_path):
        melscale = MelScale(
            n_mels=hparams.num_mels,
            sample_rate=hparams.sample_rate,
            f_min=hparams.fmin,
            f_max=hparams.fmax,
            n_stft=hparams.n_stft,
            norm="slaney",
            # mel_scale="slaney",
        )

        aud_tensor = torchaudio.load(audio_path)
        spec = torch.stft(
            aud_tensor,
            n_fft=hparams.n_fft,
            hop_length=hparams.hop_size,
            win_length=hparams.win_size,
            window=torch.hann_window(hparams.win_size),
            return_complex=True,
        )
        melspec = melscale(torch.abs(spec.detach().clone()).float())
        melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
        # NORMALIZED MEL
        normalized_mel = torch.clip(
            (2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
            -hparams.max_abs_value,
            hparams.max_abs_value,
        )
        mels = normalized_mel.unsqueeze(0)


class VideoLoader(nn.Module):
    def __init__(self):
        super().__init__()
        # video
        self.mouthprocessor = MouthPreprocessor(modality="video", face_track=True)

    def forward(self, path):
        _, cropped_mouth_video = self.mouthprocessor(path, "")
        write_lossless_video(cropped_mouth_video,"./cropped.mp4")
        video = torch.from_numpy(cropped_mouth_video)
        if video.dim() == 3:
            video = video.unsqueeze(-1)
        


video_path = "/vol/paramonos2/projects/antoni/datasets/laugh/av-laughcycle/CroppedVideos/6_sf_10171_ef_10320.avi"
audio_path = "/vol/paramonos2/projects/antoni/datasets/laugh/av-laughcycle/Audio/6_sf_10171_ef_10320.wav"
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))

audio_loader = AudioLoader()
video_loader = VideoLoader()

# audio = audio_loader(audio_path)
# print(audio.size())
video = video_loader(video_path)
print(video.size())
