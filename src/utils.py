import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import random
import torch

def load_weights():

  # authenticate and create the PyDrive client
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  # downloads StyleGAN's weights and e4e encoder's weights
  ids = ['1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT', '1e2oXVeBPXMQoUoC_4TNwAWpOPpSEhE_e']
  for file_id in ids:
    downloaded = drive.CreateFile({'id':file_id})
    downloaded.FetchMetadata(fetch_all=True)
    downloaded.GetContentFile(downloaded.metadata['title'])

  # face alignment
  os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
  os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]
  