from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from PIL import Image
import cv2
import dlib
from IPython.display import display

from models.psp import pSp
from models.e4e import e4e
from utils.common import tensor2im
from utils.inference_utils import run_on_batch
from scripts.align_faces_parallel import align_face

from src.model import Net


def inference_model_on_generated(net: Net,
                           samples: int = 9,
                           truncation: float = 0.7,
                           device: str = "cuda:0"
                           ) -> None:
    with torch.no_grad():
      net.eval()
      sample_z = torch.randn(samples, 512, device=device)

      sampled_src, sampled_dst, cliploss = net([sample_z], truncation=truncation)

      fig, axs = plt.subplots(samples, 2, figsize=(16, samples*8))  # Создание подграфиков для визуализации

      for j in range(samples):
            sampled_src_img = (sampled_src[j].cpu().detach().permute(1, 2, 0).clip(-1,1) + 1) / 2
            sampled_dst_img = (sampled_dst[j].cpu().detach().permute(1, 2, 0).clip(-1,1) + 1) / 2

            # Визуализация исходного изображения
            axs[j, 0].imshow(sampled_src_img)
            axs[j, 0].set_title("Сгенерированное изображение")
            axs[j, 0].axis('off')

            # Визуализация сгенерированного изображения
            axs[j, 1].imshow(sampled_dst_img)
            axs[j, 1].set_title("Сгенерированное изображение + joker")
            axs[j, 1].axis('off')

    plt.show()


def load_encoder(encoder_type: str = 'e4e'):
    restyle_experiment_args = {
      "model_path": f"restyle_{encoder_type}_ffhq_encode.pt",
      "transform": transforms.Compose([
          transforms.Resize((256, 256)),
          transforms.ToTensor(),
          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
      }

    model_path = restyle_experiment_args['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)

    restyle_net = (pSp if encoder_type == 'psp' else e4e)(opts)
    restyle_net.eval()
    restyle_net.cuda()
    print('Model successfully loaded!')
    return restyle_net, opts, restyle_experiment_args


def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def prepare_image(image_path: str, restyle_net, restyle_experiment_args, opts):
  original_image = Image.open(image_path).convert("RGB")
  input_image = run_alignment(image_path)

  img_transforms = restyle_experiment_args['transform']
  transformed_image = img_transforms(input_image)

  opts.n_iters_per_batch = 5
  opts.resize_outputs = False  # generate outputs at full resolution

  with torch.no_grad():
      avg_image = get_avg_image(restyle_net)
      result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), restyle_net, opts, avg_image)
  return input_image, result_batch, result_latents


def edit_image(result_latents, net):
    inverted_latent = torch.Tensor(result_latents[0][4]).cuda().unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        net.eval()

        sampled_src, sampled_dst, loss = net(inverted_latent, input_is_latent=True)
    return sampled_src, sampled_dst


# Определяем функцию для отображения результата рядом с исходным
def display_alongside_source_image(result_image, source_image, resize_dims=(512, 512)):
    """Отображает результат рядом с исходным изображением.

    Args:
        result_image: Результирующее изображение (PIL.Image).
        source_image: Исходное изображение (PIL.Image).

    Returns:
        Объединенное изображение (PIL.Image).
    """
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)


def inference_model_on_real(model, restyle_net, restyle_experiment_args, opts, image_path):

  input_image, result_batch, result_latents = prepare_image(image_path, restyle_net, restyle_experiment_args, opts)
  sampled_src, sampled_dst = edit_image(result_latents, model)

  print('Orig aligned VS edited')
  display(display_alongside_source_image(tensor2im(sampled_dst.squeeze(0)), input_image))
  return sampled_src, sampled_dst, input_image
