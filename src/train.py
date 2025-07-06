from src.train_config import TrainParams
from src.model import Net
from src.utils import mixing_noise

import os
from tqdm import notebook
import matplotlib.pyplot as plt
import numpy as np
import torch


def train_model(training_params: TrainParams) -> Net:

    if training_params.seed:
      torch.manual_seed(training_params.seed)
      np.random.seed(training_params.seed)

    # create chckpoints dir
    os.makedirs(training_params.checkpoint_path, exist_ok=True)

    # set model
    net = Net(training_params)
    optimizer = torch.optim.Adam(
        net.generator_trainable.parameters(), lr=training_params.lr
        )

    # remove warnings
    import warnings
    warnings.filterwarnings('ignore')

    # generate latents to train
    fixed_z = torch.randn(training_params.n_samples, 512, device=training_params.device)

    # loop
    print("Start training...")
    for i in range(training_params.training_iterations):
    # for i in notebook.tqdm(range(training_params.training_iterations)):

      net.train()
      sample_z = mixing_noise(training_params.batch_size, 512, training_params.mixing, training_params.device)
      sampled_src, sampled_dst, cliploss = net(sample_z)

      net.zero_grad()
      cliploss.backward()
      optimizer.step()

      if i == 0 or (i + 1) % training_params.output_interval == 0:

          net.eval()
          with torch.no_grad():
              sampled_src, sampled_dst, cliploss = net([fixed_z], truncation=training_params.sample_truncation)

          print(f"Result | Step [{i}], Loss: {cliploss.item():.4f}")  # Вывод текущего лосса
          fig, axs = plt.subplots(training_params.n_samples, 2, figsize=(16, training_params.n_samples*8))  # Создание подграфиков для визуализации

          for j in range(training_params.n_samples):
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

      if (training_params.save_interval > 0) and (i > 0) and ((i + 1) % training_params.save_interval == 0):
          torch.save(
              {
                  "g_ema": net.generator_trainable.generator.state_dict(),
                  "g_optim": optimizer.state_dict(),
              },
              f"{training_params.checkpoint_path}/{str(i).zfill(6)}.pt",
          )
          print(f"checkpoint saved to '{training_params.checkpoint_path}/{str(i).zfill(6)}.pt'")
    return net

