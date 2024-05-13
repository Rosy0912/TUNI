import numpy as np
import pandas as pd
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from PIL import Image
import open_clip
import csv
import random
from torchvision.utils import save_image

def get_random_image_path(image_dir):
    images = os.listdir(image_dir)
    if not images:
        raise ValueError(f"No images found in directory {image_dir}")
    return os.path.join(image_dir, random.choice(images))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

model_paths = [
    '/home/rosych0912/.conda/mia/clip_model/rn50x4_top01_epoch_50.pt',
    '/home/rosych0912/.conda/mia/clip_model/rn50x4_top75_epoch_50.pt',
    '/home/rosych0912/.conda/mia/clip_model/rn50_top01_epoch_50.pt',
    '/home/rosych0912/.conda/mia/clip_model/rn50_top75_epoch_50.pt',
    '/home/rosych0912/.conda/mia/clip_model/vitb32_top01_epoch_50.pt',
    '/home/rosych0912/.conda/mia/clip_model/vitb32_top75_epoch_50.pt'
]

def get_model_name_from_path(model_path):
    if "rn50_top01" in model_path or "rn50_top75" in model_path:
        return "RN50"
    elif "rn50x4_top01" in model_path or "rn50x4_top75" in model_path:
        return "RN50x4"
    elif "vitb32_top01" in model_path or "vitb32_top75" in model_path:
        return "ViT-B/32"
    else:
        return "Unknown"

def preprocess_image(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    processed_image = preprocess(image)
    return processed_image.unsqueeze(0)


optimized_images_dir = '/home/rosych0912/.conda/mia/optimized_images'


def main(rank, world_size, model_path, text_description):
    setup(rank, world_size)

    model_name = get_model_name_from_path(model_path)
    if model_name == "Unknown":
        print(f"Skipping unknown model for path: {model_path}")
        cleanup()
        return

    outputs = open_clip.create_model_and_transforms(model_name=model_name, pretrained=model_path)
    clip_model, preprocess = outputs[0].to(rank), outputs[1]
    clip_model.eval()

    img_path = get_random_image_path("/home/rosych0912/.conda/mia/UTKFace_In_the_wild Faces/")
    feature_vectors = []

    for iteration in range(100):
        img = preprocess_image(img_path, preprocess).to(rank)
        img.requires_grad = True
        text_inputs = open_clip.tokenize([text_description]).to(rank)
        optimizer = optim.SGD([img], lr=0.02)

        for step in range(2000):
            optimizer.zero_grad()
            image_features = clip_model.encode_image(img)
            text_features = clip_model.encode_text(text_inputs)
            similarity = torch.cosine_similarity(image_features, text_features)
            loss = -similarity.mean()
            loss.backward()
            optimizer.step()

            if step == 1999:
                feature_vector = image_features[0].detach().cpu().numpy()
                feature_vectors.append(feature_vector)

                if rank == 0:
                    print(f"Model {model_name}, Text: {text_description} - Iteration {iteration + 1}, Step {step + 1}, Loss {loss.item()}")

                    img_to_save = img[0].detach().cpu().clone()
                    save_path_img = f'/home/rosych0912/.conda/mia/optimized_images/{model_name}_{text_description.replace(" ", "_")}.png'
                    save_image(img_to_save, save_path_img)
                    print(f"Optimized image {iteration} saved to {save_path_img}")

    if rank == 0:
        save_path = f'/home/rosych0912/.conda/mia/MIA-CLIP/features/{model_name}_{text_description.replace(" ", "_")}.csv'
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for vector in feature_vectors:
                writer.writerow(vector)
        print(f"Feature vectors for {model_name} with {text_description} saved to {save_path}")

        vectors = np.array(feature_vectors)
        mean_vector = np.mean(vectors, axis=0)
        distances = np.sqrt(np.sum((vectors - mean_vector) ** 2, axis=1))
        total_distance = np.sum(distances)
        distances_path = f'/home/rosych0912/.conda/mia/MIA-CLIP/distances/{model_name}{text_description.replace(" ", "_")}_distances.csv'
        with open(distances_path, 'w') as f:
            f.write(f"{total_distance}\n")
        print(f"Total distance for {model_name} with {text_description}: {total_distance}")
    cleanup()


if __name__ == "__main__":
    descriptions_df = pd.read_csv('/home/rosych0912/.conda/mia/name_test.csv', header=None)
    text_descriptions = descriptions_df[0].tolist()
    world_size = torch.cuda.device_count()
    for model_path in model_paths:
        for text_description in text_descriptions:
            mp.spawn(main,
                     args=(world_size, model_path, text_description),
                     nprocs=world_size,
                     join=True)



