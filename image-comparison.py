import torch
from PIL import Image
import PIL
import torchvision.transforms as transforms
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

def remove_alpha(image):
    if image.shape[0] == 4:
        return image[:3, :, :]
    return image

def calculate_mse(image1, image2):
    return ((image1 - image2) ** 2).mean().item()

def resize_image(image):
    return torch.nn.functional.avg_pool2d(image, 2)

def similarity(image1_path, image2_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.ToTensor()

    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    assert image1.size == image2.size, "Images must have the same size"

    image1 = remove_alpha(transform(image1)).to(device)
    image2 = remove_alpha(transform(image2)).to(device)

    errors = []
    while True:
        mse = calculate_mse(image1, image2)

        errors.append(mse)

        _, h, w = image1.size()
        if h == 1 or w == 1:
            break

        image1 = resize_image(image1)
        image2 = resize_image(image2)

    average_mse = np.mean(errors)
    sim = 1 - average_mse

    return sim


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python image-comparison.py <image1_path> <image2_path>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    sim = similarity(image1_path, image2_path)
    print(sim)