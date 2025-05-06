import torch
from PIL import Image
import PIL
import torchvision.transforms as transforms

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

def remove_alpha(image):
    if image.shape[0] == 4:
        return image[:3, :, :]
    return image

def is_blank(image_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    transform = transforms.ToTensor()

    image1 = Image.open(image_path)
    image1 = remove_alpha(transform(image1)).to(device)

    unique_pixels = torch.unique(image1.view(-1))

    return len(unique_pixels) == 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python image-blank-filter.py <image1_path> <image2_path>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    if is_blank(image1_path) or is_blank(image2_path):
        print("1")
    else:
        print("0")
