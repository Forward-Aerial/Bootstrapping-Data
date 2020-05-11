import argparse
import glob
import os

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import patches

LOOKUP = {2: "Captain Falcon", 6: "Fox"}


def display_image(img: Image, prediction):
    # Visualize bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    for box, label, score in zip(
        prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"].round()
    ):
        if not score.item():
            continue
        [min_x, min_y, max_x, max_y] = box.tolist()
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle(
            (min_x, min_y), width, height, linewidth=1, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(min_x, min_y, LOOKUP[label.item()], c="r")
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def bootstrap(image_root: str, model_file: str):
    image_files = glob.iglob(os.path.join(image_root, "*"))
    model: torchvision.models.detection.FasterRCNN = torch.load("detector.pt")

    to_tensor = transforms.ToTensor()
    for image_file in image_files:
        img = Image.open(image_file)
        img_tensor = to_tensor(img).to("cuda")
        prediction = model([img_tensor])
        display_image(img, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_root",
        type=str,
        help="The directory that contains all of the image to generate annotations for.",
    )
    parser.add_argument(
        "model_file", type=str, help="Path to the PyTorch FasterRCNN model file."
    )
    args = parser.parse_args()
    bootstrap(args.image_root, args.model_file)
