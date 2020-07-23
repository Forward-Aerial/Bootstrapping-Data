import argparse
import os
import random

from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

from common import add_annotation, load_dataset


def main(dataset_path: str):
    dataset: COCO = load_dataset(dataset_path)

    
    image_dir = os.path.join(dataset_path, "images")
    imgs = random.choices(dataset.loadImgs(dataset.imgs.keys()), k=50)
    for img in imgs:
        print(img["file_name"])
        file_name = os.path.basename(img["file_name"])
        file_path = os.path.join(image_dir, file_name)
        im = Image.open(file_path)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(im)
        corresponding_anns = dataset.imgToAnns[img["id"]]
        print("Corresponding annotations =", corresponding_anns)
        for annotation in corresponding_anns:
            readable_name = dataset.loadCats(ids=annotation["category_id"])[0]["name"]
            # annotation["category_id"] = readable_name
            print(annotation["category_id"], "=", readable_name)
            add_annotation(ax, annotation)
        fig.show()
        plt.waitforbuttonpress()
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inspects a dataset for validity.")
    parser.add_argument(
        "dataset_path", type=str, help="The path of the dataset to investigate."
    )
    args = parser.parse_args()

    main(args.dataset_path)
