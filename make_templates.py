import argparse
import os
import pathlib
import PIL

import cv2

from common import load_coco_data, load_dataset, minpt_wh_to_points


def main(dataset_location: str, template_directory: str):
    coco_dataset = load_dataset(dataset_location)
    for category in coco_dataset.dataset["categories"]:
        category_id = category["id"]
        pathlib.Path(os.path.join(template_directory, str(category_id))).mkdir(
            parents=True, exist_ok=True
        )
    for annotation in coco_dataset.dataset["annotations"]:
        category_id = annotation["category_id"]
        pathlib.Path(os.path.join(template_directory, str(category_id))).mkdir(
            parents=True, exist_ok=True
        )
        coco_img_data = coco_dataset.imgs[annotation["image_id"]]
        image_filename = coco_img_data["file_name"]
        image_path = os.path.join(dataset_location, "images", image_filename)
        with PIL.Image.open(image_path) as image:
            bbox = annotation["bbox"]
            top_left, _, bottom_right, _ = minpt_wh_to_points(bbox)
            template_image = image.crop((*top_left, *bottom_right))
            template_image.save(
                os.path.join(
                    template_directory, str(category_id), f"{annotation['id']}.jpg"
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_location",
        help="The location of the dataset to create templates from.",
        type=str,
    )
    parser.add_argument(
        "--template_directory",
        help="The directory to which templates should be output",
        type=str,
        default="templates",
    )

    args = parser.parse_args()
    main(args.dataset_location, args.template_directory)
