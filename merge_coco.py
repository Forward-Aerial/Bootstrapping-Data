"""
Quick command-line tool to merge >= 2 COCO datasets together.
"""
import argparse
import copy
import functools
import json
import logging
import os
import pathlib
import shutil
from typing import Dict, List, Tuple

import skimage.io as io
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

from common import add_annotation, load_dataset

MIN_NUM_DATASETS = 2


def copy_image(
    new_img_id: int, old_img_dir: str, new_img_dir: str, old_img: Dict
) -> Dict:
    """
    Moves an image from old_img_dir to new_img_dir. Updates the filename and ID to "new_img_id".
    """
    logging.debug(f"Updating image entry: {old_img}.")
    logging.debug(f"New image ID is {new_img_id}.")
    new_img = copy.deepcopy(old_img)
    old_file_name = os.path.basename(old_img["file_name"])
    old_filepath = os.path.join(old_img_dir, "images", old_file_name)
    new_img["id"] = new_img_id
    new_filepath = os.path.join(new_img_dir, "images", f"{new_img['id']}.jpg")
    shutil.copy(old_filepath, new_filepath)

    new_img["file_name"] = f"{new_img['id']}.jpg"
    logging.debug(f"Image entry updated, is now {new_img}")
    return new_img


def update_annotation(new_img_id: int, new_annotation_id: int, old_ann: Dict,) -> Dict:
    """
    Updates the provided annotation's ID to "new_annotation_id" and the image ID to "new_img_id".
    """
    logging.debug(f"Updating annotation entry: {old_ann}.")
    logging.debug(f"Image ID = {new_img_id}, annotation offset = {new_annotation_id}.")
    new_ann = copy.deepcopy(old_ann)
    new_ann["id"] = new_annotation_id
    new_ann["image_id"] = new_img_id
    logging.debug(f"Annotation entry updated, is now {new_ann}")
    return new_ann


def assert_categories_are_equal(a_cats: Dict, b_cats: Dict):
    """
    Ensures that the provided categories are equal enough for merging (doesn't check for supercategory equality).
    "equal enough" means that the IDs are the same and the names for each category are identical.
    """
    assert len(a_cats) == len(
        b_cats
    ), "The number of categories must be identical in both COCO datasets."

    a_cat_keys = sorted(a_cats.keys())
    b_cat_keys = sorted(b_cats.keys())

    for a_cat_key, b_cat_key in zip(a_cat_keys, b_cat_keys):
        assert (
            a_cat_key == b_cat_key
        ), f"There are different category IDs present in the provided datasets: ({a_cat_key} != {b_cat_key})"
        a_cat_name = a_cats[a_cat_key]["name"]
        b_cat_name = b_cats[b_cat_key]["name"]
        assert (
            a_cat_name == b_cat_name
        ), f"Differing categories detected: ({a_cat_key} -> {a_cat_name} != {b_cat_key} -> {b_cat_name})"


def setup_ax(coco: COCO, coco_img_dir: str, img: Dict, ax):
    file_name = os.path.basename(img["file_name"])
    file_path = os.path.join(coco_img_dir, "images", file_name)
    print("Showing", file_path)
    im = Image.open(file_path)
    ax.imshow(im)
    corresponding_anns = coco.imgToAnns[img["id"]]
    for annotation in corresponding_anns:
        add_annotation(ax, annotation)


def verify_successful_transfer(
    coco_a: COCO,
    coco_a_img_dir: str,
    old_img: Dict,
    coco_b: COCO,
    coco_b_img_dir: str,
    new_img: Dict,
):
    fig, (ax, ax2) = plt.subplots(2, figsize=(24, 16))
    setup_ax(coco_a, coco_a_img_dir, old_img, ax)
    setup_ax(coco_b, coco_b_img_dir, new_img, ax2)
    fig.show()
    plt.show()


def merge_datasets(
    coco_tuple_a: Tuple[COCO, str], coco_tuple_b: Tuple[COCO, str]
) -> Tuple[COCO, str]:
    """
    Merges coco_tuple_a's images and annotations into coco_tuple_b.
    """
    coco_a, coco_a_img_dir = coco_tuple_a
    coco_b, coco_b_img_dir = coco_tuple_b

    logging.debug(
        f"Merging {pathlib.Path(coco_a_img_dir).parent} into {pathlib.Path(coco_b_img_dir).parent}"
    )

    assert_categories_are_equal(coco_a.cats, coco_b.cats)

    # Append images in coco_a to coco_b
    imgs = coco_a.imgs.values()
    for img in imgs:
        new_img_id = len(coco_b.dataset["images"])
        corresponding_anns = coco_a.imgToAnns[img["id"]]
        new_img = copy_image(new_img_id, coco_a_img_dir, coco_b_img_dir, img)
        for ann in corresponding_anns:
            new_ann_id = len(coco_b.dataset["annotations"])
            updated_ann = update_annotation(new_img_id, new_ann_id, ann)
            coco_b.dataset["annotations"].append(updated_ann)
        coco_b.dataset["images"].append(new_img)
        coco_b.createIndex()
        # verify_successful_transfer(
        #     coco_a, coco_a_img_dir, img, coco_b, coco_b_img_dir, new_img
        # )
        # clear = lambda: os.system("clear")
        # clear()

    return coco_b, coco_b_img_dir


def main(dataset_paths: List[str]):
    """
    Merges all of the provided datasets togethers into one large dataset.
    """
    datasets = map(load_dataset, dataset_paths)

    zipped = sorted(zip(datasets, dataset_paths), key=lambda x: len(x[0].imgs))

    mega_dataset, mega_dataset_path = functools.reduce(merge_datasets, zipped)

    mega_dataset_annotation_filepath = os.path.join(
        mega_dataset_path, "annotations", "instances_default.json"
    )
    with open(mega_dataset_annotation_filepath, "w") as mega_dataset_annotation_file:
        json.dump(mega_dataset.dataset, mega_dataset_annotation_file)
    print(
        f"Finished merging {len(dataset_paths)} datasets. Result is in {mega_dataset_path}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merges COCO datasets together.")
    parser.add_argument(
        "dataset_paths", nargs="*", help="Paths to the datasets to merge together."
    )
    args = parser.parse_args()

    if len(args.dataset_paths) < MIN_NUM_DATASETS:
        raise ValueError(
            f"At least {MIN_NUM_DATASETS} datasets must be provided. Received {len(args.dataset_paths)}."
        )
    main(args.dataset_paths)
