"""
Quick command-line tool to merge >= 2 COCO datasets together.
"""
import argparse
import functools
import json
import logging
import os
import pathlib
from typing import Dict, List, Tuple

import skimage.io as io
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

MIN_NUM_DATASETS = 2


def load_dataset(dataset_path: str):
    """
    Loads a COCO dataset from a specific folder. Expects folder to be similar to:
    <dir>/annotations/instances_default.json
    <dir>/images/
    """
    annotation_filepath = os.path.join(
        dataset_path, "annotations", "instances_default.json"
    )
    logging.debug(f"Attempting to open {annotation_filepath}.")
    return COCO(annotation_filepath)


def update_image(
    start_id: int, old_img_dir: str, new_img_dir: str, image: Dict
) -> Dict:
    """
    Moves an image from old_img_dir to new_img_dir. Updates the filename and ID to be offset from "start_id".
    """
    logging.debug(f"Updating image entry: {image}.")
    logging.debug(f"Image offset is {start_id}.")
    old_filepath = os.path.join(old_img_dir, "images", f"{image['id']}.jpg")
    image["id"] += start_id
    new_filepath = os.path.join(new_img_dir, "images", f"{image['id']}.jpg")
    os.rename(old_filepath, new_filepath)
    image["file_name"] = f"{image['id']}.jpg"
    logging.debug(f"Image entry updated, is now {image}")
    return image


def update_annotation(
    start_image_id: int, start_annotation_id: int, annotation: Dict,
) -> Dict:
    """
    Offsets the provided annotation's ID by "start_annotation_id" and the image ID by "start_image_id".
    """
    logging.debug(f"Updating annotation entry: {annotation}.")
    logging.debug(
        f"Image offset = {start_image_id}, annotation offset = {start_annotation_id}."
    )
    annotation["id"] += start_annotation_id
    annotation["image_id"] += start_image_id
    logging.debug(f"Annotation entry updated, is now {annotation}")
    return annotation


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


def merge_datasets(
    coco_tuple_a: Tuple[COCO, str], coco_tuple_b: Tuple[COCO, str]
) -> COCO:
    """
    Merges coco_tuple_a's images and annotations into coco_tuple_b.
    """
    coco_a, coco_a_img_dir = coco_tuple_a
    coco_b, coco_b_img_dir = coco_tuple_b

    logging.debug(
        f"Merging {pathlib.Path(coco_a_img_dir).parent} into {pathlib.Path(coco_b_img_dir).parent}"
    )

    assert_categories_are_equal(coco_a.cats, coco_b.cats)

    # We're merging coco_a into coco_b, so how many annotations/images are in coco_b?
    num_imgs = len(coco_b.imgs)
    num_anns = len(coco_b.anns)

    # Append images in coco_a to coco_b
    img_ids = coco_a.imgs.keys()
    images = coco_a.loadImgs(ids=img_ids)
    images = map(
        lambda x: update_image(num_imgs, coco_a_img_dir, coco_b_img_dir, x), images
    )
    coco_b.dataset["images"] += images
    logging.debug(f"Images merged successfully.")

    # Append annotations in min_coco to max_coco
    ann_ids = coco_a.anns.keys()
    annotations = coco_a.loadAnns(ids=ann_ids)
    annotations = list(
        map(lambda x: update_annotation(num_imgs, num_anns, x), annotations,)
    )
    coco_b.dataset["annotations"] += annotations
    logging.debug(f"Annotations merged succesfully.")
    coco_b.createIndex()

    return coco_b


def main(dataset_paths: List[str]):
    """
    Merges all of the provided datasets togethers into one large dataset.
    """
    datasets = map(load_dataset, dataset_paths)

    zipped = sorted(zip(datasets, dataset_paths), key=lambda x: len(x[0].imgs))

    mega_dataset: COCO = functools.reduce(merge_datasets, zipped)
    mega_dataset_path: str = zipped[-1][-1]

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
