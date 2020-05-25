# Bootstrapping-Data
Bootstrapping labeling using mediocre models.

## Setup
1. Have a version of Python 3 installed (Python 3.8 was used for developing this library).
2. `pip install -r requirements.txt`
3. `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'` 
4. Ensure that you have a somewhat-trained Faster-RCNN model from PyTorch available and serialized.

### Scripts

#### bootstrapping.py
Because manually annotating frame after frame is time-consuming and expensive, I've decided that I'm going to "bootstrap" my data.
This script takes a mediocre ML model and runs it over a directory of unlabeled images, generating rough labels and bounding boxes for those images.
The images and labels are then transformed into COCO format, and saved.
They can then be uploaded into [CVAT](https://cvat.org/), where the labels and boxes can be corrected by hand if necessary.
I can then download the refined data from CVAT as a COCO dataset.

To use this script, you'll need the following:

1. A PyTorch-saved `FasterRCNN` model (the more trained, the less manual intervention required)
2. A directory full of unlabeled images

Then
```shell_script
python bootstrap.py <image_dir> <model_path> [--certainty] [--template_location]
```

#### merge_coco.py
Once a datasets are refined and downloaded from CVAT, they need to be merged together to form a single dataset.
This dataset can be used to further retrain the model used in `bootstrapping.py`, which can then be used to yield better bootstrapping results, over and over.

To use this script, you'll need the following:

1. At least 2 labeled datasets in the following format:
```txt
<root_dir>/annotations/instances_default.json # <-- Annotation file
<root_dir>/images # <-- Images directory
```

Then
```shell_script
python merge_coco.py <dataset_1_root> <dataset_2_root> ...
```

The script will tell you which dataset folder has the the results in them.