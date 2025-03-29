
# Auto Extract Keypoints From Detection Annotated Dataset

This repo is being modified to extract keypoints


## Usage

### Prepare Dataset

Make sure the dataset is being annotated by YOLOv8 format 

The dataset directory looks like
```
📁 your_dataset/
├── 📁 train/
│   ├── 📁 images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── 📁 labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── 📁 valid/
│   ├── 📁 images/
│   └── 📁 labels/
├── 📁 test/              
│   ├── 📁 images/
│   └── 📁 labels/
```

After that change the folder path config in `configs/base_config.py`, modify the directory in `folder_path`

Eg:

```
_C.folder_path = "your_dataset/train/images"
```

### Run

First, change the config in `configs/base_config.py` and adjust some parameters such as `matched_over_thresh`, `euclid_thresh` based on how bbox your dataset is being annotated. Second, MUST change the parameter `target_class`, which is the class id you want to annotate, for other class ids will be labeled as 0 (which is the normal state). The `normalize_kps` is a parameter to normalize the keypoints extracted (including rotating the image and keypoints by the shoulders and hips, also normalizing the keypoints by the bbox)

```
python extract_kps.py
```