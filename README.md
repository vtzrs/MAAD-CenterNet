# MAAD-CenterNet

This repository contains the **official implementation** of the paper:

**"From Web Data to Real Fields: Low-Cost Unsupervised Domain Adaptation for Agricultural Robots"**  
*Vasileios Tzouras, Lazaros Nalpantidis, and Ronja Güldenring*  
*Scandinavian Conference on Image Analysis (SCIA), 2025*

---

## Abstract

In precision agriculture, vision models often struggle with new, unseen fields where crops and weeds have been influenced by external factors, resulting in compositions and appearances that differ from the learned distribution. This paper aims to adapt to specific fields at low cost using Unsupervised Domain Adaptation (UDA). We explore a novel domain shift from a diverse, large pool of internet-sourced data to a small set of data collected by a robot at specific locations, minimizing the need for extensive on-field data collection. Additionally, we introduce a novel module--the Multi-level Attention-based Adversarial Discriminator (MAAD)--which can be integrated at the feature extractor level of any detection model. In this study, we incorporate MAAD with CenterNet to simultaneously detect leaf, stem, and vein instances. Our results show significant performance improvements in the unlabeled target domain compared to baseline models, with a 7.5\% increase in object detection accuracy and a 5.1\% improvement in keypoint detection.

---

## Start Guide

### 1. Clone the repository

```bash
git clone https://github.com/vtzrs/MAAD-CenterNet.git
cd MAAD-CenterNet
```

### 2. Create Conda environment 

```bash
conda create --name MAAD-CenterNet python=3.8 --no-default-packages -y
conda activate MAAD-CenterNet
```

### 3. Install requirements

```bash
conda install --file requirements.txt
```
  
### 4. Download and prepare dataset

```bash
wget -O data/RumexLeaves.zip https://data.dtu.dk/ndownloader/files/41521812
python maad_centernet/data/make_dataset.py data/raw data/processed
```

### 5. Data directory setup

After Step 4, your data directory should be organized as follows:  

```bash
data/processed/RumexLeaves/
```

#### iNaturalist Folder
Manually move all images and required files into the `iNaturalist` folder so that it has the following structure:  

```bash
data/processed/RumexLeaves/iNaturalist/
│── annotations.xml
│── annotations_oriented_bb.xml
│── random_test.txt
│── random_train.txt
│── random_val.txt
│── references.txt
│── image1.jpg
│── image2.jpg
│── ...
```

#### RoboRumex Folder
- Move **all images** from sequence subfolders into the main `RoboRumex/` folder.  
- Place dataset split files (`random_test.txt`, `random_train.txt`, `random_val.txt`) **in the same directory** as the images.  
- Merge `annotations.xml` and `annotations_oriented_bb.xml` from each sequence into a **single** file and place them in `RoboRumex/`.

```bash
data/processed/RumexLeaves/RoboRumex/
│── annotations.xml
│── annotations_oriented_bb.xml
│── random_test.txt
│── random_train.txt
│── random_val.txt
│── image1.png
│── image2.png
│── ...
```

### 6. Train model from scratch

```bash
python maad_centernet/tools/train.py --exp_file exp_files/train_final_model.py
```

### 7. Continue training from the last checkpoint

```bash
python maad_centernet/tools/train.py --exp_file exp_files/train_final_model.py --resume --start_epoch epoch_number
```

### 8. Run validation of final model on RoboRumex data

```bash
python maad_centernet/tools/evaluate.py --exp_file exp_files/eval_roborumex.py -ckpt log/train/final_model/train_final_model/latest_ckpt.pth
```

---

## Citation

If you find our work useful, please consider citing it:

```bibtex
@inproceedings{MAAD-CenterNet,
  author    = {Tzouras, Vasileios and Nalpantidis, Lazaros and Güldenring, Ronja},
  title     = {From Web Data to Real Fields: Low-Cost Unsupervised Domain Adaptation for Agricultural Robots},
  booktitle = {Proceedings of the Scandinavian Conference on Image Analysis (SCIA)},
  year      = {2025}
}
```

---

## Acknowledgements

This codebase is partially based on the [RumexLeaves-CenterNet](https://github.com/DTU-PAS/RumexLeaves-CenterNet) repository.

---
