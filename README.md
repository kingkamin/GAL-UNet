# GAL-UNet
This is the official code repository for "GAL-UNet: A Lightweight Solution for Precisely Detecting Large Size-Variant Defects in Imbalanced Fabric Samples".

**1. Prepare the dataset.**

- Our real fabric defect datasets, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

**2. Train the EGE-UNet.**
```
cd EGE-UNet
```
```
python train.py
```

**3. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
