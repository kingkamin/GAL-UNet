# GAL-UNet
This is the official code repository for "GAL-UNet: A Lightweight Solution for Precisely Detecting Large Size-Variant Defects in Imbalanced Fabric Samples".

**1. Prepare the dataset.**

- Our real fabric defect datasets, can be found here {[Baidu](https://pan.baidu.com) or [GoogleDrive](https://drive.google.com)}. 

- After downloading the datasets, you are supposed to put them into './data/', and the file format reference is as follows.

- './data/'
  - data_train
    - images
      - .bmp
    - masks
      - .png
  - data_test
    - images
      - .bmp
    - masks
      - .png

**2. Train the GAL-UNet.**
```
cd GAL-UNet
```
```
python train.py
```

**3. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
