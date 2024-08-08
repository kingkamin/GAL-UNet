# GAL-UNet
This is the official code repository for "GAL-UNet: Lightweight Detection of Diverse Defects in Fabric with Imbalance Resilience".

**1. Prepare the dataset.**

- Our real fabric defect datasets, can be found [here](https://pan.baidu.com/s/18rKV6xNr5EBqE58t9aENPw?pwd=8dpz).

- The reason for fewer json files than images in the dataset is that some images do not contain any defects, hence there are no corresponding json files for them.

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
