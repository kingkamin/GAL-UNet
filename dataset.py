from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

class MY_datasets(Dataset):
    def __init__(self, config, train=True):
        super(MY_datasets, self)
        if train:
            images_list = os.listdir(config.path_of_image_for_train)
            masks_list = os.listdir(config.path_of_label_for_train)
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = config.path_of_image_for_train + images_list[i]
                mask_path = config.path_of_label_for_train + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(config.path_of_image_for_test)
            masks_list = os.listdir(config.path_of_label_for_test)
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = config.path_of_image_for_test + images_list[i]
                mask_path = config.path_of_label_for_test + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        self.num_classes = config.num_classes
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = Image.open(img_path)
        size_h = img.size[1]
        size_w = img.size[0]
        img = np.expand_dims(np.array(img), axis=2)
        seg = np.array(Image.open(msk_path))
        msk = np.expand_dims(seg, axis=2)
        img, msk = self.transformer((img, msk))
        seg_labels = np.eye(self.num_classes)[seg.reshape([-1])]
        seg_labels = seg_labels.reshape((size_h, size_w, self.num_classes))
        return img, msk, seg_labels

    def __len__(self):
        return len(self.data)
