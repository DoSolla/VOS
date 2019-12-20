import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm


def caculateIoU(mask, annotation):
    if (mask.shape != annotation.shape):
        print("wrong shape")
    union = 0
    intersect = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0 or annotation[i, j] > 0:
                union += 1
                if mask[i, j] > 0 and annotation[i, j] > 0:
                    intersect += 1
    return 1. * intersect / union




mask_dir = '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/Result/mask/'
annotation_dir = '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/DAVIS/Annotations/480p/'

videos = os.listdir(mask_dir)
sum = 0
count = 0
for seqs in videos:
    if seqs == '.' or seqs == '..' or seqs == '.DS_Store':
        continue
    mask_path = mask_dir + seqs
    annotation_path = annotation_dir + seqs
    masks = np.sort(os.listdir(mask_path))
    annotations = np.sort(os.listdir(annotation_path))
    for i in tqdm.tqdm(range(len(masks))):
        mask = np.array(cv2.imread(mask_path + '/' + masks[i]))
        annotation = np.array(cv2.imread(annotation_path + '/' + annotations[i]))
        sum += caculateIoU(mask[:, :, 2], annotation[:, :, 2])
        count += 1

print(1. * sum / count)




