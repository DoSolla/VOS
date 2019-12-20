import os
from dataloaders.helpers import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
mask_dir = '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/Result/mask/dog150_test/'
img_dir = '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/DAVIS/JPEGImages/480p/dog/'
save_dir = '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/Result/test/dog150/'

masks = os.listdir(mask_dir)
imgs = os.listdir(img_dir)

masks.sort()
imgs.sort()

for i in range(len(masks)):
    mask_path = mask_dir + masks[i]
    img_path = img_dir + imgs[i]
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    # cv2.imshow(img)
    # cv2.imshow(mask)
    overlay_img = overlay_mask(im_normalize(img), mask)
    mpimg.imsave(save_dir+imgs[i], overlay_img)





