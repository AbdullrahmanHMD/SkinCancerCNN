import os
from moleDataset import MoleDataset
import cv2

image_shape = (600, 450)

dataset = MoleDataset()

la = dataset[0][0]

import time
arr = []

tic = time.time()
for x, y, z in dataset:
    arr.append(x)
toc = time.time()

print(f'Exec time: {toc - tic}')


# cv2.imshow("Image 1", la)
# cv2.waitKey(0)



# print(dataset.labels.loc[dataset.labels[COLUMN_NAME] == image_id])


# print(dataset.images_paths)
# print(dataset.load_labels())

# print(dataset.labels.loc[dataset.labels[COLUMN_NAME == image_id]])

# cv2.imread()
# cv2.imshow("Image", image.reshape(image_shape))
# cv2.waitKey(0)
