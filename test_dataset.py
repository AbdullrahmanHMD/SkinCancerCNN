import os
from moleDataset import MoleDataset
import cv2

image_shape = (600, 450)

dataset = MoleDataset()

image = dataset.images_paths[100]
print(image)
print(os.path.basename(os.path.normpath(image)))


COLUMN_NAME = "image_id"

image_id = os.path.basename(os.path.normpath(image)).split('.')[0]
print(image_id)

print(dataset.labels[100])

# print(dataset.labels.loc[dataset.labels[COLUMN_NAME] == image_id])


# print(dataset.images_paths)
# print(dataset.load_labels())

# print(dataset.labels.loc[dataset.labels[COLUMN_NAME == image_id]])

# cv2.imread()
# cv2.imshow("Image", image.reshape(image_shape))
# cv2.waitKey(0)
