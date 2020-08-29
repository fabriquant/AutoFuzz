import cv2
import matplotlib.image as mpimg
import numpy as np

from torch.utils.data import Dataset
import pandas as pd
import os

import torch

import matplotlib.image as mpimg

# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 33, 100, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 80, 160, 3

# modification
# when we use udacity simulator
# ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_CHANNELS = 160, 320, 3
# when we use udacity simulator 0.9.6
# ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_CHANNELS = 160, 384, 3
# when we use carla simulator 0.9.9
ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_CHANNELS = 144, 256, 3

# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


# def rgb2yuv(image):
#     """
#     Convert the image from RGB to YUV (This is what the NVIDIA model does)
#     """
#     return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    # image = rgb2yuv(image)
    return image


def random_flip(image):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    return image


def random_translate(image, range_x, range_y):
    """
    Randomly shift the image virtually and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    # xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    xm, ym = np.mgrid[0:ORIGINAL_IMAGE_HEIGHT, 0:ORIGINAL_IMAGE_WIDTH] # fix

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(picture, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    # image = load_image(data_dir, picture)
    image = random_flip(picture)
    image = random_translate(image, range_x=range_x, range_y=range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    img_path = "{0}/{1}".format(data_dir, image_file)
    return mpimg.imread(img_path)


def normalize_and_reshape(x):
    x = x.astype('float32') / 255.
    x = np.moveaxis(x, -1, 0)
    # x = x.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS)
    return x

class Set_Wrapper(Dataset):
    def __init__(self, dataset, data_dir):
        self.x, self.y = dataset
        self.data_dir = data_dir

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = load_image(self.data_dir, x)
        x = preprocess(x)
        # x = augment(x)
        x = normalize_and_reshape(x)
        x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return self.x.shape[0]

# only use center for now
def load_data(data_dir, weather_indexes, route_indexes):
    x_center = None
    y = None
    Misbehavior = []
    for weather in weather_indexes:
        for route in route_indexes:

            route_str = str(route)
            if route < 10:
                route_str = '0'+route_str
            data_df = pd.read_csv(os.path.join(data_dir, 'route_'+route_str+'_'+str(weather), 'driving_log.csv'))

            if x_center is None:
                x_center = data_df['center'].values
                y = data_df['behaviors'].values
                # x_left = data_df['left'].values
                # x_right = data_df['right'].values
            else:
                x_center = np.concatenate((x_center, data_df['center'].values), axis=0)
                y = np.concatenate((y, data_df['behaviors'].values), axis=0)
                # x_left = numpy.concatenate((x_left, data_df['left'].values), axis=0)
                # x_right = numpy.concatenate((x_right, data_df['right'].values), axis=0)

            Misbehavior.extend(data_df['Misbehavior'])

    Misbehavior = np.array(Misbehavior)
    chosen_inds = (y == 0) | ((y==1)&((Misbehavior=='_collisions_vehicle')|(Misbehavior=='_collisions_layout')))

    x_center = x_center[chosen_inds]
    y = y[chosen_inds]

    print('y==0:', y[y==0].shape)
    print('y==1:', y[y==1].shape)
    return x_center, y
