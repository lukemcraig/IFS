import random

import numpy as np
# import matplotlib.pyplot as plt
import scipy.misc
import imageio


def get_2d_affine_transformation(tx=0.0, ty=0.0, theta=0.0, w=1.0, h=1.0):
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    scale_matrix = np.array([[w, 0, 0],
                             [0, h, 0],
                             [0, 0, 1]])
    return translation_matrix @ rotation_matrix @ scale_matrix


def main(n=5000):
    pixels_width = 800
    pixels_height = 800
    frequency_histogram = np.zeros((pixels_width, pixels_height))
    transformations = [get_2d_affine_transformation(w=0.5, h=0.5),
                       get_2d_affine_transformation(tx=1, ty=1, theta=45, w=0.8, h=0.8)]

    xy = np.random.uniform(low=-1.0, high=1.0, size=(1, 2))
    xy = np.append(xy, 1)
    for q in range(n):
        transformation_matrix = random.choice(transformations)
        xy = transformation_matrix @ xy
        if q >= 20:
            # plt.scatter(xy[0], xy[1], c='k', alpha=0.1, marker='.')
            pixel_coords = np.round(((xy[0:2] + 1.0) / 2.0) * [pixels_width, pixels_height]).astype(int)
            if (pixel_coords >= [0, 0]).all():
                if (pixel_coords < [pixels_width, pixels_height]).all():
                    frequency_histogram[pixel_coords[0], pixel_coords[1]] += 1
                    # frequency_histogram[]
                    pass

    # widthheight = 2.5
    # left = 0
    # bottom = -1
    # plt.xlim(left, left + widthheight)
    # plt.ylim(bottom, bottom + widthheight)
    # plt.show()
    # scipy.misc.imsave('out.jpg', frequency_histogram)

    frequency_histogram = ((frequency_histogram / frequency_histogram.max()) * 255).astype(np.uint8)
    imageio.imwrite('out.png', frequency_histogram)
    return


main()
