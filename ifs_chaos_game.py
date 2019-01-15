import random

import numpy as np
import matplotlib.pyplot as plt


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


def main(n=1000):
    transformations = [get_2d_affine_transformation(), get_2d_affine_transformation(tx=1, ty=1, theta=45, w=0.8, h=0.8)]

    xy = np.random.uniform(low=-1.0, high=1.0, size=(1, 2))
    xy = np.append(xy, 1)
    for q in range(n):
        transformation_matrix = random.choice(transformations)
        xy = transformation_matrix @ xy
        plt.scatter(xy[0], xy[1])
        
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()
    return


main()
