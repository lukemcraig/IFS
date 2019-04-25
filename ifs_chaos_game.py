import numpy as np
import imageio
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


def goal2_transformations():
    # sierpinski triangle
    transformations = [get_2d_affine_transformation(tx=-.5, ty=.5, w=0.5, h=0.5),
                       get_2d_affine_transformation(tx=.5, ty=.5, w=0.5, h=0.5),
                       get_2d_affine_transformation(tx=0, ty=-.5, w=0.5, h=0.5)]
    final_transform = get_2d_affine_transformation()
    return final_transform, transformations


def goal1_transformations():
    # some custom affine transformation fractal
    transformations = [get_2d_affine_transformation(w=0.5, h=0.5),
                       get_2d_affine_transformation(tx=1, ty=1, theta=45, w=0.8, h=0.8)]
    final_transform = get_2d_affine_transformation(tx=-.95, w=0.8, h=-0.8)
    return final_transform, transformations


def main(n=200000, visualize_algorithm=False):
    n_first_iters_to_skip = 20

    pixels_width = 200
    pixels_height = 300

    gamma = 4.0

    # number of times a pixel is landed on
    point_frequencies = np.zeros((pixels_width, pixels_height))
    point_colors = np.zeros((pixels_width, pixels_height))
    function_colors = [0.4, 0.5, 1.0]
    # final_transform, transformations = goal2_transformations()
    final_transform, transformations = goal1_transformations()

    xy = np.random.uniform(low=-1.0, high=1.0, size=(1, 2))
    xy = np.append(xy, 1)
    for q in range(n):
        transformation_matrix_i = np.random.randint(0, len(transformations))
        # the randomly selected transformation (or function)
        transformation_matrix = transformations[transformation_matrix_i]
        xy = transformation_matrix @ xy
        xy_final = final_transform @ xy
        # skip the first few iterations
        if q >= n_first_iters_to_skip:
            # quantize point to the pixel grid
            pix_coord = np.round(((xy_final[0:2] + 1.0) / 2.0) * [pixels_width, pixels_height]).astype(int)
            # if the pixel is on the canvas
            if (pix_coord >= [0, 0]).all() and (pix_coord < [pixels_width, pixels_height]).all():
                # increase the pixel's histogram count
                point_frequencies[pix_coord[0], pix_coord[1]] += 1
                # shade the pixel based on the transformations color
                pixel_color = point_colors[pix_coord[0], pix_coord[1]]
                function_color = function_colors[transformation_matrix_i]
                point_colors[pix_coord[0], pix_coord[1]] = .5 * (pixel_color + function_color)
                if visualize_algorithm:
                    plt.imshow(point_frequencies, cmap='gray')
                    plt.colorbar()
                    # plt.scatter(0, 1)
                    plt.scatter(pix_coord[1], pix_coord[0])
                    plt.pause(.001)
                    plt.clf()
                pass
    alpha = np.log(point_frequencies) / np.log(point_frequencies.max())
    final_pixel_colors = point_colors * alpha ** (1 / gamma)

    # frequency_histogram = ((point_frequencies / point_frequencies.max()) * 255.0).astype(np.uint8)
    # imageio.imwrite('out.png', frequency_histogram, transparency=0)
    imageio.imwrite('result1.png', final_pixel_colors)
    return


main(visualize_algorithm=True)
