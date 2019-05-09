import matplotlib

# matplotlib.use("Agg")
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib import rcParams
import scipy.ndimage.filters


def setup_visualization(axes, transformations):
    ims = []
    axes[0].set_title("Selected Function (Equal $p$)")
    y_labels = ["$f_" + str(i) + "$" for i in range(len(transformations))]
    axes[0].set_yticks(np.arange(len(transformations)))
    axes[0].set_yticklabels(y_labels)
    for j, transf in enumerate(transformations):
        left = 0
        width = 1
        right = left + width
        height = 1
        bottom = (height * j) - .5
        top = bottom + height
        p = patches.Rectangle(
            (left, bottom), width, height,
            fill=False, clip_on=False, alpha=0
        )
        axes[0].add_patch(p)

        axes[0].text(0.5 * (left + right), 0.5 * (bottom + top),
                     r"$ \begin{bmatrix} %.1f & %.1f & %.1f \\ %.1f & %.1f & %.1f \\ %.1f & %.1f & %.1f \end{bmatrix} $" % (
                         transf[0, 0],
                         transf[0, 1],
                         transf[0, 2],
                         transf[1, 0],
                         transf[1, 1],
                         transf[1, 2],
                         transf[2, 0],
                         transf[2, 1],
                         transf[2, 2]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20)
        pass
    return ims


def add_next_visualization_animation_frame(axes, fig, ims, pix_coord, point_frequencies, q, transformation_matrix_i):
    axes[1].clear()
    transf_scatter = axes[0].scatter(0, transformation_matrix_i, color='blue')
    im = axes[1].imshow(point_frequencies, cmap='gray')
    # plt.scatter(0, 1)
    axes[1].scatter(pix_coord[1], pix_coord[0])
    cb = fig.colorbar(im, ax=axes[1])
    # plt.pause(.0001)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ims.append(image)
    print(q)
    cb.remove()
    transf_scatter.remove()
    # plt.clf()
    pass


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


def adaptive_filter(density_alpha, max_kernel_radius, point_frequencies):
    # http://www.eecs.ucf.edu/seniordesign/su2011fa2011/g12/SD1_report.pdf
    # https://pdfs.semanticscholar.org/4522/05fd45452b2963b0d4d998128dba233987d5.pdf
    # https://flam3.com/flame_draves.pdf
    # https://en.wikipedia.org/wiki/Fractal_flame#Density_Estimation

    # higher density_alpha means smaller minimum kernel width
    kernel_width = max_kernel_radius / (point_frequencies ** density_alpha)

    filtered_histogram = np.zeros_like(point_frequencies)
    unique_values = np.unique(kernel_width)
    # print(unique_values[-1])
    # filtered_histogram = scipy.ndimage.filters.gaussian_filter(point_frequencies, sigma=unique_values[0])
    # return filtered_histogram
    for unique_value in unique_values:
        index = kernel_width == unique_value
        this_filtered_image = scipy.ndimage.filters.gaussian_filter(point_frequencies, sigma=unique_value)
        # plt.imshow(this_filtered_image)
        # plt.show()
        filtered_histogram[index] = this_filtered_image[index]
    # for i in range(filtered_histogram.shape[0]):
    #     for j in range(filtered_histogram.shape[1]):
    #         print(i, j)
    #         filtered_histogram[i, j] = scipy.ndimage.filters.gaussian_filter(point_frequencies,
    #                                                                          sigma=kernel_width[i, j])[i, j]
    return filtered_histogram


def gamma_correction(gamma, point_colors, point_frequencies):
    # point_frequencies += np.finfo(float).eps
    point_frequencies_max = point_frequencies.max()
    alpha = np.log(point_frequencies) / np.log(point_frequencies_max)
    # alpha[alpha < 0] = 0
    # alpha += alpha.min()
    final_pixel_colors = point_colors * (alpha ** (1 / gamma))[:, :, None]
    return final_pixel_colors


def main(n=200000, visualize_algorithm=False):
    if visualize_algorithm:
        fig, axes = plt.subplots(1, 2)
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    n_first_iters_to_skip = 20

    pixels_width = 900
    pixels_height = 900

    gamma = 4.0

    # number of times a pixel is landed on
    point_frequencies = np.zeros((pixels_width, pixels_height))
    point_colors = np.zeros((pixels_width, pixels_height, 3))
    function_colors = np.array([[0.3, 1.0, 0.8], [0.3, 0.7, 1.0]])
    # final_transform, transformations = goal2_transformations()
    final_transform, transformations = goal1_transformations()

    if visualize_algorithm:
        ims = setup_visualization(axes, transformations)

    xy = np.random.uniform(low=-1.0, high=1.0, size=(1, 2))
    xy = np.append(xy, 1)
    for q in range(n):
        print(q)
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
                    add_next_visualization_animation_frame(axes, fig, ims, pix_coord, point_frequencies, q,
                                                           transformation_matrix_i)
    if visualize_algorithm:
        imageio.mimsave('./visualization.gif', ims, fps=10)

    max_kernel_radius = 50
    density_alpha = 3
    epsilon = 1
    process_and_save(density_alpha=density_alpha, epsilon=epsilon, gamma=gamma, max_kernel_radius=max_kernel_radius,
                     point_colors=point_colors, point_frequencies=point_frequencies)
    return


def process_and_save(density_alpha, epsilon, gamma, max_kernel_radius, point_colors, point_frequencies):
    point_frequencies += epsilon
    adaptive_filtering = True
    if adaptive_filtering:
        point_frequencies_filtered = adaptive_filter(density_alpha, max_kernel_radius, point_frequencies)
        final_pixel_colors_filtered = gamma_correction(gamma, point_colors, point_frequencies_filtered)
        imageio.imwrite('result_adaptive_filtered.png', final_pixel_colors_filtered)

    final_pixel_colors = gamma_correction(gamma, point_colors, point_frequencies)

    # final_pixel_colors = ((final_pixel_colors / final_pixel_colors.max()) * 255.0).astype(np.uint8)
    # imageio.imwrite('out.png', final_pixel_colors, transparency=0)
    imageio.imwrite('result_not_filtered.png', final_pixel_colors)
    return


# main(n=500, visualize_algorithm=True)
main(n=50000)
