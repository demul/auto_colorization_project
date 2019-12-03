import cv2
import matplotlib.pyplot as plt
import os

def show_result(src_path, num, show=False, save=False, dst_path='result.png'):
    size_figure_grid_y = 3
    size_figure_grid_x = 3

    edge_path_list = [os.path.join(src_path, 'edge', ('%4d'%i) + '.jpg') for i in range(num, num + size_figure_grid_y)]
    GT_path_list = [os.path.join(src_path, 'resized', ('%4d' %i) +'.jpg') for i in range(num, num + size_figure_grid_y)]
    generated_path_list = [os.path.join(src_path, 'PN', ('%4d' %i) + '.PNG') for i in range(num, num + size_figure_grid_y)]

    fig, ax = plt.subplots(size_figure_grid_y, size_figure_grid_x, figsize=(25, 25))
    for i in range(size_figure_grid_y):
        for j in range(size_figure_grid_x):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    for k in range(0, size_figure_grid_y * size_figure_grid_x, 3):
        i = k // size_figure_grid_y
        j = k % size_figure_grid_x
        ax[i, j].cla()
        ax[i, j].imshow(cv2.imread(edge_path_list[i], cv2.IMREAD_COLOR))
        ax[i, j + 1].cla()
        ax[i, j + 1].imshow(cv2.cvtColor(cv2.imread(GT_path_list[i], cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))
        ax[i, j + 2].cla()
        ax[i, j + 2].imshow(cv2.cvtColor(cv2.imread(generated_path_list[i], cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))
    label = 'Left : Edge, Center : GT, Right : Generated'
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(dst_path)

    if show:
        plt.show()
    else:
        plt.close()


show_result('test', 654, show=True, save=True)