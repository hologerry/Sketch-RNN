import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch


def make_image(sequence, iter, img_output_path):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    img_save_path = os.path.join(img_output_path, f"{iter:06d}.png")
    pil_image.save(img_save_path, "JPEG")
    plt.close("all")
