import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from time import time

from src.page_dewarp.image import WarpedImage


def plot_grid(img, dx=100, dy=100):
    img = np.copy(img)
    # Custom (rgb) grid color
    grid_color = [0,0,0]

    # Modify the image to include the grid
    img[::dy,:, :] = grid_color
    img[:,::dx,:] = grid_color

    return img

def correct_alignment(img_file, **kwargs):
    if isinstance(img_file, np.ndarray):
        in_img = img_file
        img_name = str(time())
    else:
        img_file = Path(img_file)
        img_name = str(img_file.name)
        img_name = img_name[:-4]
        in_img = cv2.imread(str(img_file))
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)

    out = WarpedImage(in_img)
    out_img = np.array(out.outimg)

    save_dir = kwargs.pop("save_dir", " ")
    
    if save_dir:
        save_dir = Path(save_dir)

        in_img = cv2.resize(in_img, (1024, 1024), interpolation=cv2.INTER_AREA)
        out_img = cv2.resize(out_img, (1024, 1024), interpolation=cv2.INTER_AREA)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 30), sharey=True)
        ax1.set_title('Initial Image')
        ax1.imshow(plot_grid(in_img))
        ax2.set_title('Post Processed Output')
        ax2.imshow(plot_grid(out_img))

        file_name = save_dir / f'{img_name}.png'
        plt.savefig(file_name)
        plt.close(fig)

    return out_img



if __name__ == "__main__":
    img_file = "./test_2.jpg"
    save_dir = "./"
    correct_alignment(img_file, save_dir=save_dir)

