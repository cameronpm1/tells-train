import os
import cv2
import yaml
import shutil
import imageio
import numpy as np

def mkdir(folder):
    if os.path.islink(folder):
        # keep the symlink, clear the contents of its target
        target = os.path.realpath(folder)
        if os.path.isdir(target):
            for entry in os.listdir(target):
                path = os.path.join(target, entry)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            os.makedirs(target)
        return
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def save_argb_video(images, output_path, fps=30):

    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264' for H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in images:
        out.write(frame)

    out.release()

def save_rgb_gif(images, output_path, fps=10, loop=None):
    '''loop: number of times the gif repeats (0 = forever); None writes no loop setting, so viewers play it once'''

    duration = 1 / fps  # seconds per frame
    kwargs = {} if loop is None else {'loop': loop}
    imageio.mimsave(output_path, images, duration=duration, **kwargs)

def save_cv2_images_as_gif(images, output_path, fps=60):
    """
    images: list of cv2 images (BGR numpy arrays)
    output_path: path to save the gif (e.g., "output.gif")
    fps: frames per second
    """

    if len(images) == 0:
        raise ValueError("Image list is empty")

    rgb_frames = []

    for img in images:
        if img is None:
            continue

        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb)

    duration = 1 / fps

    imageio.mimsave(output_path, rgb_frames, duration=duration)

def furthest_point(xy, vw, x0y0, d):
    '''
    solves for the furthest point from x,y traveling in direction v,w 
    from x0, y0 that is no more than distance d
    '''

    x = xy[0]
    y = xy[1]
    v = vw[0]
    w = vw[1]
    x0 = x0y0[0]
    y0 = x0y0[1]

    a = v**2 + w**2
    if a == 0:
        return np.array([x, y])

    b = 2 * ((x - x0) * v + (y - y0) * w)
    cq = (x - x0)**2 + (y - y0)**2 - d**2

    disc = b**2 - 4 * a * cq

    if disc < 0:
        return np.array([x,y])

    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)

    t_max = max(t1, t2)

    if t_max < 0:
        # Direction points away from feasible circle
        t_max = 0

    return np.array([x + t_max * v, y + t_max * w])

