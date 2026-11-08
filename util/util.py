import os
import cv2
import yaml
import shutil
import imageio
import numpy as np

# repo layout: this code lives in tells_results/tells-train, next to tells_results/{results,models,videos}.
#   RESULTS_ROOT/<env>/results_{fully_obs,allocentric,pf}/, config.yaml, cca/, noise_sweep/
#   MODELS_ROOT/marl/<run>/checkpoint<N>/            decentralized fully observable policies
#   MODELS_ROOT/belief/_<run>_NN_<i>/epoch<N>/*.ckpt belief models
# set TELLS_RESULTS to use a tells_results folder somewhere else.
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELLS_ROOT = os.path.abspath(os.environ.get('TELLS_RESULTS', os.path.dirname(CODE_ROOT)))
RESULTS_ROOT = os.path.join(TELLS_ROOT, 'results')
MODELS_ROOT = os.path.join(TELLS_ROOT, 'models')
VIDEOS_ROOT = os.path.join(TELLS_ROOT, 'videos')

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

    duration = 1000 / fps  # milliseconds per frame: imageio >= 2.28 (pillow plugin) reads GIF duration in ms, so the
    # old value 1 / fps (seconds) was written as 0 ms and viewers played the GIF as fast as they could
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


def find_run_config(results_dir: str) -> str:
    '''
    path of the training run's config.yaml for an evaluation results dir: results/<env>/config.yaml next to
    results/<env>/results_<mode>, else the run dir two levels up in the logs layout
    (<run_dir>/<checkpoint>/results_<mode>)
    '''
    results_dir = os.path.normpath(results_dir)
    for candidate in (os.path.join(os.path.dirname(results_dir), 'config.yaml'),
                      os.path.join(os.path.dirname(os.path.dirname(results_dir)), 'config.yaml')):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'no config.yaml next to or two levels above {results_dir}')
