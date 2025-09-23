# viewer.py - Simple Matplotlib image viewer with Prev/Next buttons and arrow keys
# Usage:
#   python viewer.py               # opens images from ./outputs
#   python viewer.py <folder_path> # opens images from specified folder

try:
    import numpy as np  # not strictly needed but often available
except Exception:
    np = None

import os
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency 'matplotlib'. Please install with: pip install -r requirements.txt") from e

# For PNG/JPG reading, matplotlib uses Pillow; ensure it's installed
try:
    import PIL  # noqa: F401
except ModuleNotFoundError:
    # We don't hard fail here because some backends can still read certain formats,
    # but we provide a clear hint if loading fails later.
    pass

import matplotlib.image as mpimg


def collect_images(folder):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = []
    if not os.path.isdir(folder):
        return files
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(os.path.join(folder, name))
    return files


def run_viewer(folder):
    images = collect_images(folder)
    if not images:
        print(f"No images found in: {folder}\nGenerate figures first (run main.py) or choose another folder.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.18)  # make room for buttons
    ax.set_axis_off()

    state = {'idx': 0}

    def show(idx):
        ax.clear()
        ax.set_axis_off()
        path = images[idx]
        try:
            img = mpimg.imread(path)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(path)}\n{e}", ha='center', va='center')
            fig.canvas.draw_idle()
            return
        ax.imshow(img)
        ax.set_title(f"{os.path.basename(path)}  ({idx+1}/{len(images)})")
        fig.canvas.draw_idle()

    # Buttons
    axprev = plt.axes([0.25, 0.05, 0.2, 0.08])
    axnext = plt.axes([0.55, 0.05, 0.2, 0.08])
    bprev = Button(axprev, 'Prev')
    bnext = Button(axnext, 'Next')

    def on_prev(event):
        state['idx'] = (state['idx'] - 1) % len(images)
        show(state['idx'])

    def on_next(event):
        state['idx'] = (state['idx'] + 1) % len(images)
        show(state['idx'])

    bprev.on_clicked(on_prev)
    bnext.on_clicked(on_next)

    # Keyboard bindings: Left/Right arrows, A/D
    def on_key(event):
        if event.key in ('left', 'a'):
            on_prev(None)
        elif event.key in ('right', 'd'):
            on_next(None)

    fig.canvas.mpl_connect('key_press_event', on_key)

    show(state['idx'])
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = os.path.join(os.getcwd(), 'outputs')
    run_viewer(folder)
