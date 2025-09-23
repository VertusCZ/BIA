# tk_viewer.py - Simple Tkinter window application to browse images
# Usage:
#   python tk_viewer.py                 # opens images from ./outputs
#   python tk_viewer.py <folder_path>   # opens images from specified folder
#
# Controls:
#   - Buttons: Prev / Next
#   - Keyboard: Left/Right arrows or A/D
#   - Window resizes images to fit while keeping aspect ratio

import os
import sys
import glob
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency 'Pillow'. Please install with: pip install -r requirements.txt") from e

SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def collect_images(folder: str):
    if not os.path.isdir(folder):
        return []
    files = []
    for pattern in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files.sort()
    return files


class ImageViewer(tk.Tk):
    def __init__(self, folder: str):
        super().__init__()
        self.title("Image Viewer")
        self.minsize(600, 400)
        self.folder = folder
        self.paths = collect_images(folder)
        self.idx = 0
        self.current_img = None  # PIL Image
        self.tk_img = None       # ImageTk.PhotoImage held by reference

        # Layout: image area (Canvas) + control frame
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(self)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)

        self.prev_btn = tk.Button(ctrl, text="◀ Prev", width=10, command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=8, pady=6)

        self.label = tk.Label(ctrl, text="0/0", anchor="center")
        self.label.pack(side=tk.LEFT, expand=True)

        self.next_btn = tk.Button(ctrl, text="Next ▶", width=10, command=self.next_image)
        self.next_btn.pack(side=tk.RIGHT, padx=8, pady=6)

        # Bindings
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("a", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("d", lambda e: self.next_image())
        self.bind("<Configure>", self._on_resize)

        if not self.paths:
            msg = f"No images found in: {folder}\nGenerate figures first (run main.py) or choose another folder."
            self.canvas.create_text(
                10, 10,
                anchor="nw",
                text=msg,
                fill="white",
                font=("Segoe UI", 11)
            )
            # Disable navigation
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.label.config(text="0/0")
        else:
            self.show_image(0)

    def _on_resize(self, event):
        # Redraw image to fit the new size
        if self.paths:
            self._render_current_to_canvas()

    def prev_image(self):
        if not self.paths:
            return
        self.idx = (self.idx - 1) % len(self.paths)
        self.show_image(self.idx)

    def next_image(self):
        if not self.paths:
            return
        self.idx = (self.idx + 1) % len(self.paths)
        self.show_image(self.idx)

    def show_image(self, idx: int):
        path = self.paths[idx]
        try:
            self.current_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load {os.path.basename(path)}\n{e}")
            return
        self.idx = idx
        self.label.config(text=f"{idx+1}/{len(self.paths)} - {os.path.basename(path)}")
        self._render_current_to_canvas()

    def _render_current_to_canvas(self):
        if self.current_img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 2 or ch <= 2:
            return
        img = self.current_img
        iw, ih = img.size
        # Fit to canvas while preserving aspect ratio
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = img.resize((nw, nh), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        # Center the image
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, image=self.tk_img, anchor="nw")


def main():
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = os.path.join(os.getcwd(), "outputs")
    app = ImageViewer(folder)
    app.mainloop()


if __name__ == "__main__":
    main()
