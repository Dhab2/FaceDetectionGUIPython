import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Sawirka Canvas-ka lagu hayo
current_image = None


def upload_image():
    global current_image

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    # Fur sawirka
    image = Image.open(file_path)

    # Canvas size
    canvas_width = 700
    canvas_height = 450

    # Sawirka ku habbee Canvas-ka
    image.thumbnail(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS
    )

    # PIL -> Tkinter
    current_image = ImageTk.PhotoImage(image)

    # Canvas-ka nadiifi
    canvas.delete("all")

    # Sawirka dhex dhig Canvas-ka
    canvas.create_image(
        canvas_width // 2,
        canvas_height // 2,
        image=current_image,
        anchor="center"
    )


# Window
root = tk.Tk()
root.title("Face Detection")
root.geometry("800x600")

# Canvas
canvas = tk.Canvas(
    root,
    width=700,
    height=450,
    bg="gray20",
    highlightthickness=2
)
canvas.pack(pady=20)

# Upload Button
upload_button = tk.Button(
    root,
    text="Upload Image",
    font=("Arial", 12, "bold"),
    width=15,
    command=upload_image
)
upload_button.pack(pady=5)

root.mainloop()