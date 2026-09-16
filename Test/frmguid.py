import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis


# =====================================================
# INSIGHTFACE
# =====================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)


# =====================================================
# GLOBAL
# =====================================================

current_image = None


# =====================================================
# UPLOAD IMAGE
# =====================================================

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

    try:

        # Open image
        image = Image.open(file_path).convert("RGB")

        # PIL -> NumPy
        frame = np.array(image)

        # RGB -> BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # =================================================
        # FACE DETECTION
        # =================================================

        faces = app.get(frame)

        # =================================================
        # DRAW FACE BOX
        # =================================================

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            # Green rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Confidence
            score = float(face.det_score)

            text = f"Face: {score * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        # =================================================
        # PICTURE BOX SIZE
        # =================================================

        image.thumbnail(
            (580, 330),
            Image.Resampling.LANCZOS
        )

        current_image = ImageTk.PhotoImage(image)

        # Clear PictureBox
        picture_box.config(
            image=current_image,
            text=""
        )

        # Status
        if len(faces) == 0:
            status_label.config(
                text="No Face Detected"
            )
        else:
            status_label.config(
                text=f"{len(faces)} Face(s) Detected"
            )

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )


# =====================================================
# WINDOW
# =====================================================

root = tk.Tk()

root.title("Face Detection")

# IMPORTANT: smaller window
root.geometry("700x520")

root.resizable(False, False)


# =====================================================
# TITLE
# =====================================================

title_label = tk.Label(
    root,
    text="Face Detection System",
    font=("Arial", 18, "bold")
)

title_label.place(
    x=0,
    y=15,
    width=700,
    height=35
)


# =====================================================
# PICTURE BOX
# =====================================================

picture_box = tk.Label(
    root,
    text="No Image",
    bg="gray20",
    fg="white",
    relief="solid",
    borderwidth=2
)

# EXACT SIZE AND POSITION
picture_box.place(
    x=50,
    y=65,
    width=600,
    height=350
)


# =====================================================
# STATUS
# =====================================================

status_label = tk.Label(
    root,
    text="Please upload an image",
    font=("Arial", 11, "bold")
)

status_label.place(
    x=0,
    y=425,
    width=700,
    height=25
)


# =====================================================
# UPLOAD BUTTON
# =====================================================

upload_button = tk.Button(
    root,
    text="Upload Image",
    font=("Arial", 12, "bold"),
    command=upload_image
)

upload_button.place(
    x=275,
    y=460,
    width=150,
    height=35
)


# =====================================================
# START
# =====================================================

root.mainloop()