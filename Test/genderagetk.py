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
# GLOBAL VARIABLES
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

        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        image = Image.open(file_path).convert("RGB")

        # PIL -> NumPy
        frame = np.array(image)

        # RGB -> BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # -------------------------------------------------
        # FACE DETECTION
        # -------------------------------------------------

        faces = app.get(frame)

        if len(faces) == 0:

            picture_box.config(
                image="",
                text="No Face Detected"
            )

            status_label.config(
                text="No Face Detected",
                fg="red"
            )

            name_entry.delete(0, tk.END)
            age_entry.delete(0, tk.END)
            gender_var.set("Unknown")

            return

        # -------------------------------------------------
        # PROCESS ALL FACES
        # -------------------------------------------------

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            # ---------------------------------------------
            # AGE
            # ---------------------------------------------

            age = int(face.age)

            # ---------------------------------------------
            # GENDER
            # ---------------------------------------------

            gender_value = int(face.gender)

            if gender_value == 1:
                gender = "Male"
            else:
                gender = "Female"

            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            confidence = float(face.det_score) * 100

            # ---------------------------------------------
            # LABEL
            # ---------------------------------------------

            label = f"{gender} | Age: {age}"

            confidence_text = f"{confidence:.1f}%"

            # ---------------------------------------------
            # GREEN FACE BOX
            # ---------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ---------------------------------------------
            # LABEL BACKGROUND
            # ---------------------------------------------

            label_y1 = max(0, y1 - 65)

            cv2.rectangle(
                frame,
                (x1, label_y1),
                (x1 + 220, y1),
                (0, 255, 0),
                -1
            )

            # ---------------------------------------------
            # GENDER + AGE
            # ---------------------------------------------

            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            cv2.putText(
                frame,
                f"Face: {confidence_text}",
                (x1 + 5, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

        # -------------------------------------------------
        # SHOW FIRST FACE DATA IN FORM
        # -------------------------------------------------

        first_face = faces[0]

        detected_age = int(first_face.age)

        detected_gender_value = int(first_face.gender)

        if detected_gender_value == 1:
            detected_gender = "Male"
        else:
            detected_gender = "Female"

        # Age
        age_entry.delete(0, tk.END)
        age_entry.insert(0, str(detected_age))

        # Gender
        gender_var.set(detected_gender)

        # -------------------------------------------------
        # BGR -> RGB
        # -------------------------------------------------

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # NumPy -> PIL
        image = Image.fromarray(frame)

        # -------------------------------------------------
        # PICTURE BOX SIZE
        # -------------------------------------------------

        image.thumbnail(
            (580, 330),
            Image.Resampling.LANCZOS
        )

        # PIL -> Tkinter
        current_image = ImageTk.PhotoImage(image)

        # Show image
        picture_box.config(
            image=current_image,
            text=""
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        status_label.config(
            text=f"{len(faces)} Face(s) Detected",
            fg="green"
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            f"Something went wrong:\n\n{e}"
        )


# =====================================================
# NEW PERSON
# =====================================================

def new_person():

    global current_image

    current_image = None

    picture_box.config(
        image="",
        text="No Image"
    )

    name_entry.delete(
        0,
        tk.END
    )

    age_entry.delete(
        0,
        tk.END
    )

    gender_var.set(
        "Unknown"
    )

    status_label.config(
        text="New Person",
        fg="black"
    )


# =====================================================
# SAVE PERSON
# =====================================================

def save_person():

    name = name_entry.get().strip()
    age = age_entry.get().strip()
    gender = gender_var.get()

    if not name:

        messagebox.showwarning(
            "Missing Name",
            "Please enter the person's name."
        )

        return

    if not age:

        messagebox.showwarning(
            "Missing Age",
            "Please upload an image first."
        )

        return

    if gender == "Unknown":

        messagebox.showwarning(
            "Missing Gender",
            "Please upload an image first."
        )

        return

    messagebox.showinfo(
        "Person Saved",
        f"Person saved successfully!\n\n"
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}"
    )


# =====================================================
# MAIN WINDOW
# =====================================================

root = tk.Tk()

root.title(
    "Face Detection - New Person"
)

root.geometry(
    "950x600"
)

root.resizable(
    False,
    False
)


# =====================================================
# TITLE
# =====================================================

title_label = tk.Label(
    root,
    text="Face Detection System",
    font=("Arial", 20, "bold")
)

title_label.place(
    x=0,
    y=15,
    width=950,
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

picture_box.place(
    x=40,
    y=70,
    width=600,
    height=350
)


# =====================================================
# UPLOAD BUTTON
# =====================================================

upload_button = tk.Button(
    root,
    text="Upload Image",
    font=("Arial", 11, "bold"),
    command=upload_image
)

upload_button.place(
    x=250,
    y=435,
    width=170,
    height=35
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
    x=40,
    y=480,
    width=600,
    height=25
)


# =====================================================
# FORM TITLE
# =====================================================

form_title = tk.Label(
    root,
    text="New Person",
    font=("Arial", 16, "bold")
)

form_title.place(
    x=680,
    y=75,
    width=220,
    height=30
)


# =====================================================
# NAME
# =====================================================

name_label = tk.Label(
    root,
    text="Name:",
    font=("Arial", 11)
)

name_label.place(
    x=680,
    y=125
)

name_entry = tk.Entry(
    root,
    font=("Arial", 11)
)

name_entry.place(
    x=680,
    y=150,
    width=220,
    height=30
)


# =====================================================
# AGE
# =====================================================

age_label = tk.Label(
    root,
    text="Age:",
    font=("Arial", 11)
)

age_label.place(
    x=680,
    y=195
)

age_entry = tk.Entry(
    root,
    font=("Arial", 11)
)

age_entry.place(
    x=680,
    y=220,
    width=220,
    height=30
)


# =====================================================
# GENDER
# =====================================================

gender_label = tk.Label(
    root,
    text="Gender:",
    font=("Arial", 11)
)

gender_label.place(
    x=680,
    y=265
)

gender_var = tk.StringVar(
    value="Unknown"
)

gender_menu = tk.OptionMenu(
    root,
    gender_var,
    "Male",
    "Female",
    "Unknown"
)

gender_menu.config(
    font=("Arial", 10)
)

gender_menu.place(
    x=680,
    y=290,
    width=220,
    height=35
)


# =====================================================
# NEW PERSON BUTTON
# =====================================================

new_button = tk.Button(
    root,
    text="New Person",
    font=("Arial", 11, "bold"),
    command=new_person
)

new_button.place(
    x=680,
    y=350,
    width=105,
    height=35
)


# =====================================================
# SAVE BUTTON
# =====================================================

save_button = tk.Button(
    root,
    text="Save Person",
    font=("Arial", 11, "bold"),
    command=save_person
)

save_button.place(
    x=795,
    y=350,
    width=105,
    height=35
)


# =====================================================
# START
# =====================================================

root.mainloop()