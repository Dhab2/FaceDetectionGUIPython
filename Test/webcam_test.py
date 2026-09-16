import argparse
import cv2
from insightface.app import FaceAnalysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="InsightFace CPU-only webcam face detection"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index"
    )

    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Detector input size"
    )

    args = parser.parse_args()

    # CPU ONLY
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=(args.det_size, args.det_size)
    )

    # Open webcam
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam {args.camera}"
        )

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read frame from webcam.")
                break

            # Detect faces
            faces = app.get(frame)

            # Draw face boxes
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                score = float(face.det_score)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Face {score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Show number of faces
            cv2.putText(
                frame,
                f"Faces: {len(faces)} | Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Display webcam
            cv2.imshow(
                "InsightFace CPU Webcam Detection",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()