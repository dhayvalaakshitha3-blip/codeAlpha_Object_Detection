import cv2
import time
import queue
import threading
import subprocess
from ultralytics import YOLO

# Vision Assist AI - Object Detection and Voice Assistance

MODEL_NAME = "yolo26n.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.45
ANNOUNCEMENT_DELAY = 3.0

speech_queue = queue.Queue()


def speech_worker():
    """Use Windows built-in speech without pyttsx3."""
    while True:
        message = speech_queue.get()

        if message is None:
            speech_queue.task_done()
            break

        try:
            safe_message = message.replace("'", "''")
            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{safe_message}')"
            )

            subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15
            )
        except Exception as error:
            print("Speech error:", error)
        finally:
            speech_queue.task_done()


def main():
    print("Loading Vision Assist AI model...")

    try:
        model = YOLO(MODEL_NAME)
    except Exception as error:
        print("Could not load the YOLO model:", error)
        print("Check your internet connection and model download.")
        return

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        print("Close other camera apps and check camera permissions.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    speaker = threading.Thread(target=speech_worker, daemon=True)
    speaker.start()

    last_announced = {}
    print("Vision Assist AI started. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the camera.")
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False
            )

            detected_names = set()

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    name = model.names[class_id]
                    detected_names.add(name)

            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                "Vision Assist AI | Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            now = time.time()

            for name in sorted(detected_names):
                previous_time = last_announced.get(name, 0)

                if now - previous_time >= ANNOUNCEMENT_DELAY:
                    # Avoid building up a long speech backlog.
                    if speech_queue.qsize() < 2:
                        speech_queue.put(name)
                        last_announced[name] = now

            cv2.imshow("Vision Assist AI - Object Detection",
                       annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping Vision Assist AI...")

    except Exception as error:
        print("Runtime error:", error)

    finally:
        camera.release()
        cv2.destroyAllWindows()

        try:
            speech_queue.put(None)
            speaker.join(timeout=2)
        except Exception:
            pass

        print("Vision Assist AI closed.")


if __name__ == "__main__":
    main()
