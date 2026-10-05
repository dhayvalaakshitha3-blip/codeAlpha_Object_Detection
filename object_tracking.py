import cv2
from ultralytics import YOLO


# ---------------------------------------
# Load the pretrained YOLO model
# ---------------------------------------
model = YOLO("yolo26n.pt")


# ---------------------------------------
# Open laptop webcam
# ---------------------------------------
cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open the webcam.")

    exit()


print("Camera started.")
print("Press Q to stop the program.")


# ---------------------------------------
# Process webcam frames
# ---------------------------------------
while True:

    success, frame = cap.read()


    if not success:

        print("ERROR: Could not read camera frame.")

        break


    # -----------------------------------
    # Run YOLO object tracking
    # -----------------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4
    )


    # -----------------------------------
    # Draw boxes, labels and IDs
    # -----------------------------------
    annotated_frame = results[0].plot()


    # -----------------------------------
    # Display result
    # -----------------------------------
    cv2.imshow(
        "CodeAlpha - Object Detection and Tracking",
        annotated_frame
    )


    # -----------------------------------
    # Press Q to exit
    # -----------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ---------------------------------------
# Release camera
# ---------------------------------------
cap.release()

cv2.destroyAllWindows()

print("Camera stopped.")