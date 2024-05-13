import cv2
import easyocr
import threading

# Load the Haar Cascade classifier for license plate detection
harcascade = "models\license_plate_detector.pt"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Initialize multiple webcams
camera_indices = [2]  # Example: Two cameras (indices 0 and 1)
cameras = [cv2.VideoCapture(idx) for idx in camera_indices]


# Set camera resolutions (adjust as needed)
for cam in cameras:
    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

min_area = 500
count = 0

def process_camera(cam_idx):
    global count 
    while True:
        ret, frame = cameras[cam_idx].read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect license plates
        plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                # Draw a rectangle around the license plate
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 2)
                cv2.putText(frame, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                # Extract the license plate region
                img_roi = frame[y: y + h, x: x + w]

                # Perform OCR on the license plate
                reader = easyocr.Reader(['en'])
                output = reader.readtext(img_roi)

                # Display the recognized text
                if output:
                    plate_text = output[0][1]
                    print(plate_text)
                    cv2.putText(frame, f"Plate: {plate_text}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
        # # Display the processed frame
        cv2.imshow(f"Camera {cam_idx}", frame)

        # # Save the detected plate when 's' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     cv2.imwrite(f"plates/scaned_img_{count}_cam{cam_idx}.jpg", img_roi)
        #     cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        #     cv2.putText(frame, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        #     cv2.imshow(f"Camera {cam_idx} Results", frame)
        #     cv2.waitKey(500)
        #     count += 1

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # Release the webcam
    cameras[cam_idx].release()

# Create threads for each camera
threads = [threading.Thread(target=process_camera, args=(idx,)) for idx in range(len(cameras))]

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()
    

# Close all windows
cv2.destroyAllWindows()
