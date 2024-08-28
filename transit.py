import cv2
from imutils.video import VideoStream
import time
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
rtsp_base_url = "rtsp://admin:pt_otics1*@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
common_paths = [
    "/cam/realmonitor?channel=1&subtype=0"
]
def test_stream(path):
    rtsp_url = rtsp_base_url + path
    print(f"[INFO] testing {rtsp_url}")
    vs = VideoStream(rtsp_url).start()
    time.sleep(2.0) 
    frame = vs.read()
    vs.stop()
    return frame is not None
for path in common_paths:
    if test_stream(path):
        print(f"[INFO] stream path found: {path}")
        rtsp_url = rtsp_base_url + path
        break
else:
    print("[ERROR] no valid stream path found")
    exit()
print("[INFO] starting video stream...")

camera_stream = VideoStream(rtsp_url).start()
time.sleep(2.0)
while True:
    time.sleep(0.1)
    frame = camera_stream.read()
    if frame is None:
        break
    else:
        results = model(frame)
        annotated_frame = results[0].plot(line_width=2, labels=True, conf=False)
        resized_frame = cv2.resize(annotated_frame, (1395, 770))
        cv2.imshow("Deteksi Part HLA", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the display window
camera_stream.release()
cv2.destroyAllWindows()
