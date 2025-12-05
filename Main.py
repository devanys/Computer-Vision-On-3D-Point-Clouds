import cv2
import numpy as np
from openni import openni2
import math
import time

openni2.initialize()
dev = openni2.Device.open_any()
color_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()
color_stream.start()
depth_stream.start()

fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5

angle_y = 0
zoom = 1.5
fullscreen = False
threshold_distance = 1.0  

target_fps = 30
frame_time = 1.0 / target_fps

print("Streaming Point Cloud + Polar Histogram + Sector Data (Real View, 30 FPS)")
print("[A/D]=Rotasi | [W/S]=Zoom | [F]=Fullscreen | [Q]=Keluar")

def draw_fov_overlay(canvas):
    h, w = canvas.shape[:2]
    cx = w // 2
    lines = [-1.0, -0.5, 0.0, 0.5, 1.0]
    labels = ["Kanan", "Kanan-Depan", "Tengah", "Kiri-Depan", "Kiri"]
    for i, l in enumerate(lines):
        x = int(cx + l * (w // 2))
        cv2.line(canvas, (x, 0), (x, h), (80, 80, 80), 1)
        cv2.putText(canvas, f"{labels[i]} ({int(i*45)}°)", (x + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)

def draw_polar_histogram(canvas, histogram, max_val=2.0):
    center = (1080, 600)
    radius = 100
    cv2.circle(canvas, center, radius, (255, 255, 255), 1)
    sectors = len(histogram)
    for i, val in enumerate(histogram):
        theta = math.radians(i * (180 / sectors))
        r = radius * (val / max_val)
        x = int(center[0] + r * math.cos(theta))
        y = int(center[1] - r * math.sin(theta))
        cv2.line(canvas, center, (x, y), (0, 255, 0), 2)
    cv2.putText(canvas, "Polar Histogram (VFH)", (center[0]-100, center[1]+radius+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.namedWindow("Kinect Point Cloud (VFH)", cv2.WINDOW_NORMAL)
sector_labels = ["Kanan", "Kanan-Depan", "Tengah", "Kiri-Depan", "Kiri", "Ekstra"]

while True:
    start_time = time.time()

    color_frame = color_stream.read_frame()
    depth_frame = depth_stream.read_frame()
    color_data = color_frame.get_buffer_as_uint8()
    depth_data = depth_frame.get_buffer_as_uint16()

    rgb = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)
    depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
    depth = depth.astype(np.float32) / 1000.0  # mm → m

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    valid = depth > 0
    z = depth[valid]
    x = (c[valid] - cx) * z / fx
    y = (r[valid] - cy) * z / fy

    theta = math.radians(angle_y)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * x + sin_t * z
    z_rot = -sin_t * x + cos_t * z

    x_vis = -x_rot
    x_2d = (x_vis * fx * zoom / (z_rot + 0.5)) + 320
    y_2d = (y * fy * zoom / (z_rot + 0.5)) + 240
    colors = rgb[valid]

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    scale_x, scale_y = 1280 / 640, 720 / 480
    x_2d = (x_2d * scale_x).astype(np.int32)
    y_2d = (y_2d * scale_y).astype(np.int32)
    mask = (x_2d >= 0) & (x_2d < 1280) & (y_2d >= 0) & (y_2d < 720)
    x_2d = x_2d[mask]
    y_2d = y_2d[mask]
    colors = colors[mask]
    canvas[y_2d, x_2d] = colors

    num_sectors = 6
    histogram = np.zeros(num_sectors)
    sector_status = {}
    theta_point = np.arctan2(x_rot, z_rot)
    theta_point = theta_point % np.pi 

    for i in range(num_sectors):
        theta_min = i * (np.pi / num_sectors)
        theta_max = (i + 1) * (np.pi / num_sectors)
        mask_sector = (theta_point >= theta_min) & (theta_point < theta_max)
        
        if np.any(mask_sector):
            min_distance = np.min(z[mask_sector])
            mean_distance = np.mean(z[mask_sector])
            
            histogram[i] = min_distance if min_distance < 2 else 2
            sector_status[sector_labels[i]] = "High" if min_distance < threshold_distance else "Low"
        else:
            
            histogram[i] = 2
            sector_status[sector_labels[i]] = "Low"

    draw_fov_overlay(canvas)
    draw_polar_histogram(canvas, histogram)

    h, w = canvas.shape[:2]
    sector_positions = [int(w*0.1), int(w*0.3), int(w*0.5), int(w*0.7), int(w*0.9)]
    for i, label in enumerate(sector_labels[:5]):
        status = sector_status[label]
        color = (0, 0, 255) if status == "High" else (0, 255, 0)
        cv2.putText(canvas, f"{label}: {status}", (sector_positions[i]-40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(canvas, "[A/D]=Rotasi | [W/S]=Zoom | [F]=Fullscreen | [Q]=Keluar",
                (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    cv2.imshow("Kinect Point Cloud (VFH)", canvas)
    print(sector_status)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        angle_y -= 5
    elif key == ord('d'):
        angle_y += 5
    elif key == ord('w'):
        zoom *= 1.1
    elif key == ord('s'):
        zoom /= 1.1
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Kinect Point Cloud (VFH)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Kinect Point Cloud (VFH)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    elapsed = time.time() - start_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)

color_stream.stop()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
