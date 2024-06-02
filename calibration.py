import cv2
import numpy as np
import os

# Directory to save captured images
save_dir = 'captured_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set the dimensions of the checkerboard
checkerboard_size = (8, 6)  # Number of inner corners per a chessboard row and column
square_size = 0.025  # Actual size of a square in meters

# Prepare object points based on the real-world coordinates of the checkerboard corners
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

img_count = 0

print("Press Enter to capture image, Space to finish capturing and start calibration")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter key
        img_name = os.path.join(save_dir, f'img_{img_count:03d}.jpg')
        cv2.imwrite(img_name, frame)
        img_count += 1
        print(f"Captured {img_name}")

    elif key == 32:  # Space key
        print("Finished capturing images")
        break

cap.release()
cv2.destroyAllWindows()

# Load captured images for calibration
images = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.jpg')]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration to obtain the camera matrix and distortion coefficients
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Extract the required values
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]
k1, k2, p1, p2, _ = dist_coeffs[0]  # Ignore k3

# Output the camera parameters
print(f"Camera.fx: {fx}")
print(f"Camera.fy: {fy}")
print(f"Camera.cx: {cx}")
print(f"Camera.cy: {cy}")
print(f"Camera.k1: {k1}")
print(f"Camera.k2: {k2}")
print(f"Camera.p1: {p1}")
print(f"Camera.p2: {p2}")
