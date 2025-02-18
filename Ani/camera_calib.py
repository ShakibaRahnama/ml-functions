import numpy as np
import cv2
import os
import glob


# Compute camera calibration parameters
def calibrate_camera(calibration_images_dir, checkerboard_size, sensor_width_mm, image_width_pixels, output_dir):
    """
    Performs camera calibration using checkerboard images and converts focal length to mm.

    Args:
        calibration_images_dir (str): Directory containing selected checkerboard images.
        checkerboard_size (tuple): Number of inner corners in the checkerboard (rows, cols).
        sensor_width_mm (float): Physical sensor width of the camera in millimeters.
        image_width_pixels (int): Width of the image in pixels.
        output_dir (str): Directory to save images with detected corners.

    Returns:
        camera_matrix (np.ndarray): 3x3 intrinsic camera matrix.
        distortion_coeffs (np.ndarray): Distortion coefficients.
    """
    # Prepare object points (3D world coordinates)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    # Create output directory for saving images
    os.makedirs(output_dir, exist_ok=True)

    # Read all checkerboard images
    images = glob.glob(os.path.join(calibration_images_dir, "*.png"))

    if not images:
        print("No images found for calibration. Check the directory path.")
        return None, None

    for img_file in images:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and save detected corners
            img_with_corners = cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
            output_image_path = os.path.join(output_dir, os.path.basename(img_file))
            cv2.imwrite(output_image_path, img_with_corners)
            print(f"Saved detected corners in: {output_image_path}")

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\n✅ Camera Calibration Successful!")
        print("Camera Matrix:\n", camera_matrix)

        # Extract focal lengths in pixels
        f_x_pixels = camera_matrix[0, 0]  # Focal length along X in pixels
        f_y_pixels = camera_matrix[1, 1]  # Focal length along Y in pixels

        # Convert to mm
        f_x_mm = (f_x_pixels * sensor_width_mm) / image_width_pixels
        f_y_mm = (f_y_pixels * sensor_width_mm) / image_width_pixels

        print(f"Focal Length in Pixels: fx={f_x_pixels}, fy={f_y_pixels}")
        print(f"Focal Length in mm: fx={f_x_mm:.2f} mm, fy={f_y_mm:.2f} mm")
        print(f"Focal Length in cm: fx={f_x_mm/10:.2f} cm, fy={f_y_mm/10:.2f} cm")

        # Save the calibration results
        np.save(os.path.join(output_dir, "camera_intrinsics.npy"), camera_matrix)
        np.save(os.path.join(output_dir, "distortion_coeffs.npy"), dist_coeffs)

        return camera_matrix, dist_coeffs, f_x_mm, f_y_mm
    else:
        print("❌ Camera Calibration Failed.")
        return None, None, None, None