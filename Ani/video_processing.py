import numpy as np
import cv2
import os
import shutil
import random
import subprocess


### Functions to facilitate processing RGB videos ###

# Get frame rate of raw videos
def get_video_frame_rate(video_path):
    """
    Get frame rate of a video file.

    Parameters:
    - video_path (str): Path to video file (.mp4 or .mov)
    
    Returns:
    - frame_rate (float): Frame rate of video, or None if it couldn't be determined
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        output = subprocess.check_output(command).decode().strip()
        if '/' in output:
            num, denom = output.split('/')
            frame_rate = float(num) / float(denom)
        else:
            frame_rate = float(output)  # Direct integer FPS case
        return frame_rate
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frame rate from {video_path}: {e}")
        return None


# Extract frames 
def extract_frames(input_dir, output_dir, skip_frames=0, custom_fps=None):
    """
    Extracts frames from videos in `input_dir`. Option to extract at original frame rate or a custom FPS.

    Parameters:
    - input_dir (str): Path to directory containing input video files.
    - output_dir (str): Path to directory where extracted frames will be stored.
    - skip_frames (int): Number of frames to skip between consecutive extractions.
    - custom_fps (float or None): If provided, extracts frames at this FPS instead of the video's original frame rate.
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        if video_file.endswith(('.mp4', '.mov')):
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(input_dir, video_file)
            
            # Get original frame rate
            frame_rate = get_video_frame_rate(video_path)
            if frame_rate is None:
                print(f"Skipping {video_file} due to frame rate detection error.")
                continue

            # Decide the FPS to use
            if custom_fps is not None:
                adjusted_frame_rate = custom_fps  # User-defined FPS
                print(f"Using custom FPS: {custom_fps} for {video_file}")
            else:
                adjusted_frame_rate = frame_rate / (skip_frames + 1)  # Compute based on original FPS
                
            # Create output directory for this video
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Construct FFmpeg command for frame extraction
            frame_output_path = os.path.join(video_output_dir, "frame_%04d.png")
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"fps={adjusted_frame_rate}",
                frame_output_path
            ]
            
            try:
                subprocess.run(ffmpeg_command, check=True)
                print(f"Frames extracted for video {video_file} at {adjusted_frame_rate} FPS.")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting frames from {video_file}: {e}")


### Crop video frames 
def crop_visible_fov(frame, threshold=10, reduction_x=250, reduction_y=250):
    """
    Crop visible content region of an octagonal FoV, removing black regions and text outside the visible FoV, 
    and reducing the bounding box size by specified pixels in x and y directions.
    
    Parameters:
        frame (np.array): Input frame as a numpy array.
        threshold (int): Threshold for detecting the black regions.
        reduction_x (int): Number of pixels to reduce from each side of the bounding box in x-direction.
        reduction_y (int): Number of pixels to reduce from each side of the bounding box in y-direction.
        
    Returns:
        np.array: Cropped frame with only the visible FoV.
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to highlight non-black regions
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black regions
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 2:
        contours, _ = result
    else:
        _, contours, _ = result

    if contours:
        # Find the largest contour, assuming it's the main content area
        max_contour = max(contours, key=cv2.contourArea)

        # Get a bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)

        # Reduce bounding box size by `reduction_x` and `reduction_y` pixels in each direction symmetrically about center
        x_new = max(x + reduction_x // 2, 0)
        y_new = max(y + reduction_y // 2, 0)
        w_new = max(w - reduction_x, 1)
        h_new = max(h - reduction_y, 1)

        # Ensure new width and height stay within frame bounds
        x_end = min(x_new + w_new, frame.shape[1])
        y_end = min(y_new + h_new, frame.shape[0])

        # Crop the frame to the adjusted bounding box
        cropped_frame = frame[y_new:y_end, x_new:x_end]

        # Create a mask of the largest contour to remove any residual text or black regions
        mask = np.zeros_like(frame[y_new:y_end, x_new:x_end])
        cv2.drawContours(mask, [max_contour - [x_new, y_new]], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Apply mask to cropped frame
        cropped_frame = cv2.bitwise_and(cropped_frame, mask)

        return cropped_frame
    else:
        # Return the original frame if no contours are found
        return frame

def process_frames(input_dir, output_dir, threshold=10):
    """
    Processes frames from input_dir, applies crop_visible_fov, and saves results in output_dir.
    
    Parameters:
        input_dir (str): Path to input directory with extracted frames.
        output_dir (str): Path to output directory for storing cropped frames.
        threshold (int): Threshold for detecting black regions.
    """
    # Walk through each video folder in input directory
    for video_folder in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        # Create corresponding output directory for video
        output_video_path = os.path.join(output_dir, video_folder)
        os.makedirs(output_video_path, exist_ok=True)

        # Process each frame in video folder
        for frame_name in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame_name)
            if not frame_name.endswith('.png'):
                continue

            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read {frame_path}")
                continue

            # Apply cropping
            cropped_frame = crop_visible_fov(frame, threshold=threshold)

            # Save cropped frame to the output directory
            output_frame_path = os.path.join(output_video_path, frame_name)
            cv2.imwrite(output_frame_path, cropped_frame)

        print(f"Processed: {video_folder}")
        
        
### Split videos into train and test sets
def split_videos(input_folder, output_folder, train_ratio=0.7, random_seed=42):
    """
    Splits videos into train and test sets and reorganizes frames into the desired structure.
    
    Parameters:
        input_folder (str): Path to folder containing subfolders for each video.
        output_folder (str): Path where train and test folders will be created.
        train_ratio (float): Proportion of videos to allocate to training set.
    """
    
    # Set random seed
    random.seed(random_seed)
    
    # Ensure output folders exist
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List all videos in input folder
    video_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    
    # Randomly shuffle and split into train and test sets
    random.shuffle(video_folders)
    split_index = int(len(video_folders) * train_ratio)
    train_videos = video_folders[:split_index]
    test_videos = video_folders[split_index:]

    # Function to move frames
    def move_frames(video_list, destination_folder):
        for video_name in video_list:
            src_folder = os.path.join(input_folder, video_name)
            dest_folder = os.path.join(destination_folder, video_name)
            shutil.copytree(src_folder, dest_folder)
    
    # Move frames to train and test folders
    move_frames(train_videos, train_folder)
    move_frames(test_videos, test_folder)
    
    print(f"Train/Test split complete: {len(train_videos)} videos in train, {len(test_videos)} videos in test.")
    print(f"Train folder: {train_folder}")
    print(f"Test folder: {test_folder}")
    
    
### Generate file lists for train and test sets
# data_folder: path to folder containing train and test sub-folders
def generate_file_list(data_folder, output_file, subset_folder):
    subset_path = os.path.join(data_folder, subset_folder)
    with open(output_file, 'w') as f:
        for video_name in os.listdir(subset_path):
            video_folder = os.path.join(subset_path, video_name)
            if os.path.isdir(video_folder):
                for frame in sorted(os.listdir(video_folder)):
                    f.write(f"{subset_folder}/{video_name}/{frame}\n")