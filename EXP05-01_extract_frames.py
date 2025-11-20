import cv2
import os
import glob

def extract_frames(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{idx:04d}.png"), frame)
        idx += 1

    cap.release()
    print(f"Extracted {idx} frames to {output_dir}")

# --- MAIN SCRIPT LOGIC ---

base_dataset_dir = "dataset/people_snapshot_public"
base_output_dir  = "output"

# Specify your single subject folder name here
subject_name = "female-1-casual"
subject_path = os.path.join(base_dataset_dir, subject_name)

if not os.path.isdir(subject_path):
    print(f"Error: Subject folder not found: {subject_path}")
    exit()

# Find the first mp4 video in the given subject folder
video_search_pattern = os.path.join(subject_path, "*.mp4")
video_files = glob.glob(video_search_pattern)

if not video_files:
    print(f"No video file found for subject {subject_name}")
    exit()

video_file = video_files[0]
if len(video_files) > 1:
    print(f"Warning: Multiple videos found. Using {video_file}")

# Define output folder for this subject's extracted frames
output_folder = os.path.join(base_output_dir, subject_name, "images")

extract_frames(video_file, output_folder)
print("Processing complete.")
