import cv2
import numpy as np
import random
import json
import os


def progressive_augment_all(frame, frame_idx, total_frames, direction):
    h, w = frame.shape[:2]

    # hyperparameters
    scale = 1.0 + 0.5 * (frame_idx / total_frames)  # scale factor
    tx = min(20 * (frame_idx / total_frames), w // 2)  # horizontal
    ty = min(20 * (frame_idx / total_frames), h // 2)  # vertical
    angle = direction * 30 * (frame_idx / total_frames)  # rotation

    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    frame_translated = cv2.warpAffine(frame, translation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE, borderValue=0)

    new_w, new_h = int(w * scale), int(h * scale)
    frame_scaled = cv2.resize(frame_translated, (new_w, new_h))
    frame_scaled = cv2.resize(frame_scaled, (w, h)) 

    center = (w // 2, h // 2)
    matrix_rotate = cv2.getRotationMatrix2D(center, angle, 1)
    frame_rotated = cv2.warpAffine(frame_scaled, matrix_rotate, (w, h), borderMode=cv2.BORDER_REPLICATE, borderValue=0)

    return frame_rotated


def add_background_frames(video_path, num_frames, periods, frame_width, frame_height, color=(0, 0, 0)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_indices = set(range(total_frames))
    period_indices = {i for start, end in periods for i in range(start, end + 1)}
    non_period_indices = list(all_indices - period_indices)
    new_frames = []

    for _ in range(num_frames):
        add_type = random.choice(["blank"]) 
        if add_type == "blank":
            blank_frame = np.full((frame_height, frame_width, 3), color, dtype=np.uint8)
            new_frames.append(blank_frame)
        # else:
        #     random_idx = random.choice(non_period_indices)
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
        #     ret, frame = cap.read()
        #     if ret:
        #         new_frames.append(frame)

    cap.release()
    return new_frames


def update_labels(annotations, repeat_count, durations, speed_factors, num_background_start,
                                         num_background_end, final_frames):
    new_periods = []
    current_start = num_background_start

    for i in range(repeat_count):
        adjusted_duration = int(durations[i])
        new_end = current_start + adjusted_duration
        new_periods.append([current_start, new_end])
        current_start = new_end

    annotations['length'] = len(final_frames)
    annotations['object0']['period'] = new_periods
    annotations['object0']['count'] = len(new_periods)
    return annotations


def video_processing(video_path, json_path, output_path, updated_json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    video_name = annotations["video_name"]
    periods = annotations["object0"]["period"]

    cap = cv2.VideoCapture(video_path)
    fps = 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    period = random.choice(periods)
    start_frame, end_frame = period
    duration = end_frame - start_frame + 1

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(duration):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) < 30:
        repeat_count = random.randint(2, 15)  # Uniform(2, 15)
    elif 30 <= len(frames) < 100:
        repeat_count = random.randint(15, 30)  # Uniform(15, 30)
    elif 100 <= len(frames) <= 256:
        repeat_count = random.randint(30, 50)  # Uniform(30, 50)

    rotation_direction = random.choice([-1, 1])

    total_frames = len(frames) * repeat_count
    augmented_frames = []
    durations = []
    speed_factors = []

    for i in range(repeat_count):
        if len(frames) > 30:
            speed_factor = random.choice([1, 2])
        else:
            speed_factor = random.choice([0.5, 1])

        speed_factors.append(speed_factor)
        durations.append(duration)

        for j, frame in enumerate(frames):
            frame_idx = i * len(frames) + j
            augmented_frame = progressive_augment_all(frame, frame_idx, total_frames, rotation_direction)
            augmented_frames.append(augmented_frame)

    adjusted_frames = []
    current_frame_index = 0
    new_durations = []

    for i in range(repeat_count):
        new_duration = 0
        for j in range(len(frames)):
            if speed_factors[i] < 1.0:  # slow
                repeat_frame = int(1 / speed_factors[i])
                for _ in range(repeat_frame):
                    adjusted_frames.append(augmented_frames[current_frame_index])
                    new_duration += 1
            else:  # fast
                skip_frame = int(speed_factors[i])
                if j % skip_frame == 0:  
                    adjusted_frames.append(augmented_frames[current_frame_index])
                    new_duration += 1
            current_frame_index += 1
        new_durations.append(new_duration)

    print(durations)
    print(new_durations)
    print(speed_factors)
    num_background_start = random.randint(0, len(adjusted_frames) // 10)  # pre
    num_background_end = random.randint(0, len(adjusted_frames) // 10)  # post
    background_frames_start = add_background_frames(video_path, num_background_start, periods, frame_width,
                                                             frame_height, color=(0, 0, 0))
    background_frames_end = add_background_frames(video_path, num_background_end, periods, frame_width,
                                                           frame_height, color=(0, 0, 0))

    final_frames = background_frames_start + adjusted_frames + background_frames_end

    updated_annotations = update_labels(
        annotations, repeat_count, new_durations, speed_factors, num_background_start, num_background_end, final_frames
    )

    with open(updated_json_path, 'w') as f:
        json.dump(updated_annotations, f, indent=4)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame in final_frames:
        frame_resized = cv2.resize(frame, (frame_width, frame_height)) 
        out.write(frame_resized)

    out.release()
    print(f"Video saved to {output_path}")
    print(f"Labels saved to{updated_json_path}")


def file_processing(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)
        if os.path.isdir(subdir_path):
            video_file = None
            json_file = None
            for file in os.listdir(subdir_path):
                if file.endswith(".mp4"):
                    video_file = os.path.join(subdir_path, file)
                elif file.endswith(".json"):
                    json_file = os.path.join(subdir_path, file)

            if video_file and json_file:
                print(f"processing: {subdir}")
                output_subdir = os.path.join(output_folder, subdir)
                os.makedirs(output_subdir, exist_ok=True)

                output_video_path = os.path.join(output_subdir, f"{subdir}_augmented.mp4")
                updated_json_path = os.path.join(output_subdir, f"{subdir}_updated.json")

                video_processing(video_file, json_file, output_video_path, updated_json_path)

                print(f"Finished processing: {subdir}")
            else:
                print(f"File {subdir} does not contain both .mp4 and .json files")

    print("DONE!")


if __name__ == "__main__":
    root = "./RepCount/data"
    splits = ["train"]
    for i in splits:
        input_folder = os.path.join(root, i)
        output_folder = os.path.join(root, i + "_augmented")
        file_processing(input_folder, output_folder)