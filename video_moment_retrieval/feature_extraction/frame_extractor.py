import os
import tqdm
import numpy as np
from video_moment_retrieval.utils.utils import extract_frames_ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed

video_root = "../videos"
output_root = "../frames"

os.makedirs(output_root, exist_ok=True)

def process_video(video: str):
    try:
        video_path = os.path.join(video_root, video)
        output_path = os.path.join(output_root, video.rsplit(".", 1)[0] + ".npy")
        if os.path.exists(output_path):
            return
        frames_array = np.array(extract_frames_ffmpeg(video_path, "uniform", 32, 224, 224))
        np.save(output_path, frames_array)
    except Exception as e:
        print(e)

with ThreadPoolExecutor(max_workers=10) as executor:
    videos = [video for video in os.listdir(video_root) if video.endswith(".mp4")]
    futures = [executor.submit(process_video, video) for video in videos]
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        future.result()
