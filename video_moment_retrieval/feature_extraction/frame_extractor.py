import os
import tqdm
import numpy as np
from video_moment_retrieval.utils.utils import extract_frames_ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", type=str, required=True)
    parser.add_argument("-o", "--output_directory", type=str, required=True)
    parser.add_argument("-t", "--num_threads", type=int,
                        required=False, default=8)
    parser.add_argument("-f", "--num_frames", type=int,
                        required=False, default=100)
    return parser.parse_args()


def process_video(video_path: str, output_path: str, number_of_frames: int):
    try:
        if os.path.exists(output_path):
            try:
                arr = np.load(output_path)
                assert len(arr["frames"].shape) == 4
                return
            except:
                pass
        frames_array = np.array(
            extract_frames_ffmpeg(video_path, 32))
        np.savez_compressed(output_path, frames=frames_array)
    except Exception as e:
        print(e, file=sys.stderr)


def main():
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        videos = [video for video in os.listdir(
            args.input_directory) if video.endswith(".mp4")]
        futures = [executor.submit(process_video, os.path.join(args.input_directory, video), os.path.join(
            args.output_directory, video.replace(".mp4", ".npz")), args.num_frames) for video in videos]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
