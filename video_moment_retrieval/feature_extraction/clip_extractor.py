import numpy as np
import ffmpeg
import os
import numpy.typing as npt
import torch
from transformers import CLIPProcessor, CLIPModel
import jsonlines
import click
import tqdm
from video_moment_retrieval.utils.logging import init_logging, logger
import logging
from typing import Any
import csv

model_checkpoint = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_checkpoint)
processor = CLIPProcessor.from_pretrained(model_checkpoint)
model.eval()

def processor_output_to(output, device: str):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in output.items()}

@torch.no_grad()
def encode_video(video_path: str, fps=0.5) -> npt.NDArray[np.float32]:
    video_info = ffmpeg.probe(video_path)
    frame_width = video_info["streams"][0]["width"]
    frame_height = video_info["streams"][0]["height"]

    process = ffmpeg.input(
        video_path
    ).output("pipe:", r=fps, format='rawvideo', pix_fmt="rgb24"
    ).run_async(pipe_stdout=True, pipe_stderr=True)
    video_array = []

    while True:
        in_bytes = process.stdout.read(frame_width * frame_height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape(
            (frame_height, frame_width, 3))
        video_array.append(in_frame)
    video_array = np.array(video_array, dtype=np.uint8)
    # process.kill()
    model_input: dict[str, Any] = processor(images=video_array, return_tensors="pt")
    model_input = processor_output_to(model_input, model.device)
    model_output= model.get_image_features(**model_input)
    return model_output.cpu().numpy()

@ torch.no_grad()
def encode_text(text: str) -> npt.NDArray[np.float32]:
    model_input = processor(text=text, return_tensors="pt")
    model_input = processor_output_to(model_input, model.device)
    model_output = model.text_model(**model_input)
    text_emmbs = model.text_projection(model_output.last_hidden_state)
    return text_emmbs.cpu().numpy()

def encode_videos(video_dir: str, out_dir: str, force: bool = False) -> None:
    logger.info("Processing videos in %s", video_dir)
    os.makedirs(out_dir, exist_ok=True)
    for video_file in tqdm.tqdm(os.listdir(video_dir)):
        video_output_path= os.path.join(out_dir, video_file.rsplit(".", 1)[0]) + ".npy"
        if not (force or not os.path.exists(video_output_path)):
            continue
        video_embedding = encode_video(os.path.join(video_dir, video_file), 0.49)
        np.save(video_output_path, video_embedding)

def encode_texts(text_file: str, out_dir: str, force: bool = False) -> None:
    logger.info("Processing queries in %s", text_file)
    with open(text_file, "r") as f:
        csv_reader= csv.reader(f)
        os.makedirs(out_dir, exist_ok=True)
        for qid, query in tqdm.tqdm(csv_reader):
            # Skip header
            if qid == "qid":
                continue
            query_output_path= os.path.join(out_dir, qid) + ".npy"
            if not (force or not os.path.exists(query_output_path)):
                continue
            text_embedding = np.squeeze(encode_text(query), 0)
            np.save(query_output_path, text_embedding)


def ensure_path_exists(path: str):
    return os.path.exists(path)

@click.command()
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--video_dir", required=False, help="path of directory containing videos")
@click.option("--video_output", required=False, help="output path for video features")
@click.option("--text_file", required=False, help="path to csv file containing a query id and value mapping")
@click.option("--text_output", required=False, help="output path for text features")
@click.option("--force", default=False, required=False, help="wether to regenerate embeddings if already present")
def extract_features_dataset(device: str, video_dir: str | None, video_output: str | None, text_file: str | None, text_output: str | None, force: bool = False) -> None:
    global model
    model= model.to(device)
    init_logging(root_level=logging.WARN)
    logger.info("Running feature extraction", extra={"device": device})

    if not (video_dir and video_output or text_file and text_output):
        logger.error(
            "At least one of the pairs (video_dir, video_output) and (text_file, text_output) needs to be provided")
        exit(1)

    if video_dir and not ensure_path_exists(video_dir):
        logger.error("Path %s does not exist", video_dir)
        exit(1)

    if text_file and not ensure_path_exists(text_file):
        logger.error("Path %s does not exist", text_file)
        exit(1)

    if video_dir:
        encode_videos(video_dir, video_output, force)

    if text_file:
        encode_texts(text_file, text_output, force)

if __name__ == "__main__":
    extract_features_dataset()
