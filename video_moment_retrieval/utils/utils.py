import torch
import torch.nn as nn
from typing import overload, Any
import numpy.typing as npt
import numpy as np
import ffmpeg
from decord import VideoReader
from math import ceil


@overload
def center_to_edges(box: torch.FloatTensor) -> torch.FloatTensor:
    ...
    
@overload
def center_to_edges(box: npt.NDArray) -> npt.NDArray:
    ...

def center_to_edges(box) -> Any:
    if isinstance(box, torch.Tensor):
        center, width = box.unbind(-1)
        return torch.stack(
            [center - 0.5 * width, center + 0.5 * width],
            dim=-1
        )
    elif isinstance(box, np.ndarray):
        center, width = np.split(box, 2, axis=-1)
        center = np.squeeze(center, axis=-1)
        width = np.squeeze(width, axis=-1)
        return np.stack(
            [center - 0.5 * width, center + 0.5 * width],
            axis=-1
        )
    
    raise NotImplementedError(f"{center_to_edges.__name__} not impplemented for type {type(box)}")

@overload
def edges_to_center(box: list[int | float]) -> list[int | float]:
    ...
    
@overload
def edges_to_center(box: torch.FloatTensor) -> torch.FloatTensor:
    ...

def edges_to_center(box) -> Any:
    if isinstance(box, torch.Tensor):
        left, right = box.unbind(-1)
        center = (right + left) / 2
        length = right - left
        return torch.stack((center, length), dim=-1)

    elif isinstance(box, (list, tuple)):  # note: only words with single dimensional tuple/list
        left, right = box[0], box[1]
        center = float((right + left)) / 2
        length = float(right - left)
        return [center, length]
    
    raise NotImplementedError(f"{edges_to_center.__name__} not implemented for type {type(box)}")


def count_parameters(model: nn.Module, name = "root", depth = 1):
    def count_module_params(module: nn.Module):
        return sum(p.numel() for p in module.parameters())
    if depth <= 0:
        return
    params = {name: count_module_params(model), "children": []}
    if depth - 1 <= 0:
        return params
    for el in dir(model):
        if not isinstance(getattr(model, el), nn.Module):
            continue
        params["children"].append(count_parameters(getattr(model, el), el, depth - 1))
    return params


def extract_frames(video_path: str, frame_sampling: str, n_frames: int, width: int, height: int) -> list[npt.NDArray[np.uint8]]:
    v_reader = VideoReader(video_path, width=width, height=height)
    v_len_frames = len(v_reader)
    min_frames = min(n_frames, v_len_frames)
    intervals = np.linspace(
        0, v_len_frames, min_frames + 1, dtype=np.int32)
    if frame_sampling == "uniform":
        frame_indices = [(intervals[i] + intervals[i+1]) //
                            2 for i in range(len(intervals) - 1)]
    frames = [frame for frame in v_reader.get_batch(
        frame_indices).asnumpy()]
    return frames

def extract_frames_ffmpeg(video_path: str, n_frames: int) -> list[npt.NDArray[np.uint8]]:
    probe = ffmpeg.probe(video_path)

    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'))
    total_frames = int(stream["nb_frames"])
    width_frame, height_frame = int(stream["width"]), int(stream["height"])
    interval = max(ceil(total_frames / n_frames), 1)
    
    process = (
        ffmpeg
        .input(video_path)
        .filter("framestep", interval)
        .output("pipe:", format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    
    frames = []
    while True:
        in_bytes = process.stdout.read(width_frame * height_frame * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape((height_frame, width_frame, 3))
        frames.append(frame)
            
    return frames[:n_frames]