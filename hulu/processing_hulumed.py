"""Processor class for HuluMed with 3D support."""

import copy
import importlib.util
import os
import os.path as osp
import warnings
from collections import defaultdict
from typing import Any, List, Union, Dict, Optional, Tuple, TypedDict

import cv2
import ffmpeg
import imageio
import json
import numpy as np
import torch
import transformers
from decord import VideoReader, cpu
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel is not installed. 3D medical imaging support will be limited. Install with: pip install nibabel")

try:
    from . import image_processing_hulumed
    from .image_processing_hulumed import (
        is_valid_image, is_valid_video,
    )
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "image_processing_hulumed",
        osp.join(osp.dirname(__file__), "image_processing_hulumed.py"),
    )
    image_processing_hulumed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_processing_hulumed)
    is_valid_image = getattr(image_processing_hulumed, "is_valid_image")
    is_valid_video = getattr(image_processing_hulumed, "is_valid_video")

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

Conversation = List[Dict[str, Any]]
SingleImage = Union[Image.Image, np.ndarray, torch.Tensor]
SingleVideo = Union[List[SingleImage], np.ndarray, torch.Tensor]
BatchedImage = List[Union[SingleImage, SingleVideo]]
BatchedNamedImage = List[Tuple[str, Union[SingleImage, SingleVideo]]]


def _custom_import(class_name: str):
    try:
        attribute_class = getattr(transformers, class_name)
    except AttributeError:
        attribute_class = getattr(image_processing_hulumed, class_name)
    return attribute_class


def is_named_image(image) -> bool:
    return isinstance(image, (list, tuple)) and \
        len(image) == 2 and \
        isinstance(image[0], str) and \
        image[0] in ["image", "video", "3d"] and \
        (is_valid_image(image[1]) or is_valid_video(image[1]))


def make_batched_images(images) -> Tuple[List[str], List[ImageInput]]:
    if isinstance(images, (list, tuple)) and all(is_named_image(image) for image in images):
        modals = [image[0] if image[0] != "3d" else "video" for image in images]
        data = [image[1] for image in images]
        return modals, data
    elif isinstance(images, (list, tuple)) and all(is_valid_image(image) or is_valid_video(image) for image in images):
        batch = []
        for image in images:
            if is_valid_video(image):
                batch.append(("video", image))
            elif is_valid_image(image):
                batch.append(("image", image))
            else:
                raise ValueError(f"Could not make batched images from {images}")
        return [x[0] for x in batch], [x[1] for x in batch]
    elif is_named_image(images):
        modal = images[0] if images[0] != "3d" else "video"
        return [modal], [images[1]]
    elif is_valid_video(images):
        return ["video"], [images]
    elif is_valid_image(images):
        return ["image"], [images]

    raise ValueError(f"Could not make batched images from {images}")


def frame_sample(duration, mode='uniform', num_frames=None, vid_fps=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ValueError(f'Unsupported frame sampling mode: {mode}')


def load_video_from_ids(video_path, s=None, e=None, fps=None, max_frames=128, temporal_factor=1):
    if s is not None and e is not None:
        s = s if s >= 0. else 0.
        e = e if e >= 0. else 0.
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))
        vid_fps = 3
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)
        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    f_start = 0 if s is None else max(int(s * vid_fps) - 1, 0)
    f_end = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    if fps is not None and duration / vid_fps < max_frames:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', vid_fps=vid_fps, fps=fps)]
    else:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=max_frames)]

    if os.path.isdir(video_path):
        frames = np.array([cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[frame_idx])), cv2.COLOR_BGR2RGB) for frame_idx in sampled_frame_indices])
    elif video_path.endswith('.gif'):
        frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices])
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    frames = frames.transpose(0, 3, 1, 2)
    timestamps = [x / vid_fps for x in sampled_frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    frames = [frame for frame in frames]

    return frames, timestamps


class ChatTemplateKwargs(TypedDict, total=False):
    chat_template: Optional[str]
    add_system_prompt: Optional[bool]
    add_generation_prompt: Optional[bool]


class HulumedProcessorKwargs(ProcessingKwargs, ChatTemplateKwargs, total=False):
    chat_template_kwargs: ChatTemplateKwargs = {
        **ChatTemplateKwargs.__annotations__,
    }

    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            
        },
        "chat_template_kwargs": {
            "chat_template": None,
            "add_system_prompt": False,
            "add_generation_prompt": False,
        },
    }


class HulumedProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "HulumedImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template", "image_merge_size", "video_merge_size", "fps", "max_frames"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template: str = None,
        image_merge_size: int = 1,
        video_merge_size: int = 2,
        fps: Optional[int] = 1,
        max_frames: Optional[int] = 128,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        if chat_template is None:
            chat_template = self.tokenizer.chat_template
        self.chat_template = chat_template

        self.image_merge_size = image_merge_size
        self.video_merge_size = video_merge_size
        self.fps = fps
        self.max_frames = max_frames

        self.generation_prompt = self._infer_generation_prompt()
        self.generation_prompt_ids = self.tokenizer.encode(self.generation_prompt, return_tensors="pt")
        self.generation_prompt_length = len(self.generation_prompt_ids[0])
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(_custom_import(n) if n is not None else None for n in class_name)
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = _custom_import(class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args

    def get_generation_prompt(self):
        return self.generation_prompt

    def get_generation_prompt_ids(self):
        return self.generation_prompt_ids

    def _infer_generation_prompt(self):
        pseudo_message = [{"role": "user", "content": ""}]
        instruction = self.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=True)
        conversation = self.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=False)
        return instruction.replace(conversation, "")

    def _get_downsampled_grid_sizes(self, image_inputs: Dict[str, Any]):
        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])
        return grid_sizes

    def _get_visual_seq_len(self, grid_size: torch.Tensor):
        num_tokens = int(grid_size.prod().item())
        return num_tokens

    def load_images(self, image_path: Union[str, List[str], Image.Image, List[Image.Image]]):
        if isinstance(image_path, str) and os.path.isfile(image_path):
            images = [Image.open(image_path).convert('RGB')]
        elif isinstance(image_path, str) and os.path.isdir(image_path):
            images = [Image.open(os.path.join(image_path, f)).convert('RGB') for f in sorted(os.listdir(image_path))]
        elif isinstance(image_path, list) and isinstance(image_path[0], str):
            images = [Image.open(f).convert('RGB') for f in image_path]
        elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
            images = [np.array(x) for x in image_path]
        elif isinstance(image_path, Image.Image):
            images = [np.array(image_path)]
        else:
            raise ValueError(f"Unsupported image path type: {type(image_path)}")
        return images

    def load_nii(
        self,
        nii_path: str,
        num_slices: Optional[int] = None,
        axis: int = 2,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        normalize: bool = True,
    ):
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")
        
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f"NIfTI file not found: {nii_path}")
        
        nii_img = nib.load(nii_path)
        volume = nii_img.get_fdata()
        
        if axis == 0:
            slices = [volume[i, :, :] for i in range(volume.shape[0])]
        elif axis == 1:
            slices = [volume[:, i, :] for i in range(volume.shape[1])]
        elif axis == 2:
            slices = [volume[:, :, i] for i in range(volume.shape[2])]
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")
        
        if num_slices is not None and num_slices < len(slices):
            indices = np.linspace(0, len(slices) - 1, num_slices, dtype=int)
            slices = [slices[i] for i in indices]
        
        processed_slices = []
        for slice_2d in slices:
            if window_center is not None and window_width is not None:
                lower = window_center - window_width / 2
                upper = window_center + window_width / 2
                slice_2d = np.clip(slice_2d, lower, upper)
            
            if normalize:
                slice_min = slice_2d.min()
                slice_max = slice_2d.max()
                if slice_max > slice_min:
                    slice_2d = (slice_2d - slice_min) / (slice_max - slice_min) * 255.0
                else:
                    slice_2d = np.zeros_like(slice_2d)
            
            slice_2d = slice_2d.astype(np.uint8)
            slice_rgb = np.stack([slice_2d] * 3, axis=0)
            
            processed_slices.append(slice_rgb)
        
        return processed_slices

    def load_video(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        fps: Optional[float] = None,
        max_frames: Optional[float] = None,
        size: Optional[int] = None,
        size_divisible: int = 1,
        precise_time: bool = False,
        verbose: bool = False,
        temporal_factor: int = 1
    ):
        fps = self.fps if fps is None else fps
        max_frames = self.max_frames if max_frames is None else max_frames

        if start_time is not None and end_time is not None and end_time - start_time < 1:
            return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
        if os.path.isdir(video_path):
            return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
        if video_path.endswith('.gif'):
            return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
        
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        w, h = int(video_stream['width']), int(video_stream['height'])

        kwargs, input_kwargs, output_kwargs = {}, {}, {}
        do_trim = start_time is not None or end_time is not None
        if start_time is not None:
            new_start_time = max(float(video_stream['start_time']), start_time)
            duration -= new_start_time - start_time
            start_time = new_start_time
        else:
            start_time = float(video_stream['start_time'])
        if end_time is not None:
            duration = min(duration, end_time - start_time)
        if do_trim:
            kwargs = {'ss': start_time, 't': duration}
        if precise_time:
            output_kwargs.update(kwargs)
        else:
            input_kwargs.update(kwargs)

        if size is not None:
            scale_factor = size / min(w, h)
            new_w, new_h = round(w * scale_factor), round(h * scale_factor)
        else:
            new_w, new_h = w, h
        new_w = new_w // size_divisible * size_divisible
        new_h = new_h // size_divisible * size_divisible

        stream = ffmpeg.input(video_path, **input_kwargs)
        if fps is not None:
            stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
        if new_w != w or new_h != h:
            stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
        stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
        out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

        frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

        if fps is not None:
            timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[:len(frames)]
        else:
            timestamps = np.linspace(start_time, start_time + duration, len(frames))

        if max_frames is not None and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = frames[indices]
            timestamps = timestamps[indices]

        if temporal_factor > 1:
            pad_length = temporal_factor - len(frames) % temporal_factor
            frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
            timestamps = np.concatenate([timestamps, timestamps[-1:].repeat(pad_length) + np.arange(1, pad_length + 1) / fps])

        frames = [frame for frame in frames]
        timestamps = [timestamp for timestamp in timestamps]

        return frames, timestamps

    def _load_multimodal_data(self, conversation: Conversation):
        multimodal_info = defaultdict(list)
        new_conversation = []
        for message in conversation:
            new_message = {"role": message["role"]}
            if not isinstance(message["content"], (list, tuple)):
                new_message["content"] = message["content"]
                new_conversation.append(new_message)
                continue

            new_contents = []
            for content in message["content"]:
                if not isinstance(content, dict):
                    new_contents.append(content)
                    continue
                assert "type" in content, "Content must have 'type' field."
                
                if content["type"] in ["image", "video", "3d"] and content["type"] in content and isinstance(content[content["type"]], dict):
                    load_args = content[content["type"]]
                    data_id = json.dumps({k: v for k, v in load_args.items() if k not in ["start_time", "end_time"]})
                    new_content = copy.deepcopy(content)
                    multimodal_info[data_id].append(new_content)
                    new_contents.append(new_content)
                else:
                    new_contents.append(content)

            new_message["content"] = new_contents
            new_conversation.append(new_message)

        for data_id, contents in multimodal_info.items():
            data_type = contents[0]["type"]
            
            if data_type == "image":
                image = self.load_images(contents[0][data_type]["image_path"])[0]
                for content in contents:
                    content["image"] = [image.copy()]

            elif data_type == "3d":
                load_args = contents[0]["3d"]
                nii_path = load_args["image_path"]
                num_slices = load_args.get("nii_num_slices", None)
                axis = load_args.get("nii_axis", 2)
                window_center = load_args.get("window_center", None)
                window_width = load_args.get("window_width", None)
                
                slices = self.load_nii(
                    nii_path=nii_path,
                    num_slices=num_slices,
                    axis=axis,
                    window_center=window_center,
                    window_width=window_width,
                )
                
                for content in contents:
                    content["type"] = "video"
                    content["video"] = slices
                    content["num_frames"] = len(slices)
                    content.pop("3d", None)

            elif data_type == "video":
                start_times = [content["video"].get("start_time", 0.) for content in contents]
                end_times = [content["video"].get("end_time", float("inf")) for content in contents]

                load_args = contents[0][data_type]
                start_time, end_time = min(start_times), max(end_times)
                if start_time > 0:
                    load_args["start_time"] = start_time
                if end_time < float("inf"):
                    load_args["end_time"] = end_time
                images, timestamps = self.load_video(**load_args)

                for content, start_time, end_time in zip(contents, start_times, end_times):
                    cur_images, cur_timestamps = [], []
                    for image, timestamp in zip(images, timestamps):
                        if start_time <= timestamp <= end_time:
                            cur_images.append(image.copy())
                            cur_timestamps.append(timestamp)

                    content[data_type] = cur_images
                    content["num_frames"] = len(cur_images)
                    content["timestamps"] = cur_timestamps

        return new_conversation

    def _gather_multimodal_data(self, conversation: Conversation):
        images = []
        for message in conversation:
            if not isinstance(message["content"], (list, tuple)):
                continue
            for content in message["content"]:
                if not isinstance(content, dict):
                    continue
                if content["type"] == "video":
                    video = content["video"]
                    assert is_valid_video(video), f"Invalid video data: {video}."
                    images.append(("video", video))
                elif content["type"] == "image":
                    image = content["image"]
                    images.append(("image", image))
        images = images if len(images) > 0 else None
        return images

    def _process_conversation_with_label(
        self,
        conversation: Conversation,
        image_inputs: Dict[str, Any],
        **kwargs,
    ):
        assert kwargs.pop("return_tensors", "pt") == "pt", "Only PyTorch tensors are supported when return_labels=True."
        assert "add_generation_prompt" not in kwargs, "'add_generation_prompt' argument is not supported when return_labels=True."

        output_kwargs = self._merge_kwargs(
            HulumedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        output_kwargs["chat_template_kwargs"].pop("add_generation_prompt")

        grid_sizes = self._get_downsampled_grid_sizes(image_inputs)
        text_inputs = {"input_ids": [], "labels": []}
        sample_types_list = []
        image_idx = 0

        for message_idx, message in enumerate(conversation):
            prompt = self.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=False,
                **output_kwargs["chat_template_kwargs"],
            )
            prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
            prompt = []
            for chunk_idx in range(len(prompt_chunks) - 1):
                prompt.append(prompt_chunks[chunk_idx])
                num_tokens = self._get_visual_seq_len(grid_sizes[image_idx])
                prompt.append(DEFAULT_IMAGE_TOKEN * num_tokens)
                image_idx += 1
            prompt.append(prompt_chunks[-1])
            prompt = "".join(prompt)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", **output_kwargs["text_kwargs"])[0]
            text_inputs["input_ids"].append(input_ids)

            targets = torch.full_like(input_ids, IGNORE_INDEX)
            sample_types = torch.full_like(input_ids, IGNORE_INDEX)
            if message["role"] == "assistant":
                targets[self.generation_prompt_length:-1] = input_ids[self.generation_prompt_length:-1].clone()
            elif message["role"] == "stream":
                diff = torch.diff((input_ids == self.image_token_id).float())
                image_end_indices = torch.nonzero(diff < 0)[:, 0]
                targets[image_end_indices + 1] = input_ids[image_end_indices + 1]
                sample_types = targets.clone()
                sample_types[torch.logical_and(sample_types > 0, sample_types != self.eos_token_id)] = 0
                targets[-2] = input_ids[-2]

            if message_idx > 0 and conversation[message_idx - 1]["role"] == "stream":
                targets[0] = input_ids[0]
                sample_types[0] = input_ids[0]

            text_inputs["labels"].append(targets)
            sample_types_list.append(sample_types)

        text_inputs = {k: torch.cat(v) for k, v in text_inputs.items()}
        sample_types = torch.cat(sample_types_list)
        types, counts = torch.unique(sample_types[sample_types > -1], return_counts=True)

        if len(types) > 0:
            target_num_samples = counts.amin()
            for type_id, type_count in zip(types, counts):
                if type_count > target_num_samples:
                    indices = torch.nonzero(sample_types == type_id)[:, 0]
                    random_selector = torch.randperm(indices.size(0))[:-target_num_samples]
                    text_inputs["labels"][indices[random_selector]] = IGNORE_INDEX

        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        return text_inputs

    def _process_conversation_without_label(
        self,
        conversation: Conversation,
        image_inputs: Dict[str, Any],
        **kwargs,
    ):
        output_kwargs = self._merge_kwargs(
            HulumedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        prompt = self.apply_chat_template(
            conversation,
            tokenize=False,
            **output_kwargs["chat_template_kwargs"],
        )
        return self.process_text(prompt, image_inputs, **output_kwargs["text_kwargs"])

    def _process_conversation(
        self,
        conversation: Conversation,
        images: Optional[Union[BatchedImage, BatchedNamedImage]] = None,
        return_labels: bool = False,
        **kwargs: Unpack[HulumedProcessorKwargs],
    ) -> BatchFeature:
        assert isinstance(conversation, list), "Conversation must be a list of messages."

        if images is None:
            conversation = self._load_multimodal_data(conversation)
            images = self._gather_multimodal_data(conversation)

        if not images:  
            images = None
        elif isinstance(images, (list, tuple)):
            images = [img for img in images if img and (not isinstance(img, (list, tuple)) or len(img) > 0)]
            if not images:
                images = None
        output_kwargs = self._merge_kwargs(
            HulumedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            if "merge_size" not in output_kwargs["images_kwargs"]:
                has_video_or_3d = any(
                    content.get("type") in ["video", "3d"] or "video" in content or "3d" in content
                    for message in conversation
                    if isinstance(message.get("content"), list)
                    for content in message["content"]
                    if isinstance(content, dict)
                )
                
                output_kwargs["images_kwargs"]["merge_size"] = 2 if has_video_or_3d else 1
            
            image_inputs = self.process_images(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if return_labels:
            text_inputs = self._process_conversation_with_label(conversation, image_inputs, **kwargs)
        else:
            text_inputs = self._process_conversation_without_label(conversation, image_inputs, **kwargs)

        return BatchFeature(data={**text_inputs, **image_inputs})

    def _process_plain(
        self,
        text: Union[TextInput, PreTokenizedInput] = None,
        images: Optional[Union[BatchedImage, BatchedNamedImage]] = None,
        return_labels: bool = False,
        **kwargs: Unpack[HulumedProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You must provide 'text' or 'conversation'.")
        if return_labels:
            raise ValueError("return_labels is not supported for plain text processing.")

        output_kwargs = self._merge_kwargs(
            HulumedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.process_images(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        text_inputs = self.process_text(text, image_inputs, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def process_images(self, images: Union[BatchedImage, BatchedNamedImage], **kwargs):
        modals, images = make_batched_images(images)
        
        if "merge_size" not in kwargs:
            kwargs["merge_size"] = [
                self.video_merge_size if modal == "video" else self.image_merge_size
                for modal in modals
            ]
        
        image_inputs = self.image_processor(images=images, **kwargs)
        image_inputs["modals"] = modals
        return image_inputs

    def process_text(
        self,
        text: TextInput,
        image_inputs: Dict[str, Any],
        **kwargs,
    ):
        grid_sizes = self._get_downsampled_grid_sizes(image_inputs)

        kwargs.pop("padding", None)
        kwargs.pop("padding_side", None)

        if len(grid_sizes) > 0:
            image_idx = 0
            while DEFAULT_IMAGE_TOKEN in text:
                num_tokens = self._get_visual_seq_len(grid_sizes[image_idx])
                text = text.replace(DEFAULT_IMAGE_TOKEN, "<placeholder>" * num_tokens, 1)
                image_idx += 1
            text = text.replace("<placeholder>", DEFAULT_IMAGE_TOKEN)
    
            assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = self.tokenizer(text, **kwargs)
        return text_inputs

    def __call__(
        self,
        text: Optional[TextInput] = None,
        conversation: Optional[Conversation] = None,
        images: Optional[Union[BatchedImage, BatchedNamedImage]] = None,
        return_labels: bool = False,
        **kwargs: Unpack[HulumedProcessorKwargs],
    ) -> BatchFeature:
        if conversation is not None:
            if text is not None:
                raise ValueError("You cannot provide both 'conversation' and 'text'.")
            return self._process_conversation(conversation, images, return_labels, **kwargs)
        return self._process_plain(text, images, return_labels, **kwargs)

    def batch_decode(self, *args, skip_special_tokens=True, use_think=False, **kwargs):
        outputs = self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
        
        if not use_think:
            outputs = [self._remove_think_tags(output) for output in outputs]
        
        return outputs

    def decode(self, *args, skip_special_tokens=True, use_think=False, **kwargs):
        output = self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
        
        if not use_think:
            output = self._remove_think_tags(output)
        
        return output
    
    def _remove_think_tags(self, text: str) -> str:
        import re
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        return cleaned

    def apply_chat_template(
        self,
        conversation: Conversation,
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        add_system_prompt: bool = False,
        add_generation_prompt: bool = False,
        image_token: Optional[str] = DEFAULT_IMAGE_TOKEN,
        **kwargs,
    ) -> str:
        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument."
                )
        return self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=tokenize,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
            image_token=image_token,
            **kwargs
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names)) + ["modals"]

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Dict]:
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "chat_template_kwargs": {},
            "common_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "chat_template_kwargs": {},
            "common_kwargs": {},
        }

        used_keys = set()

        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                if modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        
        output_kwargs.update(default_kwargs)

        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality in output_kwargs:
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed twice: "
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if kwarg_value != "__empty__":
                    output_kwargs[modality][modality_key] = kwarg_value
                    used_keys.add(modality_key)

        if any(key in default_kwargs for key in kwargs):
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            for key in kwargs:
                if key not in used_keys:
                    output_kwargs["common_kwargs"][key] = kwargs[key]

        for modality in output_kwargs:
            output_kwargs[modality].update(output_kwargs["common_kwargs"])
        
        return output_kwargs