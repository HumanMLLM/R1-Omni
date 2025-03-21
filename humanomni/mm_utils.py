import ast
import os
import math
import base64
import traceback
from io import BytesIO
import io
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu, AudioReader
from transformers import StoppingCriteria
from .constants import NUM_FRAMES, NUM_FRAMES_PER_SECOND, MODAL_INDEX_MAP, DEFAULT_IMAGE_TOKEN

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit
def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image
def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches
def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"] if "shortest_edge" in processor.size else processor.size["height"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
   # print("@@@@@@@", image.size)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    
    image_padded = resize_and_pad_image(image, best_resolution)
  #  print("@@@@@", processor.size)
    patches = divide_to_patches(image_padded, processor.size["height"])
    print("image.size:", image.size, "possible_resolutions:", possible_resolutions, "best_resolution:", best_resolution, len(patches))
    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"] if "shortest_edge" in processor.size else processor.size["height"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))

    image_patches = [image_original_resize] + patches

    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def read_video_patch(patch_info, data_folder="/mnt/data/yixing.pyx/checkpoints/Oryx-SFT-DATA"):
   # import ipdb;ipdb.set_trace()
    is_image = False
    if 'img_path' in patch_info.keys():
        image = Image.open(patch_info['img_path']).convert('RGB')
        is_image = True
    else:
        image_file_name = os.path.join(data_folder, patch_info['patch'])
        start_bytes = int(patch_info['start_num'])
        file_size = patch_info['size'] # list of int
        if len(file_size) == 1:
            is_image = True
        else:
            is_image = False
        total_file_size = 0
        images_all = []
        with open(image_file_name, 'rb') as f:
            for idx in range(len(file_size)):
                f.seek(start_bytes + total_file_size)
                if 'image_encoding' in patch_info.keys() and patch_info['image_encoding'] == 'base64':
                    image = Image.open(io.BytesIO(base64.b64decode(f.read(int(file_size[idx])).decode()))).convert("RGB")
                else:
                    if 'sharegpt4o' in image_file_name or 'ShareGPT4Video/new_patch' in image_file_name or 'cinepile' in image_file_name or 'nextqa' in image_file_name or 'perceptiontest' in image_file_name:
                        byte_str = io.BytesIO(f.read(int(file_size[idx])))
                        array = np.frombuffer(byte_str.getvalue(), dtype=np.uint8)
                        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
                        image = Image.fromarray(image)
                    else:
                        image = Image.open(io.BytesIO(f.read(int(file_size[idx])))).convert("RGB")
                images_all.append(image)
                total_file_size += int(file_size[idx])
          #  import ipdb;ipdb.set_trace()
    return images_all, is_image

def resize_with_limit(image, max_size=512):
    """辅助函数：限制图像最长边"""
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def process_image(image_path, processor, aspect_ratio='pad'):
    image = Image.open(image_path).convert('RGB')

    images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images=images, videos=None, return_tensors='pt')['pixel_values']
    return images

def process_image_qwen(image_path, processor, aspect_ratio='pad'):
    image = Image.open(image_path).convert('RGB')
    
    # 使用resize_with_limit函数限制图像大小
    image = resize_with_limit(image)
    
    images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor(images=images, return_tensors='pt')
    return images

def process_image_npary(images, processor, aspect_ratio='pad'):
    if images is None:
        return None
    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images

def frame_sample(duration, mode='uniform', num_frames=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end   = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')



def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=NUM_FRAMES):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):                
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        if num_frames > 10000:
            num_frames = num_frames_of_video
        # 2. Determine frame range & Calculate frame indices
        f_start = 0                       if s is None else max(int(s * fps) - 1, 0)
        f_end   = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices 
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]


        if os.path.isdir(video_path): 
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
        elif video_path.endswith('.gif'):
            video_data = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
        else:
            video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]
    
    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")
    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))
    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        images = [f for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return video



def process_video_qwen(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=NUM_FRAMES):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):                
            frame_files = sorted(os.listdir(video_path))
            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)
            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        if num_frames > 10000:
            num_frames = num_frames_of_video
        # 2. Determine frame range & Calculate frame indices
        f_start = 0 if s is None else max(int(s * fps) - 1, 0)
        f_end = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices 
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]
        
        # 加载帧并应用大小限制
        if os.path.isdir(video_path): 
            video_data = [resize_with_limit(Image.open(os.path.join(video_path, frame_files[f_idx]))) for f_idx in sampled_frame_indices]
        elif video_path.endswith('.gif'):
            video_data = [resize_with_limit(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
        else:
            video_data = [resize_with_limit(Image.fromarray(frame)) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]
    
    elif isinstance(video_path, np.ndarray):
        video_data = [resize_with_limit(Image.fromarray(f)) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [resize_with_limit(Image.fromarray(f)) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [resize_with_limit(Image.open(f)) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = [resize_with_limit(f) for f in video_path]
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
        video = processor(images=None, videos=images, return_tensors='pt')
    else:
        images = [f for f in video_data]
        video = processor(images=None, videos=images, return_tensors='pt')
    return video



def process_audio(audio_path, processor=None, sample_rate=16000, duration=10, s=None, e=None, return_empty=False):
    if return_empty:
        num_samples = int(duration * sample_rate)
        audio_data = torch.zeros(num_samples, dtype=torch.float32)
        if processor is not None:
            audio_data = processor(audio_data, sampling_rate=sample_rate, return_tensors='pt')['input_features']
            if torch.isnan(audio_data).any():
                audio_data = torch.nan_to_num(audio_data, nan=-1.5)
            return audio_data, processor.sampling_rate
        return audio_data, sample_rate

    try:
        audio_reader = AudioReader(audio_path, ctx=cpu(0), sample_rate=sample_rate)
        audio_data = torch.from_numpy(audio_reader._array)
        audio_sample_rate = audio_reader.sample_rate
        
        if torch.isnan(audio_data).any():
            audio_data = torch.nan_to_num(audio_data, nan=-1.5)
        
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1
            
            start_idx = int(s * audio_sample_rate)
            end_idx = int(e * audio_sample_rate)
            start_idx = max(0, start_idx)
            end_idx = min(len(audio_data), end_idx)
            audio_data = audio_data[start_idx:end_idx]
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(dim=0)
            
    except Exception as e:
        num_samples = int(duration * sample_rate)
        audio_data = torch.zeros(num_samples, dtype=torch.float32)
        audio_sample_rate = sample_rate

    if processor is not None:
        audio_data = processor(audio_data, sampling_rate=audio_sample_rate, return_tensors='pt')['input_features']
        if torch.isnan(audio_data).any():
            audio_data = torch.nan_to_num(audio_data, nan=-1.5)
        audio_sample_rate = processor.sampling_rate

    return audio_data, audio_sample_rate


def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    """Tokenize text and multimodal tag to input_ids.
    Args:
        prompt (str): Text prompt (w/ multimodal tag), e.g., '<video>\nDescribe the video.'
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        multimodal_token (int): Token index corresponding to the multimodal tag.
    """
    # multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token is None or multimodal_token=="<text>":
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [prompt]
        separators = []
        # Split prompt by each token type
        for token_type, token_index in MODAL_INDEX_MAP.items():
            lower_token = token_type
            if lower_token in prompt:
                split_chunks = []
                for chunk in prompt_chunks:
                    if isinstance(chunk, str):
                        parts = chunk.split(lower_token)
                        split_chunks.extend([part for sublist in zip(parts, [token_index] * len(parts)) for part in sublist][:-1])
                    else:
                        split_chunks.append(chunk)
                    # split_chunks.append(parts[-1])
                prompt_chunks = split_chunks
                # Log the token index for insertion
                # sep_positions = [token_index for _ in range(len(parts)-1)]
                # separators.extend(sep_positions)
        encoded_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids if isinstance(chunk, str) else [chunk] for chunk in prompt_chunks]
        # Insert tokens into encoded chunks
        input_ids = []
        for chunk in encoded_chunks:
            input_ids.extend(chunk)
    
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        else:
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
    
    return input_ids



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
