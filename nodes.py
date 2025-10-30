#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,gc
import math
import torch
import folder_paths
import comfy.utils

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from huggingface_hub import snapshot_download
from .src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from .src.models.TCDecoder import build_tcdecoder
from .src.models.utils import clean_vram, Buffer_LQ4x_Proj
from .src.models import wan_video_dit

def get_device_list():
    devs = ["auto"]
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs

device_choices = get_device_list()

def log(message:str, message_type:str='normal'):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(folder_paths.models_dir, "FlashVSR")
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]

    return cropped_tensor.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH).to('cpu').to(dtype) * 2.0 - 1.0
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(mode, device, dtype, alt_vae="none"):
    model_downlod()
    model_path = os.path.join(folder_paths.models_dir, "FlashVSR")
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"')
    if alt_vae != "none":
        vae_path = folder_paths.get_full_path_or_raise("vae", alt_vae)
        if not os.path.exists(vae_path):
            raise RuntimeError(f'"{alt_vae}" does not exist!')
    else:
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist!\nPlease save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()
        
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    pipe.offload_model()

    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing"):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            val = next(self.iterable)
            if self.pbar:
                self.pbar.update(1)
            return val
        except StopIteration:
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

def _blend_overlapping_frames(frame1, frame2, alpha=0.5):
    """
    Blend two overlapping frames using weighted averaging.
    
    Args:
        frame1: First frame tensor
        frame2: Second frame tensor
        alpha: Blending factor (0.5 = equal weight)
    
    Returns:
        Blended frame tensor
    """
    return frame1 * alpha + frame2 * (1.0 - alpha)

def _process_frames_in_batches(pipe, _frames, original_frames, scale, color_fix, tiled_vae, 
                                tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, 
                                kv_ratio, local_range, seed, force_offload, batch_size, 
                                frame_overlap, original_frame_count):
    """
    Process frames in batches with overlapping frames for temporal consistency.
    
    Args:
        batch_size: Number of frames per batch
        frame_overlap: Number of frames to overlap between consecutive batches
    
    Returns:
        Processed frames tensor with original frame count maintained
    """
    _device = pipe.device
    dtype = pipe.torch_dtype
    
    # Minimum frames required for processing (pipeline needs at least 21 frames)
    MIN_BATCH_FRAMES = 21
    
    total_frames = _frames.shape[0]
    log(f"[FlashVSR] Processing {total_frames} frames in batches of {batch_size} with {frame_overlap} frame overlap", message_type='info')
    
    # Calculate batch boundaries
    batch_outputs = []
    batch_start = 0
    batch_idx = 0
    
    while batch_start < total_frames:
        # Determine batch end
        batch_end = min(batch_start + batch_size, total_frames)
        actual_batch_size = batch_end - batch_start
        
        # Check if this would be the last batch and if it's too small
        # If so, extend it to include more frames from the previous batch region
        remaining_frames = total_frames - batch_end
        if remaining_frames > 0 and remaining_frames < MIN_BATCH_FRAMES:
            # Extend current batch to include the remaining frames
            batch_end = total_frames
            actual_batch_size = batch_end - batch_start
            log(f"[FlashVSR] Extended batch to include remaining {remaining_frames} frames (would be too small)", message_type='info')
        elif actual_batch_size < MIN_BATCH_FRAMES and batch_start > 0:
            # Current batch is too small, extend it backwards
            batch_start = max(0, batch_end - MIN_BATCH_FRAMES)
            actual_batch_size = batch_end - batch_start
            log(f"[FlashVSR] Extended batch backwards to meet minimum frame requirement", message_type='info')
        
        log(f"[FlashVSR] Processing batch {batch_idx + 1}: frames {batch_start} to {batch_end} (size: {actual_batch_size})", message_type='info')
        
        # Extract batch frames
        batch_frames = _frames[batch_start:batch_end, :, :, :]
        
        # Process this batch
        if tiled_dit:
            N, H, W, C = batch_frames.shape
            num_aligned_frames = largest_8n1_leq(N + 4) - 4
            
            final_output_canvas = torch.zeros(
                (num_aligned_frames, H * scale, W * scale, C), 
                dtype=dtype, 
                device="cpu"
            )
            weight_sum_canvas = torch.zeros_like(final_output_canvas)
            tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
            
            for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc=f"Processing Batch {batch_idx + 1} Tiles")):
                input_tile = batch_frames[:, y1:y2, x1:x2, :]
                
                LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                if not isinstance(pipe, FlashVSRTinyLongPipeline):
                    LQ_tile = LQ_tile.to(_device)
                    
                output_tile_gpu = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                    LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                    topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                    color_fix=color_fix, unload_dit=unload_dit, force_offload=force_offload
                )
                
                processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
                
                mask_nchw = create_feather_mask(
                    (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                    tile_overlap * scale
                ).to("cpu")
                mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
                out_x1, out_y1 = x1 * scale, y1 * scale
                
                tile_H_scaled = processed_tile_cpu.shape[1]
                tile_W_scaled = processed_tile_cpu.shape[2]
                out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
                final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
                
                del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
                clean_vram()
                
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            batch_output = final_output_canvas / weight_sum_canvas
        else:
            LQ, th, tw, F = prepare_input_tensor(batch_frames, _device, scale=scale, dtype=dtype)
            if not isinstance(pipe, FlashVSRTinyLongPipeline):
                LQ = LQ.to(_device)
            
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                progress_bar_cmd=cqdm, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit, force_offload=force_offload
            )
            
            batch_output = tensor2video(video).to('cpu')
            
            del video, LQ
            clean_vram()
        
        # Log the batch output size for debugging
        log(f"[FlashVSR] Batch {batch_idx + 1} produced {batch_output.shape[0]} frames from {actual_batch_size} input frames", message_type='info')
        
        batch_outputs.append(batch_output)
        
        # Move to next batch, accounting for overlap
        if batch_end < total_frames:
            batch_start = batch_end - frame_overlap
        else:
            batch_start = batch_end
        
        batch_idx += 1
    
    # Merge batches with blending for overlapping regions
    log(f"[FlashVSR] Merging {len(batch_outputs)} batches with overlap blending", message_type='info')
    
    if len(batch_outputs) == 1:
        final_output = batch_outputs[0]
    else:
        # Start with first batch - convert to list of individual frames using torch.unbind for efficiency
        merged_frames = list(torch.unbind(batch_outputs[0], dim=0))
        
        for i in range(1, len(batch_outputs)):
            current_batch = batch_outputs[i]
            
            # Blend overlapping frames
            if frame_overlap > 0 and len(merged_frames) >= frame_overlap:
                # Get the last frame_overlap frames from merged output (already as separate tensors)
                prev_batch_last_frames = merged_frames[-frame_overlap:]
                
                # Get the first frame_overlap frames from current batch
                curr_batch_first_frames = list(torch.unbind(current_batch[:frame_overlap, :, :, :], dim=0))
                
                # Blend overlapping frames with gradual transition
                blended_frames = []
                for j in range(frame_overlap):
                    # Gradually transition from previous to current batch
                    alpha = (j + 1) / (frame_overlap + 1)
                    blended = _blend_overlapping_frames(
                        prev_batch_last_frames[j], 
                        curr_batch_first_frames[j], 
                        alpha=alpha
                    )
                    blended_frames.append(blended)
                
                # Replace last frame_overlap frames in merged output with blended ones
                merged_frames = merged_frames[:-frame_overlap] + blended_frames
                
                # Add remaining frames from current batch (skip overlapped ones) using torch.unbind
                merged_frames.extend(list(torch.unbind(current_batch[frame_overlap:, :, :, :], dim=0)))
            else:
                # No overlap, just append using torch.unbind for efficiency
                merged_frames.extend(list(torch.unbind(current_batch, dim=0)))
        
        # Stack all frames
        final_output = torch.stack(merged_frames, dim=0)
    
    # Integrity check: ensure output frame count matches
    output_frame_count = final_output.shape[0]
    log(f"[FlashVSR] Frame count after batch processing: {output_frame_count}", message_type='info')
    
    # Handle case where we have fewer frames than original due to alignment
    if output_frame_count < original_frame_count:
        log(f"[FlashVSR] Padding output from {output_frame_count} to {original_frame_count} frames", message_type='info')
        # Pad with the last frame expanded (memory efficient view)
        frames_needed = original_frame_count - output_frame_count
        last_frame = final_output[-1:, :, :, :]
        padding = last_frame.expand(frames_needed, -1, -1, -1)
        final_output = torch.cat([final_output, padding], dim=0)
    else:
        # Trim to original count if we have more frames
        final_output = final_output[:original_frame_count, :, :, :]
    
    # Final integrity check
    if final_output.shape[0] != original_frame_count:
        log(f"[FlashVSR] WARNING: Output frame count ({final_output.shape[0]}) != input frame count ({original_frame_count})", message_type='warning')
    else:
        log(f"[FlashVSR] Integrity check passed: {final_output.shape[0]} frames", message_type='finish')
    
    log("[FlashVSR] Batch processing complete.", message_type='finish')
    return final_output

def flashvsr(pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, batch_size=1, frame_overlap=2):
    """
    Process video frames with optional batch processing and temporal overlap.
    
    Args:
        batch_size: Number of frames to process in each batch (default: 1, means process all at once)
        frame_overlap: Number of frames to overlap between batches for temporal consistency (default: 2)
    """
    _frames = frames
    _device = pipe.device
    dtype = pipe.torch_dtype
    
    original_frame_count = frames.shape[0]
    log(f"[FlashVSR] Input frame count: {original_frame_count}", message_type='info')

    if frames.shape[0] < 21:
        add = 21 - frames.shape[0]
        last_frame = frames[-1:, :, :, :]
        padding_frames = last_frame.repeat(add, 1, 1, 1)
        _frames = torch.cat([frames, padding_frames], dim=0)
    
    # Check if batch processing is requested (batch_size > 1 means process in batches)
    if batch_size > 1 and _frames.shape[0] > batch_size:
        log(f"[FlashVSR] Batch processing enabled: batch_size={batch_size}, overlap={frame_overlap}", message_type='info')
        return _process_frames_in_batches(
            pipe, _frames, frames, scale, color_fix, tiled_vae, tiled_dit, 
            tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, 
            local_range, seed, force_offload, batch_size, frame_overlap,
            original_frame_count
        )
        
    if tiled_dit:
        N, H, W, C = _frames.shape
        num_aligned_frames = largest_8n1_leq(N + 4) - 4
        
        final_output_canvas = torch.zeros(
            (num_aligned_frames, H * scale, W * scale, C), 
            dtype=dtype, 
            device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        latent_tiles_cpu = []
        
        for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc="Processing Tiles")):
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})", message_type='info')
            input_tile = _frames[:, y1:y2, x1:x2, :]
            
            LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
            if not isinstance(pipe, FlashVSRTinyLongPipeline):
                LQ_tile = LQ_tile.to(_device)
                
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit, force_offload=force_offload
            )
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log("[FlashVSR] Preparing frames...")
        LQ, th, tw, F = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        if not isinstance(pipe, FlashVSRTinyLongPipeline):
            LQ = LQ.to(_device)
        log(f"[FlashVSR] Processing {frames.shape[0]} frames...", message_type='info')
        
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            progress_bar_cmd=cqdm, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, unload_dit=unload_dit, force_offload=force_offload
        )
        
        final_output = tensor2video(video).to('cpu')
        
        del video, LQ
        clean_vram()
        
    log("[FlashVSR] Done.", message_type='info')
    if frames.shape[0] == 1:
        final_output = final_output.to(_device)
        stacked_image_tensor = torch.median(final_output, dim=0).values.unsqueeze(0).to('cpu')
        del final_output
        clean_vram()
        return stacked_image_tensor
    
    return final_output[:frames.shape[0], :, :, :]

class FlashVSRNodeInitPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Using "tiny-long" mode can significantly reduce VRAM used with long video input.'
                }),
                "alt_vae": (["none"] + folder_paths.get_filename_list("vae"), {
                    "default": "none",
                    "tooltip": 'Replaces the built-in VAE, only available in "full" mode.'
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload all weights to CPU after running a workflow to free up VRAM."
                }),
                "precision": (["fp16", "bf16"], {
                    "default": "bf16",
                    "tooltip": "Data and inference precision."
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Device to load the weights, default: auto (CUDA if available, else CPU)"
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention"], {
                    "default": "sparse_sage_attention",
                    "tooltip": '"sparse_sage_attention" is available for sm_75 to sm_120\n"block_sparse_attention" is available for sm_80 to sm_100'
                }),
            }
        }
    
    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Download the entire "FlashVSR" folder with all the files inside it from "https://huggingface.co/JunhaoZhuang/FlashVSR" and put it in the "ComfyUI/models"'
    
    def main(self, mode, alt_vae, force_offload, precision, device, attention_mode):
        _device = device
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else device
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        if attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
            
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = dtype_map[precision]
        except:
            dtype = torch.bfloat16
            
        pipe = init_pipeline(mode, _device, dtype, alt_vae=alt_vae)
        return((pipe, force_offload),)

class FlashVSRNodeAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE", {
                    "tooltip": "FlashVSR pipeline"
                }),
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use wavelet transform to correct output video color."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable tiling: faster decode but higher VRAM usage.\nSet to True for lower memory consumption at the cost of speed."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Significantly reduces VRAM usage at the cost of speed."
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                }),
                "tile_overlap": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload DiT before decoding to reduce VRAM peak at the cost of speed."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.5 or 2.0\n1.5 → faster; 2.0 → more stable"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.0 to 3.0\n1.0 → less vram; 3.0 → high quality"
                }),
                "local_range": ("INT", {
                    "default": 11,
                    "min": 9,
                    "max": 11,
                    "step": 2,
                    "tooltip": "Recommended: 9 or 11\nlocal_range=9 → sharper details; 11 → more stable results"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of frames to process per batch. Default 1 disables batch processing (processes all frames together). Set higher (e.g., 30-50) to enable batch processing for faster inference on long videos."
                }),
                "frame_overlap": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Number of frames to overlap between batches for temporal consistency. Recommended: 2-4 frames. Set to 0 to disable overlap. Only used when batch_size > 1."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    #DESCRIPTION = ""
    
    def main(self, pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, batch_size, frame_overlap):
        _pipe, force_offload = pipe
        output = flashvsr(_pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, batch_size, frame_overlap)
        return(output,)

class FlashVSRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Using "tiny-long" mode can significantly reduce VRAM used with long video input.'
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable tiling: faster decode but higher VRAM usage.\nSet to True for lower memory consumption at the cost of speed."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Significantly reduces VRAM usage at the cost of speed."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload DiT before decoding to reduce VRAM peak at the cost of speed."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of frames to process per batch. Default 1 disables batch processing (processes all frames together). Set higher (e.g., 30-50) to enable batch processing for faster inference on long videos."
                }),
                "frame_overlap": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Number of frames to overlap between batches for temporal consistency. Recommended: 2-4 frames. Set to 0 to disable overlap. Only used when batch_size > 1."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Download the entire "FlashVSR" folder with all the files inside it from "https://huggingface.co/JunhaoZhuang/FlashVSR" and put it in the "ComfyUI/models"'
    
    def main(self, frames, mode, scale, tiled_vae, tiled_dit, unload_dit, seed, batch_size, frame_overlap):
        wan_video_dit.USE_BLOCK_ATTN = False
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "auto"
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        pipe = init_pipeline(mode, _device, torch.float16)
        output = flashvsr(pipe, frames, scale, True, tiled_vae, tiled_dit, 256, 24, unload_dit, 2.0, 3.0, 11, seed, True, batch_size, frame_overlap)
        return(output,)

NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
    "FlashVSRNodeAdv": FlashVSRNodeAdv,
    "FlashVSRInitPipe": FlashVSRNodeInitPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRNode": "FlashVSR Ultra-Fast",
    "FlashVSRNodeAdv": "FlashVSR Ultra-Fast (Advanced)",
    "FlashVSRInitPipe": "FlashVSR Init Pipeline",
}