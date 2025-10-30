# ComfyUI-FlashVSR_Ultra_Fast
Running FlashVSR on lower VRAM without any artifacts.   
**[[📃中文版本](./README_zh.md)]**

## Changelog
#### 2025-10-30
- Added batch processing with configurable `batch_size` and `frame_overlap` parameters.
- Implemented temporal consistency with frame overlap blending for smoother transitions.
- Added integrity checks to ensure output frame count matches input.
- Improved performance with optimized tensor operations.

#### 2025-10-24
- Added long video pipeline that significantly reduces VRAM usage when upscaling long videos.

#### 2025-10-21
- Initial this project, introducing features such as `tile_dit` to significantly reducing VRAM usage.  

#### 2025-10-22
- Replaced `Block-Sparse-Attention` with `Sparse_Sage`, removing the need to compile any custom kernels.  
- Added support for running on RTX 50 series GPUs.

## Preview
![](./img/preview.jpg)

## Usage
- **mode:**  
`tiny` -> faster (default); `full` -> higher quality  
- **scale:**  
`4` is always better, unless you are low on VRAM then use `2`    
- **color_fix:**  
Use wavelet transform to correct the color of output video.  
- **tiled_vae:**  
Set to True for lower VRAM consumption during decoding at the cost of speed.  
- **tiled_dit:**  
Significantly reduces VRAM usage at the cost of speed.
- **tile\_size, tile\_overlap**:  
How to split the input video.  
- **unload_dit:**  
Unload DiT before decoding to reduce VRAM peak at the cost of speed.
- **batch_size:**  
Number of frames to process per batch. Default `1` disables batch processing (processes all frames together). Set higher (e.g., `30-50`) to enable batch processing for faster inference on long videos. Only applies when processing more frames than the batch size.
- **frame_overlap:**  
Number of frames to overlap between batches for temporal consistency. Recommended: `2-4` frames. Set to `0` to disable overlap. This helps maintain smooth transitions between batches by blending overlapping frames. Only used when `batch_size > 1`.  

## Installation

#### nodes: 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```

#### models:

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in the `ComfyUI/models`

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
