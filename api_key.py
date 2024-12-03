import sys
import subprocess
import os
import urllib.request

def install_dependencies():
    """
    Robust dependency installation with flexible version handling
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    dependencies = [
        "numpy",              # Latest stable version
        "torch",              # Latest compatible version
        "torchvision",        # Automatically matched with torch
        "basicsr",            # Latest version
        "realesrgan",         # Latest version
        "Pillow"              # Latest image processing library
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Successfully installed {dep}")
        except subprocess.CalledProcessError:
            print(f"Warning: Could not install {dep}. Please install manually.")

def download_model_weights(url, save_path):
    """
    Download model weights with comprehensive error handling
    
    Args:
        url (str): Direct download URL for model weights
        save_path (str): Local path to save the weights
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading model weights from {url}...")
        
        # Improved download with progress tracking
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            progress = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\rDownload progress: {progress:.1f}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, save_path, progress_hook)
        print("\nModel weights downloaded successfully.")
    except Exception as e:
        print(f"Critical error downloading model weights: {e}")
        raise

def debug_model_loading():
    """
    Diagnostic function to verify model weight loading
    """
    import torch
    import os
    
    weights_path = 'weights/RealESRGAN_x4plus.pth'
    
    # Verify file exists and is not empty
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at {weights_path}")
    
    if os.path.getsize(weights_path) == 0:
        raise ValueError(f"Model weights file at {weights_path} is empty")
    
    try:
        # Attempt to load weights manually
        state_dict = torch.load(weights_path)
        print("Model weights loaded successfully")
        return state_dict
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise

def upscale_image(input_path, output_path='upscaled_image.jpg', scale_factor=4):
    """
    Upscale an image using RealESRGAN with comprehensive error handling
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save upscaled image
        scale_factor (int): Upscaling factor
    """
    from realesrgan import RealESRGANer
    from PIL import Image
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import numpy as np
    import torch
    # Print diagnostic information
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    try:
        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # Define weights path
        weights_dir = 'weights'
        weights_path = os.path.join(weights_dir, 'RealESRGAN_x4plus.pth')
        
        # Download weights if not present
        if not os.path.exists(weights_path):
            download_model_weights(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                weights_path
            )
        
        # Debug and verify model weights
        debug_model_loading()

        # Initialize upscaling model with explicit error handling
       
        model = RRDBNet(
            num_in_ch=3,      # RGB channels
            num_out_ch=3,     # Output RGB channels
            num_feat=64,      # Feature dimensions
            num_block=23,     # Network depth
            num_grow_ch=32,   # Growth channels
            scale=scale_factor
        )

        # Advanced upsampler initialization
        upsampler = RealESRGANer(
            scale=scale_factor,
            model_path=weights_path,
            model=model,
            tile=0,            # Disable tiling to avoid potential issues
            tile_pad=10,       # Tile padding
            pre_pad=0,
            half=torch.cuda.is_available()  # Use half precision if CUDA available
        )

        # Image processing with comprehensive error checking
        input_image = Image.open(input_path).convert('RGB')
        np_image = np.array(input_image)

        # Explicit enhancement with error tracking
        try:
            output, _ = upsampler.enhance(np_image, outscale=scale_factor)
        except Exception as enhance_error:
            print(f"Enhancement Error Details: {enhance_error}")
            import traceback
            traceback.print_exc()
            raise

        # Save processed image
        output_image = Image.fromarray(output)
        output_image.save(output_path)

        print(f"Image successfully upscaled to {output_path}")

    except Exception as general_error:
        print(f"Comprehensive Upscaling Error: {general_error}")
        import traceback
        traceback.print_exc()
        raise
def main():
    # Manage dependencies
    install_dependencies()
    
    # Specify your low-resolution image path
    input_image_path = 'low_rosulation.jpg'
    
    # Perform image upscaling
    upscale_image(input_image_path)

if __name__ == "__main__":
    main()