import os
import glob
import shutil
import tarfile
import tempfile
import argparse
import torch
from safetensors.torch import load_file, save_file

def average_weights(tar_path, output_tar_path):
    if not tar_path.endswith('.tar.gz'):
        raise ValueError(f"Error: The input file '{tar_path}' must end with .tar.gz")
    if not output_tar_path.endswith('.tar.gz'):
        output_tar_path += '.tar.gz'

    print(f"Processing tarball: {tar_path}")
    
    # Use a temporary directory for BOTH extracting and building the new checkpoint
    with tempfile.TemporaryDirectory(dir=".") as temp_workspace:
        extract_dir = os.path.join(temp_workspace, "extracted")
        build_dir = os.path.join(temp_workspace, "averaged_model")
        os.makedirs(extract_dir)
        
        print(f"Extracting tarball to temporary space...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
            
        # Find checkpoints
        search_pattern = os.path.join(extract_dir, "**", "*_ep2*")
        checkpoint_dirs = glob.glob(search_pattern, recursive=True)
        
        valid_ckpts = []
        for ckpt in checkpoint_dirs:
            if os.path.isdir(ckpt):
                safetensors_path = os.path.join(ckpt, "unet", "diffusion_pytorch_model.safetensors")
                if os.path.exists(safetensors_path):
                    valid_ckpts.append(ckpt)
        
        valid_ckpts.sort()
        if not valid_ckpts:
            raise ValueError("No valid checkpoints matching '*_ep2*' found.")
            
        print(f"\nFound {len(valid_ckpts)} checkpoints. Averaging...")
        
        # 1. Copy the structure to our build directory
        base_ckpt = valid_ckpts[0]
        shutil.copytree(base_ckpt, build_dir, dirs_exist_ok=True)
        
        # 2. Load & Average UNet weights
        state_dicts = [load_file(os.path.join(c, "unet", "diffusion_pytorch_model.safetensors")) for c in valid_ckpts]
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            avg_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
            
        # 3. Save the averaged weights
        output_safetensors = os.path.join(build_dir, "unet", "diffusion_pytorch_model.safetensors")
        save_file(avg_state_dict, output_safetensors)
        
        # 4. Compress the build directory directly to the requested output path
        print(f"\nCompressing averaged model to {output_tar_path}...")
        
        # Get the base name for the root folder inside the tarball (e.g., "main_3d_l1_averaged")
        arc_root_name = os.path.basename(output_tar_path).replace('.tar.gz', '')
        
        with tarfile.open(output_tar_path, "w:gz") as tar:
            tar.add(build_dir, arcname=arc_root_name)
            
    print("Done! Temporary workspace cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average weights from a tarball and output a new compressed tarball.")
    parser.add_argument("tarball", type=str, help="Path to the input .tar.gz file.")
    parser.add_argument("--out", type=str, required=True, help="Path for the output .tar.gz file (e.g., /staging/groups/cs_geodes/model_averaged.tar.gz)")
    
    args = parser.parse_args()
    average_weights(args.tarball, args.out)
    
    