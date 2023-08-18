import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Image denoiser')

    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--result_folder_path", type=str, default="")
    parser.add_argument("--min_patches_per_class", type=int, default=5)
    parser.add_argument("--max_patches_per_class", type=int, default=100)
    parser.add_argument("--iteration_counter", type=int, default=15)
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--termination_number", type=int, default=3)
    parser.add_argument("--analyze", type=bool, default=False)
    parser.add_argument("--clustering_factor", type=float, default=2.7)
                                
    return parser.parse_args()