from denoise import denoiser
from args import get_args

import tifffile
import numpy as np

def run(image_path, result_folder_path, min_patches_per_class, max_patches_per_class, iteration_counter, patch_size, termination_number, analyze, clustering_factor):

    image = tifffile.imread(image_path).astype(np.float32)
    file_name = image_path.split("/")[-1].split("\\")[-1].split(".")[0]

    if(len(image.shape)==2):
        image = image[np.newaxis,...]
        denoised_image = denoiser(image, 
                                  min_patches_per_class, 
                                  max_patches_per_class, 
                                  iteration_counter, 
                                  patch_size, 
                                  termination_number, 
                                  analyze, 
                                  clustering_factor)
    elif(image.shape[0]==1):
        denoised_image = denoiser(image, 
                                  min_patches_per_class, 
                                  max_patches_per_class, 
                                  iteration_counter, 
                                  patch_size, 
                                  termination_number, 
                                  analyze, 
                                  clustering_factor)
    elif(len(image.shape)==3):
        denoised_image = np.zeros((image.shape))
        for i in range(image.shape[0]):
            denoised_image[i,:,:] = denoiser(image[i:i+1,:,:], 
                                             min_patches_per_class, 
                                             max_patches_per_class, 
                                             iteration_counter, 
                                             patch_size, 
                                             termination_number, 
                                             analyze, 
                                             clustering_factor)[0]

    tifffile.imsave(result_folder_path + "/denoised_" + file_name + ".tif", denoised_image)

if __name__ == "__main__":

    args = get_args()

    run(args.image_path, 
        args.result_folder_path,
        args.min_patches_per_class, 
        args.max_patches_per_class, 
        args.iteration_counter,
        args.patch_size,
        args.termination_number,
        args.analyze,
        args.clustering_factor)
