// Author: Srirajani Sriram

#include <iostream>
#include <opencv2/opencv.hpp>
#include <SimpleITK.h>

#include <../headers/Segmentation.h>

int main() {
    // Define file paths
    const std::string modelPath = "Add Path Here"; // Path to the trained model
    const std::string csvPath = "Add Path Here"; // Path to the CSV file
    const std::string imageDir = "noisy_mri_dcm"; // Directory containing input images
    const std::string maskDir = "masks"; // Directory containing ground truth masks
    const std::string prefix = "Add Path Here"; // Path prefix for images

    // Create a Segmenter object, loading the model from the specified path
    Segmenter unetModel(modelPath);

    // Perform segmentation on all images, specifying directories, CSV file, prefix, and whether to save PNGs
    unetModel.segmentAll(imageDir, maskDir, csvPath, prefix, true);

    // Print a message indicating completion
    std::cout<<"Segmentation complete!"<<std::endl;

    return 0;
}
