#ifndef SEGMENTER_H
#define SEGMENTER_H

// Author: Srirajani Sriram

// Include necessary headers
#include <Dataset.h>
#include <torch/script.h>
#include <torch/python.h>
#include <torch/csrc/autograd/python_variable.h>
#include <boost/python.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>


// Class for performing image segmentation
class Segmenter {
public:
    // Constructor
    // model_path: Path to the PyTorch model file. (string)
    Segmenter(const std::string& modelPath);

    // Function to perform segmentation on all images in a directory.
    // image_dir: Path to the directory containing the images. (string)
    // mask_dir: Path to the directory containing the segmentation masks. (string)
    // csv_file: Path to the CSV file containing image and mask filenames. (string)
    // prefix: Path prefix to prepend to the filenames in the CSV. (string)
    // save_png: Boolean flag indicating whether to save segmented images as PNG files. (bool)
    void segmentAll(const std::string& imageDir, const std::string& maskDir, const std::string& csvFile,
                     const std::string& prefix, bool savePng);

    // Function to score using a Python wrapper (currently unused).
    // pred_mask: Predicted segmentation mask tensor. (at::Tensor)
    // true_mask: Ground truth segmentation mask tensor. (at::Tensor)
    float scorePyWrapper(const at::Tensor& predMask, const at::Tensor& trueMask);

    // Function to calculate IoU score for a batch of masks.
    // pred_masks: Batch of predicted segmentation masks. (torch::Tensor)
    // true_masks: Batch of ground truth segmentation masks. (torch::Tensor)
    torch::Tensor batchIOUScore(const torch::Tensor& predMasks, const torch::Tensor& trueMasks);

    // Function to convert a PyTorch tensor to an OpenCV image.
    // tensor: PyTorch tensor representing an image. (torch::Tensor)
    cv::Mat tensorToOpencv(const torch::Tensor& tensor);

    // Function to create a grid of subplots with images and subtitles.
    // images: Vector of OpenCV images to display in the subplots. (std::vector<cv::Mat>)
    // subtitles: Vector of subtitles for each image. (std::vector<std::string>)
    // rows: Number of rows in the subplot grid. (int)
    // cols: Number of columns in the subplot grid. (int)
    // outputPath: Path to save the generated subplot image. (std::string)
    void createSubplots(const std::vector<cv::Mat>& images, const std::vector<std::string>& subtitles, int rows, int cols, const std::string& outputPath);

    // Function to denoise an image using a median filter.
    // image: Input image to denoise. (cv::Mat)
    cv::Mat denoiseImage(const cv::Mat& image);

    // Function to threshold an image.
    // image: Input image to threshold. (cv::Mat)
    // threshold: Threshold value for binarization. (int)
    cv::Mat thresholdImage(const cv::Mat& image, int threshold);

    // Function to calculate IoU score using OpenCV.
    // pred_masks: Predicted segmentation mask image. (cv::Mat)
    // true_masks: Ground truth segmentation mask image. (cv::Mat)
    float cvIOUScore(const cv::Mat& predMasks, const cv::Mat& trueMasks);

private:
    // PyTorch JIT module for the segmentation model
    torch::jit::script::Module model_;
};

// Function to save vectors to a CSV file.
// filename: Path to the output CSV file. (std::string)
// iou_unet: Vector of IoU scores for the U-Net model. (std::vector<float>)
// iou_cv: Vector of IoU scores for OpenCV-based segmentation. (std::vector<float>)
// iou_denoised_cv: Vector of IoU scores for OpenCV-based segmentation with denoising. (std::vector<float>)
void saveVectorsToCSV(const std::string& filename,
                      const std::vector<float>& iouUnet,
                      const std::vector<float>& iouCv,
                      const std::vector<float>& iouDenoisedCv);



#endif // SEGMENTER_H
