#ifndef DATASET_H  // Header guard to prevent multiple inclusions
#define DATASET_H


// Author: Srirajani Sriram

// Include necessary headers
#include <torch/torch.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <SimpleITK.h>

#include <cstddef>
#include <string>


// SegmentationDataset class inherits from torch::data::datasets::Dataset,
// which provides a standard interface for datasets in LibTorch.
class SegmentationDataset : public torch::data::datasets::Dataset<SegmentationDataset> {
private:
    // Member variables to store image directory, mask directory,
    // and file paths for images and masks.
    std::string imageDir_; // Path to the directory containing the images.
    std::string maskDir_;  // Path to the directory containing the segmentation masks.
    std::vector<std::string> imagePaths_; // List of image file paths
    std::vector<std::string> maskPaths_;  // List of mask file paths

public:

    // Constructor
    // image_dir: Path to the directory containing the images (string).
    // mask_dir: Path to the directory containing the segmentation masks (string).
    // csv_file: Path to the CSV file containing image and mask filenames (string).
    // prefix: Path prefix to prepend to the filenames in the CSV (string).
    SegmentationDataset(const std::string& imageDir, const std::string& maskDir, const std::string& csvFile, const std::string& prefix);

    // Returns the size of the dataset (number of image/mask pairs)
    torch::optional<size_t> size() const override;

    // Returns a data sample at the given index
    // index: Index of the data sample to retrieve (size_t).
    torch::data::Example<> get(size_t index) override;

    // Reads a specific column from a CSV file and stores the values in a vector.
    // filename: Path to the CSV file to read (string).
    // prefix: Path prefix to prepend to the read values (string).
    void readColumnFromCSV(const std::string& filename, const std::string& prefix);

    // Normalizes int16_t data to float32 in the range [0, 1].
    // input: Pointer to the input int16_t data.
    // output: Pointer to the output float32 data.
    // size: Size of the data arrays (size_t).
    void normalizeInt16ToFloat32(const int16_t* input, float* output, size_t size);

    // Normalizes uint16_t data to float32 in the range [0, 1].
    // input: Pointer to the input uint16_t data.
    // output: Pointer to the output float32 data.
    // size: Size of the data arrays (size_t).
    void normalizeUInt16ToFloat32(const uint16_t* input, float* output, size_t size);
};

#endif // DATASET_H
