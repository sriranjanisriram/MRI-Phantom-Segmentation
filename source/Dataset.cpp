// Author: Srirajani Sriram

#include <fstream>
#include <sstream>
#include <../headers/Dataset.h>

// Method to read a specific column from a CSV file
void SegmentationDataset::readColumnFromCSV(const std::string& filename, const std::string& prefix) {
    // Initialize a vector to store column data
    std::vector<std::string> column;
    // Initialize an unordered map to store column indices, mapping column names to their indices
    std::unordered_map<std::string, int> columnIndices;
    // Open the CSV file for reading
    std::ifstream file(filename);
    // Declare variables to store a line and a cell from the CSV file
    std::string line, cell;

    // Check if the file is open
    if (file.is_open()) {
        // Read the first line (header) from the file
        std::getline(file, line);
        // Create a string stream from the header line
        std::stringstream ss(line);
        // Initialize a variable to track the current column index
        int currentColumn = 0;

        // Iterate through each cell in the header line
        while (std::getline(ss, cell, ',')) {
            // Store the column index in the map
            columnIndices[cell] = currentColumn;
            // Increment the current column index
            currentColumn++;
        }

        // Check if the mask directory column exists in the header
        if (columnIndices.find(maskDir_) == columnIndices.end()) {
            // Print an error message if the column is not found
            std::cerr << "Column '" << maskDir_ << "' not found in the CSV file." << std::endl;

        }
        // Check if the image directory column exists in the header
        if (columnIndices.find(imageDir_) == columnIndices.end()) {
            // Print an error message if the column is not found
            std::cerr << "Column '" << imageDir_ << "' not found in the CSV file." << std::endl;
        }

        // Read the remaining lines in the CSV file
        while (std::getline(file, line)) {
            // Create a string stream from the current line
            std::stringstream ss(line);
            // Reset the current column index for the new line
            int currentColumn = 0;

            // Iterate through each cell in the current line
            while (std::getline(ss, cell, ',')) {
                // If the current column corresponds to the mask directory column
                if (currentColumn == columnIndices[maskDir_]) {
                    // Add the mask path to the mask paths vector, adding the prefix
                    maskPaths_.push_back(prefix + cell);
                } else if (currentColumn == columnIndices[imageDir_]) {
                    // If the current column corresponds to the image directory column, add the image path to the image paths vector, adding the prefix
                    imagePaths_.push_back(prefix + cell);
                }
                // Increment the current column index
                currentColumn++;
            }
        }

        // Close the CSV file
        file.close();
    } else {
        // If the file cannot be opened, print an error message
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    // Print a message indicating completion of CSV file reading
    std::cout << "done with reading csv" << std::endl;
}

// Method to normalize int16 data to float32
void SegmentationDataset::normalizeInt16ToFloat32(const int16_t* input, float* output, size_t size) {
    // Find the maximum value in the input array
    int16_t maxValue = *std::max_element(input, input + size);

    // Normalize the input array and convert to float32
    for (size_t i = 0; i < size; i++) {
        output[i] = static_cast<float>(input[i]) / static_cast<float>(maxValue);
    }
}

// Method to normalize uint16 data to float32
void SegmentationDataset::normalizeUInt16ToFloat32(const uint16_t* input, float* output, size_t size) {
    // Find the maximum value in the input array
    uint16_t maxValue = *std::max_element(input, input + size);

    // Normalize the input array and convert to float32
    for (size_t i = 0; i < size; i++) {
        output[i] = static_cast<float>(input[i]) / static_cast<float>(maxValue);
    }
}

// Constructor for the SegmentationDataset class
SegmentationDataset::SegmentationDataset(const std::string& imageDir, const std::string& maskDir, const std::string& csvFile,
                                         const std::string& prefix)
    : imageDir_(imageDir), maskDir_(maskDir) {
    // Load the image and mask file names from the CSV file
    readColumnFromCSV(csvFile, prefix);
}

// Method to retrieve a data sample at a given index
torch::data::Example<> SegmentationDataset::get(size_t index) {
    // Load the image using SimpleITK
    using ImageType = itk::Image<float, 3>;

    // Load the SimpleITK image
    itk::simple::ImageFileReader reader;
    reader.SetFileName(imagePaths_[index].c_str());
    itk::simple::Image itkImage = reader.Execute();

    // Create a PyTorch tensor from the C-style array
    int C = itkImage.GetNumberOfComponentsPerPixel();
    int H = itkImage.GetHeight();
    int W = itkImage.GetWidth();
    size_t arraySize = C * H * W;
    float itkImageDataFloat[arraySize];

    // Get the image data as a C-style array
    std::string pixelType = itkImage.GetPixelIDTypeAsString(); // dcm has mixed type for some reason
    if (pixelType == "16-bit signed integer") {
        int16_t* itkImageData = static_cast<int16_t*>(itkImage.GetBufferAsInt16());
        normalizeInt16ToFloat32(itkImageData, itkImageDataFloat, arraySize);
    } else {
        uint16_t* itkImageData = static_cast<uint16_t*>(itkImage.GetBufferAsUInt16());
        normalizeUInt16ToFloat32(itkImageData, itkImageDataFloat, arraySize);
    }

    // load mask using opencv
    cv::Mat mask = cv::imread(maskPaths_[index], cv::IMREAD_GRAYSCALE);

    // Convert the image to a PyTorch tensor and move it to CUDA
    torch::Tensor imageTensor = torch::from_blob(itkImageDataFloat, {C, H, W}, torch::kFloat32).to(at::kCUDA);

    // Convert the mask to a PyTorch tensor, convert its type to Long, and move it to CUDA
    torch::Tensor maskTensor = torch::from_blob(mask.data, {1, mask.rows, mask.cols}, torch::kByte).to(torch::kLong).to(at::kCUDA);

    // Return a data example containing the image and mask tensors
    return torch::data::Example<torch::Tensor, torch::Tensor>(imageTensor, maskTensor);
}

// Method to get the size of the dataset
torch::optional<size_t> SegmentationDataset::size() const {
    return imagePaths_.size();
}
