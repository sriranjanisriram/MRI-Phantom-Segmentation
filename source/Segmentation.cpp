// Author: Srirajani Sriram

#include "../headers/Segmentation.h"

// Function to save vectors to a CSV file
void saveVectorsToCSV(const std::string& filename,
                      const std::vector<float>& iouUnet,
                      const std::vector<float>& iouCv,
                      const std::vector<float>& iouDenoisedCv) {
    // Define the path for the output folder where the csv will be saved
    std::filesystem::path outputFolder = "output_folder";

    // Define the full output file path within the output folder
    std::filesystem::path outputFilePath = outputFolder / filename;

    // Open the output file stream
    std::ofstream file(outputFilePath);

    // Write header row
    file << "IOU U-Net,IOU OpenCV,IOU OpenCV Denoised\n";

    // Write data rows
    for (size_t i = 0; i < iouUnet.size(); ++i) {
        file << iouUnet[i] << "," << iouCv[i] << "," << iouDenoisedCv[i] << "\n";
    }

    // Calculate average
    float avgIouUnet = std::accumulate(iouUnet.begin(), iouUnet.end(), 0.0f);
    avgIouUnet = avgIouUnet / iouUnet.size();

    float avgiouCv = std::accumulate(iouCv.begin(), iouCv.end(), 0.0f);
    avgiouCv = avgiouCv / iouCv.size();

    float avgiouDenoisedCv = std::accumulate(iouDenoisedCv.begin(), iouDenoisedCv.end(), 0.0f);
    avgiouDenoisedCv = avgiouDenoisedCv / iouDenoisedCv.size();

    // write average
    file << avgIouUnet << "," << avgiouCv << "," << avgiouDenoisedCv << "\n";

    // Close the file
    file.close();
    std::cout << "CSV file saved: " << filename << std::endl;
}

// Constructor for the Segmenter class
Segmenter::Segmenter(const std::string& modelPath) {
    // Load the PyTorch model
    model_ = torch::jit::load(modelPath);
    // Move the model to CUDA
    model_.to(torch::kCUDA);
    // Set the model to evaluation mode
    model_.eval();
}

// Function to score using a Python wrapper (currently unused)
float Segmenter::scorePyWrapper(const at::Tensor& predMask, const at::Tensor& trueMask){

    /*
     * This function is not used as the one have to setup the pytorch for python-dev version and cannot use conda environments
     *
     */

    // Unused code (commented out)
    std::string modulePath = "/home/maniraman/Desktop/PMRI/PMRI/source";

    Py_Initialize();
    torch::Tensor cppTensor = torch::ones({ 100 });
    PyRun_SimpleString(("import sys\nsys.path.append(\"" + modulePath + "\")").c_str());
    boost::python::object module = boost::python::import("score");
    boost::python::object pythonFunction = module.attr("iou_score");

    PyObject* castedPredMask = THPVariable_Wrap(predMask);
    //PyObject* castedTrueMask = THPVariable_Wrap(trueMask);

    boost::python::handle<> boostHandlePM(castedPredMask);
    boost::python::object inputTensorPM(boostHandlePM);

    //boost::python::handle<> boostHandleTM(castedTrueMask);
    //boost::python::object inputTensorTM(castedTrueMask);

    boost::python::object result = pythonFunction(inputTensorPM);

    float iou = boost::python::extract<float>(result);

    Py_Finalize();
    return iou;

}

// Function to calculate intersection over union (IoU) for a batch of masks
torch::Tensor Segmenter::batchIOUScore(const torch::Tensor& predMasks, const torch::Tensor& trueMasks) {
    // Calculate intersection
    torch::Tensor intersection = torch::logical_and(predMasks, trueMasks).sum({1, 2, 3}).toType(torch::kFloat);

    // Calculate union
    torch::Tensor union_ = torch::logical_or(predMasks, trueMasks).sum({1, 2, 3}).toType(torch::kFloat);

    // Calculate IoU with division by zero handling
    torch::Tensor iou = intersection / (union_ + 1e-10f); // Avoid division by zero

    return iou;
}


// Function to convert a PyTorch tensor to an OpenCV image
cv::Mat Segmenter::tensorToOpencv(const torch::Tensor& tensor) {
    // Ensure tensor is on CPU
    torch::Tensor tensorCpu = tensor.to(torch::kCPU);

    // Remove unnecessary dimension
    tensorCpu = tensorCpu.squeeze(1);

    // Get tensor dimensions
    auto dims = tensorCpu.sizes();

    // Create OpenCV Mat with appropriate data type and dimensions
    tensorCpu = tensorCpu.contiguous();
    cv::Mat mat(dims[1], dims[2], CV_32FC1, tensorCpu.data_ptr());

    // Normalize tensor values to [0, 255]
    mat = mat * 255.0;

    // Convert to 8-bit unsigned integer
    cv::Mat image;
    mat.convertTo(image, CV_8UC1);

    return image.clone();
}

// Function to create a grid of subplots with images and subtitles
void Segmenter::createSubplots(const std::vector<cv::Mat>& images, const std::vector<std::string>& subtitles, int rows, int cols, const std::string& outputPath) {

  // Check if the number of images exceeds the available space in the grid
  int numImages = static_cast<int>(images.size());
  if (numImages > rows * cols) {
    std::cerr << "Error: Not enough subplots for the given number of images." << std::endl;
    return;
  }

  // Find the maximum height and width of all images
  int maxHeight = 0, maxWidth = 0;
  for (const auto& img : images) {
    maxHeight = std::max(maxHeight, img.rows);  // Update maxHeight if current image is taller
    maxWidth = std::max(maxWidth, img.cols);  // Update maxWidth if current image is wider
  }

  // Calculate the total height and width of the canvas based on number of rows, columns, image sizes and spacing
  int canvasHeight = maxHeight * rows + (rows) * 50 + 25;
  int canvasWidth = maxWidth * cols + (cols) * 25 + 25;

  // Create a new OpenCV Mat to hold the entire subplot collage
  cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC1, cv::Scalar(255));  // Single channel 8-bit unsigned integer with white background

  // Variables to track current row and column positions for placing images
  int currentRow = 0, currentCol = 0;

  // Loop through all images
  for (int i = 0; i < numImages; ++i) {

    // Define a region of interest (ROI) within the canvas for the current image
    cv::Mat roi(canvas, cv::Rect(currentCol * (maxWidth + 25)+25, currentRow * (maxHeight + 50)+ 25, maxWidth, maxHeight));

    // Copy the current image into its designated ROI on the canvas
    images[i].copyTo(roi);

    // Font properties for adding subtitles
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.85;
    float thickness = 1;
    int baseline = 0;

    // Get the text size of the current subtitle for positioning
    cv::Size textSize = cv::getTextSize(subtitles[i], fontFace, fontScale, thickness, &baseline);

    // Calculate the coordinates to center the subtitle text within its image ROI
    cv::Point textPos(currentCol * (maxWidth + 25) + 25 + (maxWidth - textSize.width)/2, (currentRow + 1) * (maxHeight+50) + 25 - 20);

    // Add the subtitle text to the image ROI on the canvas
    cv::putText(canvas, subtitles[i], textPos, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);

    // Move to the next column after placing the image and subtitle
    currentCol++;

    // Wrap around to the first column if we reach the end of a row
    if (currentCol == cols) {
      currentCol = 0;
      currentRow++;
    }
  }

  // Define the path for the output folder where the subplot collage will be saved
  std::filesystem::path outputFolder = "output_folder";

  // Create the output folder if it doesn't already exist
  if (!std::filesystem::exists(outputFolder)) {
    std::filesystem::create_directory(outputFolder);
    std::cout << "Created folder: " << outputFolder << std::endl;
  }

  // Define the full output file path within the output folder
  std::filesystem::path outputFilePath = outputFolder / outputPath;

  // Save the subplot collage as a PNG image
  cv::imwrite(outputFilePath.string(), canvas);

  // Print a success message indicating where the image was saved
  std::cout << "Images saved as subplots to: " << outputPath << std::endl;
}

// Function to denoise an image using a median filter
cv::Mat Segmenter::denoiseImage(const cv::Mat& image){
    // Kernel size for the median filter
    int kernelSize = 3;

    // Apply median blur to the image
    cv::Mat denoised;
    cv::medianBlur(image, denoised, kernelSize);

    return denoised;
}


// Function to threshold an image
cv::Mat Segmenter::thresholdImage(const cv::Mat& image, int threshold){
    // Apply thresholding to the image
    cv::Mat thresholded;
    cv::threshold(image, thresholded, threshold, 255, cv::THRESH_BINARY);

    return thresholded;
}

// Function to calculate IoU score using OpenCV
float Segmenter::cvIOUScore(const cv::Mat& predMasks, const cv::Mat& trueMasks){
    // Calculate intersection
    cv::Mat intersection;
    cv::bitwise_and(predMasks, trueMasks, intersection);
    float intersectionArea = cv::countNonZero(intersection);

    // Calculate union
    cv::Mat union_;
    cv::bitwise_or(predMasks, trueMasks, union_);
    float unionArea = cv::countNonZero(union_);

    // Calculate IoU with division by zero handling
    float iou = intersectionArea / (unionArea + 1e-10);

    return iou;
}

// Function to perform segmentation on all images in a directory
void Segmenter::segmentAll(const std::string& imageDir, const std::string& maskDir, const std::string& csvFile,
                            const std::string& prefix, bool savePng){

    // Create a segmentation dataset
    auto dataset = SegmentationDataset(imageDir, maskDir, csvFile, prefix).map(torch::data::transforms::Stack<>());

    // Create a data loader
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), 1);

    // Initialize batch ID and vectors to store IoU scores
    int batchId = 0;
    std::vector<float> iouUnet;
    std::vector<float> iouCv;
    std::vector<float> iouDenoisedCv;

    // Iterate over the data loader
    for (auto& batch : *dataLoader) {
        // Get data and target from the batch
        auto data = batch.data;
        auto target = batch.target;

        // Perform inference using the model
        torch::Tensor outputs = model_.forward({data}).toTensor();

        // Apply sigmoid activation and thresholding
        outputs = torch::sigmoid(outputs);
        outputs = torch::where(outputs < 0.5, torch::zeros_like(outputs), torch::ones_like(outputs));

        // Calculate IoU scores
        auto ious = Segmenter::batchIOUScore(outputs, target);

        // Convert tensors to OpenCV images
        cv::Mat dataCv = Segmenter::tensorToOpencv(data);
        cv::Mat targetCv = Segmenter::tensorToOpencv(target.to(torch::kFloat32));
        cv::Mat outputsCv = Segmenter::tensorToOpencv(outputs);
        cv::Mat thresholdCv = Segmenter::thresholdImage(dataCv, 25);

        // Denoise input image and apply thresholding
        cv::Mat denoisedInput = Segmenter::denoiseImage(dataCv);
        cv::Mat denoisedThresholdCv = Segmenter::thresholdImage(denoisedInput, 25);

        // Calculate IoU scores for OpenCV-based segmentation
        float thresholdCvIou = Segmenter::cvIOUScore(thresholdCv, targetCv);
        float denoisedThresholdCvIou = Segmenter::cvIOUScore(denoisedThresholdCv, targetCv);

        // Store IoU scores in respective vectors
        iouUnet.push_back(ious[0].item<float>());
        iouCv.push_back(thresholdCvIou);
        iouDenoisedCv.push_back(denoisedThresholdCvIou);

        if (savePng){

            // Create vectors to hold images and subtitles for subplots
            std::vector<cv::Mat> images = {dataCv, dataCv, denoisedInput, outputsCv, thresholdCv, denoisedThresholdCv};
            std::vector<std::string> subtitles = {"Input", "Input", "Denoised", "U-Net IOU:" + std::to_string(ious[0].item<float>()),
                                                  "Open CV IOU:" + std::to_string(thresholdCvIou),
                                                  "Open CV Denoised IOU:" + std::to_string(denoisedThresholdCvIou)};

            // Create subplots with subtitles
            Segmenter::createSubplots(images, subtitles, 2, 3, "subplots_with_subtitles_" +  std::to_string(batchId)  +".png");
        }

        // Print IoU scores for the current batch
        for (int i = 0; i < ious.size(0); ++i) {
            std::cout << "Image " << batchId + 1 << ": " << ious[i].item<float>() << std::endl;
        }

        // Increment batch ID
        batchId++;

        // Break the loop after processing 10 batches
        //if (batchId > 10){
        //    break;
        //}
    }

    // Save IoU scores to a CSV file
    saveVectorsToCSV("IOU_scores.csv", iouUnet, iouCv, iouDenoisedCv);
}
