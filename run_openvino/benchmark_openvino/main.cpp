# define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <sstream>
#include <filesystem>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <stdlib.h>
#include <Shlobj.h>
#include <chrono>

#include "infer_engine.h"

using namespace cv;
using namespace std;

// inference device
std::string infer_device = "GPU"; // CPU, GPU, VPUX

// model path
std::string model_path;
std::string cache_path = "../cache";

// input shape
int input_w;
int input_h;
int input_c = 3;

int user_input;
string user_input_str;
string input_folder = "../input";

void UserInput();
void PrintConfig();
void CVCamera();
void CVImage();

// color normalization
std::vector<float> color_norm_mean({ 0.485, 0.456, 0.406 });
std::vector<float> color_norm_std({ 0.229, 0.224, 0.225 });

// parameters for drawing depth map
float max_val = (pow(2, 8)) - 1;

bool model_status = false;

Mat frame, blob_image, show;
InferEngine* ai_core = nullptr;

// entry points
int main(int argc, char* argv[]) {
    UserInput();
    PrintConfig();

    std::vector<int> input_shape({ 1, input_c, input_h, input_w });

    // CV----------------------------------------------------------------------------
    // init inference engine 
    ai_core = new InferEngine(model_path, cache_path, infer_device, input_shape);
    
    if (input_folder == "")
        CVCamera();
    else
        CVImage();

    return EXIT_SUCCESS;
}

void UserInput() {
    // set input image folder
    std::cout << "Enter input image folder directory (default '../input' or press enter to use camera): ";
    std::getline(std::cin, input_folder);

    // set model
    std::vector<std::filesystem::path> model_list;
    for (const auto& file : std::filesystem::recursive_directory_iterator("../")) {
        if (file.path().extension().string() == ".xml" && file.path().filename().string() != "plugins.xml") model_list.push_back(file);
    }

    if (!model_list.size()) {
        printf("Can not find any IR model (.xml) please put it in root folder\n");
        exit;
    }
    else if (model_list.size() == 1) {
        model_path = model_list[0].string();
    }
    else {
        std::cout << "Find model: \n";
        for (int i = 0; i < model_list.size(); ++i) {
            printf("(%d): %s\n", i, model_list[i].string().c_str());
        }
        std::cout << "Choose model: ";
        std::cin >> user_input;
        model_path = model_list[min(user_input, model_list.size())].string();
        printf("\n");
    }

    // set model input shape
    std::cout << "Input model width  [default 256]: ";
    std::getline(std::cin, user_input_str);
    if (user_input_str == "") input_w = 256;
    else input_w = stoi(user_input_str);

    std::cout << "Input model height [default 256]: ";
    std::getline(std::cin, user_input_str);
    if (user_input_str == "") input_h = 256;
    else input_h = stoi(user_input_str);

    if (input_w % 32 != 0) {  // Input shape for model must be a multiple of 32
        input_w = input_w - (input_w % 32);
    }

    if (input_h % 32 != 0) {  // Input shape for model must be a multiple of 32
        input_h = input_h - (input_h % 32);
    }

    // Select adapter
    printf("\nUse CPU (0) or GPU (1) VPU (2) to inference: ");
    std::cin >> user_input;
    printf("\n");

    switch (user_input)
    {
    case (0):
        infer_device = "CPU";
        break;
    case (1):
        infer_device = "GPU";
        break;
    case (2):
        infer_device = "VPUX";
        break;
    default:
        infer_device = "CPU";
        break;
    }
}

void PrintConfig() {
    std::cout << "\nConfiguration: " << std::endl;
    std::cout << "Model Input Shape: ( 1 x 3 x " << input_h << " x " << input_w << ")" << std::endl;
    std::cout << std::endl;
}

void CVCamera() {
    // set camera 
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    while (true) {
        cap.read(frame);

        // pre-processing
        cv::resize(frame, frame, cv::Size(input_w, input_h));
        cv::cvtColor(frame, blob_image, cv::COLOR_BGR2RGB);
        blob_image.convertTo(blob_image, CV_32F, 1. / 255.);

        // mat to array
        float* pixels = (float*)(blob_image.data);

        // pass input array to blob
        ai_core->GetInputBlob(pixels, color_norm_mean, color_norm_std);

        // do inference
        ai_core->Inference();

        float* output_data = new float[input_w * input_h];
        ai_core->GetOutputBlob(output_data, true);


        std::vector<float> output_img(input_w * input_h);
        for (size_t i = 0; i < input_w * input_h; i++) {
            float val = static_cast<float>(output_data[i]);
            output_img[i] = val;
        }

        cv::Mat depth_map(input_h, input_w, CV_32FC1, output_img.data());
        depth_map = max_val * (depth_map);
        depth_map.convertTo(depth_map, CV_8UC1);

        cv::cvtColor(depth_map, depth_map, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, depth_map, show);

        cv::imshow("Output", show);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
}

void CVImage() {
    // check input folder
    if (!std::experimental::filesystem::is_directory(input_folder) || !std::experimental::filesystem::exists(input_folder)) {
        printf("\nWarning: folder '%s' is not exit ...", std::filesystem::current_path().string().append(input_folder).c_str());
        return;
    }

    // find all image in folder
    std::vector<std::filesystem::path> fn;
    std::vector<std::string> allowedextensions = { ".jpg", ".png", ".jpeg", ".bmp"};
    for (const auto& file : std::filesystem::recursive_directory_iterator(input_folder)) {
        for (int i = 0; i < allowedextensions.size(); i++) {
            if (file.path().extension().string() == allowedextensions[i]) fn.push_back(file);
        }
    }
    size_t count = fn.size(); 

    for (size_t i = 0; i < count; i++) {
        string input_path = fn[i].string();
        printf("Loading image from '%s'\n", input_path.c_str());

        // read image
        frame = cv::imread(input_path);
        const int org_h = frame.rows;
        const int org_w = frame.cols;

        // pre-processing
        cv::resize(frame, frame, cv::Size(input_w, input_h));
        cv::cvtColor(frame, blob_image, cv::COLOR_BGR2RGB);
        blob_image.convertTo(blob_image, CV_32F, 1. / 255.);

        // Mat to array
        float* pixels = (float*)(blob_image.data);

        // pass input array to blob
        ai_core->GetInputBlob(pixels, color_norm_mean, color_norm_std);

        // do inference
        ai_core->Inference();

        float* output_data = new float[input_w * input_h];
        ai_core->GetOutputBlob(output_data, true);

        std::vector<float> output_img(input_w * input_h);
        for (size_t i = 0; i < input_w * input_h; i++) {
            float val = static_cast<float>(output_data[i]);
            output_img[i] = val;
        }

        cv::Mat depth_map(input_h, input_w, CV_32FC1, output_img.data());
        depth_map = max_val * (depth_map);
        depth_map.convertTo(depth_map, CV_8UC1);

        cv::cvtColor(depth_map, depth_map, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, depth_map, show);

        cv::imshow("Output", show);
        cv::waitKey(1000);

        std::string base_filename = input_path.substr(input_path.find_last_of("/\\") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string output_name = base_filename.substr(0, p).append("_depth.png");

        std::string output_folder = "../output/";
        if (!std::experimental::filesystem::is_directory(output_folder) || !std::experimental::filesystem::exists(output_folder)) {
            std::experimental::filesystem::create_directory(output_folder); // create src folder
        }

        std::string output_path = output_folder + output_name;

        cv::resize(depth_map, depth_map, cv::Size(org_w, org_h));
        cv::imwrite(output_path, depth_map);
    }
}

