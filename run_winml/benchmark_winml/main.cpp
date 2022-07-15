#include "pch.h"
#include "infer_engine.h"

using namespace cv;

// get model path
winrt::hstring model_path;

int input_width;
int input_height;

int user_input;
int inference_device;

std::string user_input_str;
std::string input_folder = "../input";

void UserInput();
void PrintConfig();
void SelectAdapter();

void CVCamera();
void CVImage();

float max_val = (pow(2, 8)) - 1;

Mat frame, blob_image, show;
InferEngine* ai_core = nullptr; 

// Entry
int main(int argc, char* argv[]) {

    UserInput();
    PrintConfig();

    std::vector<int> input_size{ input_height, input_width };
    ai_core = new InferEngine(model_path, inference_device, input_size);

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

    // set model input shape
    std::cout << "Input model width  [default 256]: ";
    std::getline(std::cin, user_input_str);
    if (user_input_str == "") input_width = 256;
    else input_width = stoi(user_input_str);

    std::cout << "Input model height [default 256]: ";
    std::getline(std::cin, user_input_str);
    if (user_input_str == "") input_height = 256;
    else input_height = stoi(user_input_str);

    if (input_width % 32 != 0) {  // Input shape for model must be a multiple of 32
        input_width = input_width - (input_width % 32);
    }

    if (input_height % 32 != 0) {  // Input shape for model must be a multiple of 32
        input_height = input_height - (input_height % 32);
    }

    // set model
    std::vector<std::filesystem::path> model_list;
    for (const auto& file : std::filesystem::recursive_directory_iterator("../")) {
        if (file.path().extension().string() == ".onnx") model_list.push_back(file);
    }

    if (!model_list.size()) {
        printf("Can not find any IR model (.xml) please put it in root folder\n");
        exit;
    }
    else if (model_list.size() == 1) {
        model_path = model_list[0].wstring();
    }
    else {
        std::cout << "Find model: \n";
        for (int i = 0; i < model_list.size(); ++i) {
            printf("(%d): %s\n", i, model_list[i].string().c_str());
        }
        std::cout << "Choose model: ";
        std::cin >> user_input;
        model_path = model_list[min(user_input, model_list.size())].wstring();
        printf("\n");
    }

    // Select adapter
    printf("\nUse CPU (0) or GPU (1) to inference: ");
    std::cin >> inference_device;
    printf("\n");
}

void PrintConfig() {
    std::cout << "\nConfiguration: " << std::endl;
    std::cout << "Model Input Shape: ( 1 x 3 x " << input_height << " x " << input_width << ")" << std::endl;
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
        cv::resize(frame, frame, cv::Size(input_width, input_height));
        cv::cvtColor(frame, blob_image, cv::COLOR_BGR2RGB);
        blob_image.convertTo(blob_image, CV_32F, 1. / 255.);

        // Mat to array
        const float* pixels = (float*)(blob_image.data);

        // pass input array to blob
        ai_core->GetInputBlob(pixels);

        // do inference
        ai_core->Inference();

        float* output_data = new float[input_width * input_height];
        ai_core->GetOutputBlob(output_data, true);

        std::vector<float> output_img(input_width * input_height);
        for (size_t i = 0; i < input_width * input_height; i++) {
            float val = static_cast<float>(output_data[i]);
            output_img[i] = val;
        }

        cv::Mat depth_map(input_height, input_width, CV_32FC1, output_img.data());
        depth_map = max_val * (depth_map);
        depth_map.convertTo(depth_map, CV_8UC1);

        cv::cvtColor(depth_map, depth_map, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, depth_map, show);

        cv::imshow("Output", show);
        cv::waitKey(1);
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
    std::vector<std::string> allowedextensions = { ".jpg", ".png", ".jpeg", ".bmp" };
    for (const auto& file : std::filesystem::recursive_directory_iterator(input_folder)) {
        for (int i = 0; i < allowedextensions.size(); i++) {
            if (file.path().extension().string() == allowedextensions[i]) fn.push_back(file);
        }
    }
    size_t count = fn.size();

    for (size_t i = 0; i < count; i++) {
        std::string input_path = fn[i].string();
        printf("Loading image from '%s'\n", input_path.c_str());

        // read image
        frame = cv::imread(input_path);
        const int org_h = frame.rows;
        const int org_w = frame.cols;

        // pre-processing
        cv::resize(frame, frame, cv::Size(input_width, input_height));
        cv::cvtColor(frame, blob_image, cv::COLOR_BGR2RGB);
        blob_image.convertTo(blob_image, CV_32F, 1. / 255.);

        //frame.convertTo(blob_image, CV_32F);

        // Mat to array
        const float* pixels = (float*)(blob_image.data);

        // pass input array to blob
        ai_core->GetInputBlob(pixels);

        // do inference
        ai_core->Inference();

        float* output_data = new float[input_width * input_height];
        ai_core->GetOutputBlob(output_data, true);
        //ai_core->GetOutputBlob(output_data, false);

        std::vector<float> output_img(input_width * input_height);
        for (size_t i = 0; i < input_width * input_height; i++) {
            float val = static_cast<float>(output_data[i]);
            output_img[i] = val;
        }

        cv::Mat depth_map(input_height, input_width, CV_32FC1, output_img.data());
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



