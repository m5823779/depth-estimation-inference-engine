#define TIMER   

#include "infer_engine.h"

InferEngine::InferEngine(
    std::string model_path,
    std::string cache_path,
    std::string infer_device,
    std::vector<int>& model_input_shape) {

    HRESULT hr;
    hr = Loadmodel(model_path, cache_path, infer_device, model_input_shape);
}

InferEngine::~InferEngine() {
    infer_request.Cancel();
    input_blob.reset();
    output_blob.reset();
}

void InferEngine::GetInputBlob(float* buffer, std::vector<float>& mean, std::vector<float>& std) {

    float* input_tensor = static_cast<float*>(input_blob->buffer());

    float* temp = buffer;
    for (int i = 0; i < input_size; ++i) {
        UINT32 index = i / input_c;
        //input_tensor[(input_w * input_h * 0) + index] = (temp[i + 0] - 123.675 ) / 58.395;
        //input_tensor[(input_w * input_h * 1) + index] = (temp[i + 1] - 116.28) / 57.12;
        //input_tensor[(input_w * input_h * 2) + index] = (temp[i + 2] - 103.53) / 57.375;

        for (int j = 0; j < input_c; ++j) {
            input_tensor[(input_w * input_h * j) + index] = (temp[i + j] - mean[j]) / std[j];
        }
    }
}

void InferEngine::GetOutputBlob(float* buffer, bool min_max_norm) {
    float* output_data = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    if (min_max_norm) {
        float maxVal = FLT_MIN;
        float minVal = FLT_MAX;

        for (size_t i = 0; i < output_h * output_w; i++) {
            float val = static_cast<float>(output_data[i]);

            if (val > maxVal)
                maxVal = val;
            else if (val < minVal)
                minVal = val;
        }

        for (size_t i = 0; i < output_h * output_w; i++) {
            float val = static_cast<float>(output_data[i]);
            //*buffer = (val - minVal) / (maxVal - minVal);
            *buffer = (maxVal - val) / (maxVal - minVal);
            buffer++;
        }
    }
    else {
        for (size_t i = 0; i < output_h * output_w; i++) {
            float val = static_cast<float>(output_data[i]);
            *buffer = val;
            buffer++;
        }
    }
}

void InferEngine::Inference() {
#ifdef TIMER
    std::chrono::steady_clock::time_point start_time = std::chrono::high_resolution_clock::now();
#endif

    infer_request.Infer();

#ifdef TIMER
    std::chrono::steady_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    float time_count = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(end_time - start_time).count();
    wchar_t msg[128];
    std::printf("Inference time : %d (ms)\n", (int)(time_count));
#endif
}

HRESULT InferEngine::Loadmodel(
    std::string model_path,
    std::string cache_path,
    std::string infer_device,
    std::vector<int> &model_input_shape) {

    std::string bin_path = model_path.substr(0, model_path.find_last_of('.')) + ".bin";
    std::string blob_path = model_path.substr(0, model_path.find_last_of('.')) + ".blob";

    if (!CheckFileExist(model_path) || !CheckFileExist(bin_path)) {
        wchar_t msg[128];
        swprintf_s(msg, L"%s not exist ...\n", model_path);
        OutputDebugStringW(msg);
        return EXIT_FAILURE;
    }

    // output message
    wchar_t msg[128];

    try {
        // step 1. initialize inference engine core
        InferenceEngine::Core ie;

        // saving cache file to speed up model loading
        if (!cache_path.empty()) {
            ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), cache_path} });
            printf("Saving cache file to %s ...\n", cache_path.c_str());

        }

        // step 2. load IR / ONNX model
        if (infer_device == "VPUX") {
            executable_network = ie.ImportNetwork("../midas_512x288.blob", "VPUX");
            printf("Loading model from %s ...\n", blob_path.c_str());

            InferenceEngine::ConstInputsDataMap inputs = executable_network.GetInputsInfo();
            InferenceEngine::ConstOutputsDataMap outputs = executable_network.GetOutputsInfo();

            input_name = inputs.begin()->first;
            output_name = outputs.begin()->first;
        }
        else {
            CNNNetwork network = ie.ReadNetwork(model_path, bin_path);
            printf("Loading model from %s ...\n", model_path.c_str());

            // step 3. get input & output format (allow to set input precision & get input / output layer name)
            InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
            InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

            auto input_shapes = network.getInputShapes();

            InferenceEngine::SizeVector input_shape;
            std::tie(input_name, input_shape) = *input_shapes.begin(); // let's consider first input only

            // set input shape
            for (unsigned i = 0; i < input_shape.size(); ++i) {
                input_shape[i] = model_input_shape[i];
            }

            //input_shape[0] = 1; // set batch size
            //input_shape[0] = 1; // set channel size
            //input_shape[2] = 256; // set input height
            //input_shape[3] = 256; // set input width

            input_shapes[input_name] = input_shape;
            network.reshape(input_shapes);

            inputs.begin()->second->setLayout(Layout::NCHW);
            inputs.begin()->second->setPrecision(Precision::FP32);
            inputs.begin()->second->getPreProcess().setColorFormat(ColorFormat::RGB);

            output_name = outputs.begin()->first;
            outputs.begin()->second->setPrecision(Precision::FP32);

            // step 4. loading a model to the device
            executable_network = ie.LoadNetwork(network, infer_device);
            printf("Loading model to device: %s ...\n", infer_device);
        }

        // model status
        model_status = true;

        // step 5. create an infer request
        infer_request = executable_network.CreateInferRequest();
        

        input_blob = infer_request.GetBlob(input_name);
        input_n = input_blob->getTensorDesc().getDims()[0];
        input_c = input_blob->getTensorDesc().getDims()[1];
        input_h = input_blob->getTensorDesc().getDims()[2];
        input_w = input_blob->getTensorDesc().getDims()[3];
        input_size = input_h * input_w * input_c;
        printf("Input shape: [ %d x %d x %d x %d ]\n", input_n, input_c, input_h, input_w);

        output_blob = infer_request.GetBlob(output_name);
        output_n = output_blob->getTensorDesc().getDims()[0];
        output_h = output_blob->getTensorDesc().getDims()[1];
        output_w = output_blob->getTensorDesc().getDims()[2];
        output_size = output_h * output_w;
        printf("Output shape: [ %d x %d ]\n", output_h, output_w);
        
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}

bool InferEngine::CheckFileExist(const std::string& file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}

bool InferEngine::CheckModelStatus() {
    return model_status;
}

