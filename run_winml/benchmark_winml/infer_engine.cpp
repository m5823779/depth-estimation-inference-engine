#include "infer_engine.h"


InferEngine::InferEngine(winrt::hstring model_path, const int inference_device, std::vector<int> input_size) {
    start_time = std::chrono::high_resolution_clock::now();

    SelectAdapter(inference_device);

    model = LearningModel::LoadFromFilePath(model_path);
    printf("Loading model file from '%ws' ...\n", model_path.c_str());

    // Set model input shape
    input_batch = model.InputFeatures().Size();
    input_channel = 3;
    input_height = input_size[0];
    input_width = input_size[1];
    size = input_batch * input_channel * input_height * input_width;

    // Create a session and binding
    LearningModelSessionOptions sessionOptions;
    // Define input dimensions 
    sessionOptions.OverrideNamedDimension(L"input_cx", input_height);
    sessionOptions.OverrideNamedDimension(L"input_cy", input_width);
    sessionOptions.BatchSizeOverride(1);
    session = LearningModelSession(model, device, sessionOptions);
    binding = LearningModelBinding(session);

    end_time = std::chrono::high_resolution_clock::now();
    time_count = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(end_time - start_time).count();
    printf("Take %d (ms) for initialization\n\n", (int)(time_count));
}

InferEngine::~InferEngine() { 
    if (binding != nullptr) {
        binding.Clear();
        binding = nullptr;
    }
    if (session != nullptr) {
        session.Close();
        //session = nullptr;
    }
    if (model != nullptr) {
        model.Close();
        model = nullptr;
    }
}

void InferEngine::GetInputBlob(const float* buffer) {
    // Create WinML tensor float
    std::vector<int64_t> inputShape = { input_batch, input_channel, input_height, input_width };
    //std::vector<int64_t> inputShape = { input_height, input_width, input_channel };

    TensorFloat tf = TensorFloat::Create(inputShape);

    // Create memory for input array
    float* pCPUInputTensor = nullptr;
    uint32_t uInputCapacity;

    // Gets the tensor buffer as an bytes array
    winrt::com_ptr<ITensorNative>  itn = tf.as<ITensorNative>();
    itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUInputTensor), &uInputCapacity);

    // Feed input data
    for (int i = 0; i < size; ++i) {
        UINT32 index = i / input_channel;
        for (int j = 0; j < input_channel; ++j) {
            pCPUInputTensor[(input_width * input_height * j) + index] = buffer[i + j];
        }
    }
    //for (int i = 0; i < size; ++i) {
    //    pCPUInputTensor[i] = buffer[i];
    //}

    // Create binding and then bind input features
    binding.Clear();
    auto&& description = model.InputFeatures().GetAt(0);
    binding.Bind(description.Name(), tf);
}

void InferEngine::GetOutputBlob(float* buffer, bool min_max_norm) {
    TensorFloat resultTensor = result.Outputs().Lookup(session.Model().OutputFeatures().First().Current().Name()).as<TensorFloat>();

    // Get output shape
    std::vector<int> shape = {};
    for (auto item : resultTensor.Shape()) shape.push_back(item);

    // Gets the tensor buffer as an bytes array
    uint32_t uCapacity;
    winrt::com_ptr<ITensorNative> itn_out = resultTensor.as<ITensorNative>();
    float* output_data = nullptr;
    itn_out->GetBuffer(reinterpret_cast<BYTE**>(&output_data), &uCapacity);


    if (min_max_norm) {
        float maxVal = FLT_MIN;
        float minVal = FLT_MAX;

        for (size_t i = 0; i < shape[1] * shape[2]; i++) {
            float val = static_cast<float>(output_data[i]);

            if (val > maxVal)
                maxVal = val;
            else if (val < minVal)
                minVal = val;
        }

        for (size_t i = 0; i < shape[1] * shape[2]; i++) {
            float val = static_cast<float>(output_data[i]);
            //*buffer = (val - minVal) / (maxVal - minVal);
            *buffer = (maxVal - val) / (maxVal - minVal);
            buffer++;
        }
    }
    else {
        for (size_t i = 0; i < shape[1] * shape[2]; i++) {
            float val = static_cast<float>(output_data[i]);
            *buffer = val;
            buffer++;
        }
    }
}


void InferEngine::Inference() {
    start_time = std::chrono::high_resolution_clock::now();

    // Inference
    result = session.Evaluate(binding, L"");

    end_time = std::chrono::high_resolution_clock::now();
    time_count = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(end_time - start_time).count();
    printf("Inference time : %d (ms)\n", (int)(time_count));
}

void InferEngine::SelectAdapter(const int inference_device) {
    if (!inference_device)
        device = LearningModelDevice(LearningModelDeviceKind::Cpu);
    else {
        D3D_FEATURE_LEVEL FeatureLevels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_9_1 };
        UINT NumFeatureLevels = ARRAYSIZE(FeatureLevels);
        D3D_FEATURE_LEVEL FeatureLevel;

        UINT i = 0;
        IDXGIAdapter* pAdapter;
        std::vector <IDXGIAdapter*> vAdapters;
        IDXGIFactory1* pFactory = NULL;
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);

        // Show all adapter
        while (pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND)
        {
            DXGI_ADAPTER_DESC adapter_desc;
            pAdapter->GetDesc(&adapter_desc);
            printf("Adapter %d: %ls\n", i, adapter_desc.Description);
            vAdapters.push_back(pAdapter);
            ++i;
        }
        printf("Choose adapter (0 - %d): ", i - 1);
        int user_input_adapter;
        std::cin >> user_input_adapter;
        printf("\n");

        int inference_adapter = user_input_adapter;

        // Create D3D11 Device
        winrt::com_ptr<ID3D11Device> m_inference_device;
        winrt::com_ptr<ID3D11DeviceContext> m_inference_context;
        D3D11CreateDevice(vAdapters[inference_adapter], D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, FeatureLevels, NumFeatureLevels, D3D11_SDK_VERSION, m_inference_device.put(), &FeatureLevel, m_inference_context.put());

        IDXGIDevice* m_DxgiDevice = nullptr;
        m_inference_device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&m_DxgiDevice));

        auto m_d3d_device = CreateDirect3DDevice(m_DxgiDevice);
        device = LearningModelDevice::CreateFromDirect3D11Device(m_d3d_device);
    }
}