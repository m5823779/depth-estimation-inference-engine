#pragma once
#include "pch.h"
// WinRT
#include <winrt/Windows.Foundation.Collections.h>

// WinML includes
#include <winrt/Microsoft.AI.MachineLearning.h>
#include "Microsoft.AI.Machinelearning.Native.h" 

using namespace winrt::Windows::Media;
using namespace winrt::Microsoft::AI::MachineLearning;

class InferEngine {
public:
	InferEngine(winrt::hstring model_path, const int inference_device, std::vector<int> input_size);
	~InferEngine();

	void GetInputBlob(const float* buffer);
	void GetOutputBlob(float* buffer, bool min_max_norm);
	void Inference();

private:
	void SelectAdapter(const int inference_device);

private:
	LearningModel model = nullptr;
	LearningModelSession session = nullptr;
	LearningModelDevice device = nullptr;
	LearningModelBinding binding = nullptr;
	LearningModelEvaluationResult result = nullptr;

	int input_batch;
	int input_channel;
	int input_height;
	int input_width;
	uint32_t size;

private:
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point end_time;
	float time_count;
};