#pragma once
#include <Windows.h>
#include <chrono>
#include <queue>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace InferenceEngine;


class InferEngine {
public:
	InferEngine(
		std::string model_path,
		std::string cache_path,
		std::string infer_device,
		std::vector<int>& model_input_shape);

	~InferEngine();

	void GetInputBlob(float* buffer, std::vector<float>& mean, std::vector<float>& std);
	void GetOutputBlob(float* buffer, bool min_max_norm);
	void Inference();

	bool CheckModelStatus();

private:
	HRESULT Loadmodel(
		std::string model_path,
		std::string cache_path,
		std::string infer_device,
		std::vector<int>& model_input_shape);

	bool CheckFileExist(const std::string& file_path);

private:
	InferRequest infer_request;
	Blob::Ptr input_blob;
	Blob::Ptr output_blob;

	size_t input_n;
	size_t input_c;
	size_t input_h;
	size_t input_w;
	size_t input_size;

	size_t output_n;
	size_t output_h;
	size_t output_w;
	size_t output_size;

	ExecutableNetwork executable_network;

	std::string input_name;
	std::string output_name;


private:
	bool model_status = false;
};