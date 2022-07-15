# define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <iostream>
#include <string>
#include<sstream> 
#include <codecvt>
#include <cstdlib>

// OpenCV
#include <opencv2/opencv.hpp>

#include <Windows.h>
#include <wrl.h>

// D3D 
#include <d3d11.h>
#include "direct3d11.interop.h"

#include <filesystem>
#include <experimental/filesystem>

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;



