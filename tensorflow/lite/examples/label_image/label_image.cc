/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/examples/label_image/label_image.h"
#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <arpa/inet.h>


#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#define LOG(severity) (std::cerr << (#severity) << ": ")
#define RUNNING 1
#define STOPPED 0
#define ERROR -1
#define UDP_PORT 6789
#define BUF_SIZE 1024
#define LINE_SIZE 50
#define SENDING_INTERVAL 1 // seconds

struct Client {
    struct sockaddr_in addr;
    socklen_t addr_len;
};

std::vector<Client> clients;

void add_client(struct sockaddr_in* client_addr, socklen_t addr_len) {
    for (const auto& client : clients) {
        if (client.addr.sin_addr.s_addr == client_addr->sin_addr.s_addr &&
            client.addr.sin_port == client_addr->sin_port) {
            // Client already in list
            return;
        }
    }

    Client new_client;
    new_client.addr = *client_addr;
    new_client.addr_len = addr_len;
    clients.push_back(new_client);
}



void send_data_to_clients(int sockfd, std::vector<std::vector<float>> message, cv::Mat frame) {
    int lines =message.size(); 
    if (lines <= 0) return;
    std::string out;
    for (int l = 0; l < lines ; l++) {
      char line [50];
      sprintf(line, "[%f, %f, %f, %f, %f, %f]\n", message[l][0], 
                                                 message[l][1],
                                                 message[l][2],
                                                 message[l][3],
                                                 message[l][4],
                                                 message[l][5]); 
      out += line; 
    } 

    int msg_size = out.size();
    for (const auto& client : clients) {
        sendto(sockfd, out.c_str(), msg_size, 0,
               (struct sockaddr*)&client.addr, client.addr_len);
    }
}

namespace tflite {
namespace label_image {

// Default for SSD_MOBILENET_300x300
int argWidth = 320; 
int argHeight = 320; 
int CHANNELS = 3;
const std::string URL = "rtsp://bolt:bolt@192.168.10.90/axis-media/media.amp";




double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;


class RTSP_BOY {
public:
    cv::VideoCapture cap;
    std::string url;
    int status;
    std::string err_msg;

    RTSP_BOY(std::string url = URL) : url(url), status(RUNNING) {
        open_stream();
    }

    int open_stream() {
        int tries = 30;
        cap.open(url);
        
        if (!cap.isOpened()) {
            status = ERROR;
            err_msg = "Could not open rtsp stream: " + url;
            return ERROR;
        }

        double last_frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
        cap.set(cv::CAP_PROP_POS_FRAMES, last_frame_num);

        status = RUNNING;
        return RUNNING;
    }

    cv::Mat get_last_frame() {
        cv::Mat frame;
        bool ret = cap.read(frame);

        if (!ret) {
            status = ERROR;
            err_msg = "Could not read frame from " + url;
        }
        return frame;
    }

    void close_stream() {
        cap.release();
    }
};
void PrintTensorInfo(const TfLiteTensor* tensor) {
    // Print the tensor name
    if (tensor->name) {
        std::cout << "Tensor Name: " << tensor->name << std::endl;
    } else {
        std::cout << "Tensor Name: (unnamed)" << std::endl;
    }

    // Print the shape of the tensor
    std::cout << "Tensor Shape: [";
    for (int i = 0; i < tensor->dims->size; ++i) {
        std::cout << tensor->dims->data[i];
        if (i < tensor->dims->size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

class DelegateProviders {
 public:
  DelegateProviders()
      : delegates_list_(tflite::tools::GetRegisteredDelegateProviders()) {
    for (const auto& delegate : delegates_list_) {
      params_.Merge(delegate->DefaultParams());
    }
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int* argc, const char** argv) {
    std::vector<tflite::Flag> flags;
    for (const auto& delegate : delegates_list_) {
      auto delegate_flags = delegate->CreateFlags(&params_);
      flags.insert(flags.end(), delegate_flags.begin(), delegate_flags.end());
    }

    const bool parse_result = Flags::Parse(argc, argv, flags);
    if (!parse_result) {
      std::string usage = Flags::Usage(argv[0], flags);
      LOG(ERROR) << usage;
    }
    return parse_result;
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  TfLiteDelegatePtrMap CreateAllDelegates() const {
    TfLiteDelegatePtrMap delegates_map;
    for (const auto& delegate : delegates_list_) {
      auto ptr = delegate->CreateTfLiteDelegate(params_);
      // It's possible that a delegate of certain type won't be created as
      // user-specified benchmark params tells not to.
      if (ptr == nullptr) continue;
      LOG(INFO) << delegate->GetName() << " delegate created.\n";
      delegates_map.emplace(delegate->GetName(), std::move(ptr));
    }
    return delegates_map;
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  const tflite::tools::DelegateProviderList& delegates_list_;
};

TfLiteDelegatePtr CreateGPUDelegate(Settings* s) {
#if defined(__ANDROID__)
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.inference_priority1 =
      s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                    : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  return evaluation::CreateGPUDelegate(&gpu_opts);
#else
  return evaluation::CreateGPUDelegate();
#endif
}

TfLiteDelegatePtrMap GetDelegates(Settings* s,
                                  const DelegateProviders& delegate_providers) {
  // TODO(b/169681115): deprecate delegate creation path based on "Settings" by
  // mapping settings to DelegateProvider's parameters.
  TfLiteDelegatePtrMap delegates;
  if (s->gl_backend) {
    auto delegate = CreateGPUDelegate(s);
    if (!delegate) {
      LOG(INFO) << "GPU acceleration is unsupported on this platform.\n";
    } else {
      delegates.emplace("GPU", std::move(delegate));
    }
  }

  if (s->accel) {
    StatefulNnApiDelegate::Options options;
    options.allow_fp16 = s->allow_fp16;
    auto delegate = evaluation::CreateNNAPIDelegate(options);
    if (!delegate) {
      LOG(INFO) << "NNAPI acceleration is unsupported on this platform.\n";
    } else {
      delegates.emplace("NNAPI", std::move(delegate));
    }
  }

  if (s->hexagon_delegate) {
    const std::string libhexagon_path("/data/local/tmp");
    auto delegate =
        evaluation::CreateHexagonDelegate(libhexagon_path, s->profiling);

    if (!delegate) {
      LOG(INFO) << "Hexagon acceleration is unsupported on this platform.\n";
    } else {
      delegates.emplace("Hexagon", std::move(delegate));
    }
  }

  if (s->xnnpack_delegate) {
    auto delegate = evaluation::CreateXNNPACKDelegate(s->number_of_threads);
    if (!delegate) {
      LOG(INFO) << "XNNPACK acceleration is unsupported on this platform.\n";
    } else {
      delegates.emplace("XNNPACK", std::move(delegate));
    }
  }

  // Independent of above delegate creation options that are specific to this
  // binary, we use delegate providers to create TFLite delegates. Delegate
  // providers have been used in TFLite benchmark/evaluation tools and testing
  // so that we have a single and more comprehensive set of command line
  // arguments for delegate creation.
  TfLiteDelegatePtrMap delegates_from_providers =
      delegate_providers.CreateAllDelegates();
  for (auto& name_and_delegate : delegates_from_providers) {
    delegates.emplace("Delegate_Provider_" + name_and_delegate.first,
                      std::move(name_and_delegate.second));
  }

  return delegates;
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(ERROR) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symbolic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Subgraph " << std::setw(3) << std::setprecision(3)
            << subgraph_index << ", Node " << std::setw(3)
            << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
            << std::setprecision(3) << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}
cv::Mat resizeAndReshape(cv::Mat src, int width, int height, int dtype) {
    cv::Mat resized, reshaped, converted;
    
    // Resize the source image to the specified width and height
    cv::resize(src, resized, cv::Size(width, height));
    
    // Convert the resized image to the specified data type
    switch(dtype) {
        case kTfLiteFloat32: // float
            resized.convertTo(converted, CV_32F, 1.0 / 255, 0);
            break;
        case kTfLiteInt8: // int8
            resized.convertTo(converted, CV_8S);
            break;
        case kTfLiteUInt8: // uint
            resized.convertTo(converted, CV_8U);
            break;
        default:
            converted = resized; // If the dtype is not recognized, do not convert
            break;
    }
    
    // Reshape the converted image to 4D for deep learning model input
    reshaped = converted.reshape(1, std::vector<int>{1, height, width, 3});
    // LOG(INFO) << "IMAGE CONVERTED!\n";
    return reshaped;
}


std::string decodeOutputType(int i){
  if (i == 1) {
    return "Float 32";
  } 
  if (i == 2) {
    return "int 8";
  }
  if (i == 3) {
    return "uint 8";
  }
  return "Unkown";
}
void RunInference(Settings* settings, const DelegateProviders& delegate_providers) {
  
  /* ---------------------------------------------------------
                            Load the model 
     --------------------------------------------------------- */
   if (!settings->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }
  std::unique_ptr<tflite::Interpreter> interpreter;

  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(settings->model_name.c_str());
  if (!model) {
    LOG(ERROR) << "\nFailed to mmap model " << settings->model_name << "\n";
    exit(-1);
  }
  settings->model = model.get();
  LOG(INFO) << "Loaded model " << settings->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(ERROR) << "Failed to construct interpreter\n";
    exit(-1);
  }

  // interpreter->SetAllowFp16PrecisionForFp32(settings->allow_fp16);

  if (settings->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";
  }

  if (settings->number_of_threads != -1) {
    interpreter->SetNumThreads(settings->number_of_threads);
  }

  /* ---------------------------------------------------------
                            Apply NNAPI deleget
                            (Run on NPU) 
     --------------------------------------------------------- */
  
  auto delegates_ = GetDelegates(settings, delegate_providers);
  for (const auto& delegate : delegates_) {
    if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
        kTfLiteOk) {
      LOG(ERROR) << "Failed to apply " << delegate.first << " delegate.\n";
      exit(-1);
    } else {
      LOG(INFO) << "Applied " << delegate.first << " delegate.\n";
    }
  }



  /* ---------------------------------------------------------
                            Get Info About the Model 
     --------------------------------------------------------- */

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!\n";
    exit(-1);
  }
  int input = interpreter->inputs()[0];
  if (settings->verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (settings->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";

    for (int i = 0; i < outputs.size(); i++) {
        int output = outputs[i];
        TfLiteIntArray* out_dims = interpreter->tensor(output)->dims;
        LOG(INFO) << "Output  [" << i << "]'s shape: " << " [" <<  out_dims->data[0] << " " << out_dims->data[1] << " " << out_dims->data[2] << " " << out_dims->data[3] << " ]" << " Type: " << decodeOutputType(interpreter->tensor(output)->type) << "\n"; 
    }
    for (int i = 0; i < inputs.size(); i++) {
        int input = inputs[i];
        TfLiteIntArray* in_dims = interpreter->tensor(input)->dims;
        LOG(INFO) << "Input  [" << i << "]'s shape: " << " [" <<  in_dims->data[0] << " " << in_dims->data[1] << " " << in_dims->data[2] << " " << in_dims->data[3] << " ]" << " Type: " << decodeOutputType(interpreter->tensor(input)->type) << "\n" ; 
    }
  }
  
  /* ---------------------------------------------------------
                            Start Video 
     --------------------------------------------------------- */
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);

  RTSP_BOY boy(URL);

  if (boy.status != RUNNING) {
      LOG(ERROR) << boy.err_msg << std::endl;
  } else {
    LOG(INFO) << "Video from " << URL << " is estab.\n";
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "RTSP_BOY init time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms \n";

  
  /* ---------------------------------------------------------
                          Setup the UDP server 
    --------------------------------------------------------- */
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    char buffer[BUF_SIZE];
    socklen_t client_addr_len = sizeof(client_addr);

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    // Fill server information
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(UDP_PORT);

    // Bind the socket with the server address
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    LOG(INFO) << "UDP server is running on port "<< UDP_PORT <<" and waiting for clients..." << std::endl;

/* ---------------------------------------------------------
                            Detection Loop 
    --------------------------------------------------------- */
  while (true) {

/* ---------------------------------------------------------
                            Listen for new clients
                            (ideally you will be using threads..) 
  --------------------------------------------------------- */

  // Receive message from clients (this is just to register clients)
    int n = recvfrom(sockfd, buffer, BUF_SIZE, MSG_DONTWAIT, (struct sockaddr*)&client_addr, &client_addr_len);
    if (n > 0) {
        buffer[n] = '\0';
        LOG(INFO) << "Received message from client: " << buffer << std::endl;
        add_client(&client_addr, client_addr_len);
    }


  /* ---------------------------------------------------------
                            Get a Frame 
    --------------------------------------------------------- */
    gettimeofday(&start_time, nullptr);
    cv::Mat frame = boy.get_last_frame();
    if (boy.status != RUNNING) {
          LOG(ERROR) << boy.err_msg << std::endl;
    }
    gettimeofday(&stop_time, nullptr);
    // LOG(INFO) << "boy.get_last_frame() avg time: " << (get_us(stop_time) - get_us(start_time)) / (1000)  << " ms \n";


  /* ---------------------------------------------------------
                            Resize the Frame 
    --------------------------------------------------------- */
   

    cv::Mat in = resizeAndReshape(frame, argWidth, argHeight,settings->input_type );


    
    /* ---------------------------------------------------------
                              Copy the data to input_Layer 
      --------------------------------------------------------- */
    gettimeofday(&start_time, nullptr);
    settings->input_type = interpreter->tensor(input)->type;
    int totalSizeInput = argWidth * argHeight * 3;
    switch (settings->input_type) {
      case kTfLiteFloat32:
        LOG(INFO) << "Input type: kTfLiteFloat32\n";
        for (int i = 0; i < totalSizeInput; i++ ) {
          interpreter->typed_input_tensor<float>(input)[i] = in.data[i];
        }
        break;
      case kTfLiteInt8:
        for (int i = 0; i < totalSizeInput; i++ ) {
          interpreter->typed_input_tensor<int8_t>(input)[i] = in.at<int8_t>(i);
        }
        break;
      case kTfLiteUInt8:
        for (int i = 0; i < totalSizeInput; i++ ) {
          interpreter->typed_input_tensor<uint8_t>(input)[i] = in.at<uint8_t>(i);
        }
        break;
      default:
        LOG(ERROR) << "cannot handle input type "
                  << interpreter->tensor(input)->type << " yet\n";
        exit(-1);
    }
    gettimeofday(&stop_time, nullptr);
    // LOG(INFO) << "copy data to input layer time: " << (get_us(stop_time) - get_us(start_time)) / (1000)  << " ms \n";



    /* ---------------------------------------------------------
                              Make a detection 
      --------------------------------------------------------- */
    gettimeofday(&start_time, nullptr);
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(ERROR) << "Failed to invoke tflite!\n";
      exit(-1);
    }
    gettimeofday(&stop_time, nullptr);
    // LOG(INFO) << "Invoke time: " << (get_us(stop_time) - get_us(start_time)) / (1000)  << " ms \n";


    /* ---------------------------------------------------------
                              Get the output 
      --------------------------------------------------------- */

    const float threshold = 0.001f;

    int outputSize = outputs.size(); 

    // Total number of detected objects StatefulPartitionedCall:0
    TfLiteTensor* num_detec = interpreter->tensor(interpreter->outputs()[2]);
    // PrintTensorInfo(num_detec);

    // Confidence of detected objects StatefulPartitionedCall:1
    TfLiteTensor* scores    = interpreter->tensor(interpreter->outputs()[0]);
    // PrintTensorInfo(scores);

    // Class index of detected objects StatefulPartitionedCall:2
    TfLiteTensor* classes   = interpreter->tensor(interpreter->outputs()[3]);
    // PrintTensorInfo(classes);

    // Bounding box coordinates of detected objects StatefulPartitionedCall:3
    TfLiteTensor* bboxes    = interpreter->tensor(interpreter->outputs()[1]);
    // PrintTensorInfo(bboxes);


    auto          bboxes_   = bboxes->data.f;
    auto          classes_  = classes->data.f;
    auto          scores_   = scores->data.f;
    
    auto bboxes_xywh        = bboxes->dims->data[bboxes->dims->size - 1]; 
    auto classes_size       = classes->dims->data[classes->dims->size - 1];
    auto scores_size        = scores->dims->data[scores->dims->size - 1];

    
    
    if (bboxes_xywh != 4){
        std::cerr << "Incorrect bbox size: " << bboxes_xywh << std::endl;
        exit(0);
    }
    if (classes_size != scores_size){
        std::cerr << "Number of classes and scores does not match: " << classes_size << " " << scores_size << std::endl;
        exit(0);
    }

    std::vector<float> locations;
    std::vector<float> cls;

    for (int i = 0; i < bboxes_xywh * classes_size; i++){
        locations.push_back(bboxes_[i]);
    }

    for (int i = 0; i < classes_size; i++){
        cls.push_back(classes_[i]);
    }


    std::vector<std::vector<float>> detections;
    for(int j = 0; j <locations.size(); j+=4){
        float score = scores_[j];
        if (score < 0.1 || score > 1)
         continue;

        float cls = classes_[j];

        auto top=locations[j]  *argHeight;
        auto left=locations[j+1]*argWidth;
        auto down=locations[j+2]*argHeight;
        auto right=locations[j+3]*argWidth;
        std::vector<float> det {cls, score, top, left, down, right};
        
        detections.push_back(det);
    }
  /* ---------------------------------------------------------
                            send the output to clients 
  --------------------------------------------------------- */
    send_data_to_clients(sockfd, detections, in);
    
    
  }
    boy.close_stream();
    close(sockfd);

}

void display_usage() {
  LOG(INFO)
      << "label_image\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--gl_backend, -g: [0|1]: use GL GPU Delegate on Android\n"
      << "--hexagon_delegate, -j: [0|1]: use Hexagon Delegate on Android\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--xnnpack_delegate, -x [0:1]: xnnpack delegate\n"
      << "--argWidth, -o \n"
      << "--argHeight, -n \n"
      << "\n";
}

int Main(int argc, char** argv) {
  DelegateProviders delegate_providers;
  bool parse_result = delegate_providers.InitFromCmdlineArgs(
      &argc, const_cast<const char**>(argv));
  if (!parse_result) {
    return EXIT_FAILURE;
  }

  Settings s;

  int c;
  while (true) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {"hexagon_delegate", required_argument, nullptr, 'j'},
        {"xnnpack_delegate", required_argument, nullptr, 'x'},
        {"argWidth", required_argument, nullptr, 'o'},
        {"argHeight", required_argument, nullptr, 'n'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:d:e:f:g:i:j:l:m:p:r:s:t:v:w:x:o:n", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'e':
        s.max_profiling_buffer_entries =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'g':
        s.gl_backend =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'j':
        s.hexagon_delegate = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'x':
        s.xnnpack_delegate =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'o':
        argHeight = strtol(optarg, nullptr, 10);
        break;
      case 'n':
        argWidth = strtol(optarg, nullptr, 10);
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
   }
 }
  
  RunInference(&s, delegate_providers);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image::Main(argc, argv);
}
