// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "asserts.h"
#include <iterator>
#include "gtest/gtest.h"
#include <core/platform/path_lib.h>
#include <test/onnx/OrtValueList.h>
#include "test/onnx/TestCase.h"
#include "test/onnx/runner.h"
#include "test/compare_ortvalue.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
class ModelTest : public testing::TestWithParam<::std::basic_string<ORTCHAR_T>> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
};

TEST_P(ModelTest, Run) {
  ::std::basic_string<ORTCHAR_T> param = GetParam();
  size_t pos = param.find(ORT_TSTR("_"));
  ASSERT_NE(pos, std::string::npos);
  std::string provider_name = ToMBString(param.substr(0, pos));
  std::basic_string<ORTCHAR_T> model_dir = param.substr(pos + 1);
  double per_sample_tolerance = 1e-3;
  // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
  // when openvino is enabled, set it to a larger value for resolving MNIST accuracy mismatch
  double relative_per_sample_tolerance = 1e-3;
  if(provider_name == "cuda"){
      relative_per_sample_tolerance = 0.017;
  } else if(provider_name == "openvino"){
      relative_per_sample_tolerance = 0.009;
  }

  std::unique_ptr<TestModelInfo> model_info(TestModelInfo::LoadOnnxModel(model_dir.c_str()));
  //TODO: filter model based on opset
  std::basic_string<ORTCHAR_T> my_dir_name = GetLastComponent(model_dir);
  std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);
  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToMBString(test_case_name), std::move(model_info),
                                                       per_sample_tolerance,
                                                    relative_per_sample_tolerance);
  SessionOptions so;
  InferenceSession session_object(so,(**ort_env).GetEnvironment());
  //TODO: register provider
  ASSERT_STATUS_OK(session_object.Load(model_dir));
  ASSERT_STATUS_OK(session_object.Initialize());
  const size_t data_count = l->GetDataCount();
  for (size_t task_id = 0; task_id != data_count; ++task_id) {
      onnxruntime::test::HeapBuffer holder;
      std::unordered_map<std::string, OrtValue*> feeds;
      l->LoadTestData(task_id, holder, feeds, true);

      std::pair<common::Status, const OutputDefList*> output_meta_data = session_object.GetModelOutputs();
      ASSERT_STATUS_OK(output_meta_data.first);
      // Create output feed
      size_t output_count = output_meta_data.second->size();
      std::vector<std::string> output_names(output_count);
      for (size_t i = 0; i != output_count; ++i) {
          output_names[i] = (*output_meta_data.second)[i]->Name();
      }
      if (feeds.size() > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
          ORT_THROW("length overflow");
      }
      std::vector<const char*> input_names(feeds.size());
      OrtValueArray input_values(static_cast<int>(feeds.size()));
      size_t input_index = 0;
      for (auto& kvp : feeds) {
          input_names[input_index] = kvp.first.c_str();
          input_values.Set(input_index, kvp.second);
          ++input_index;
      }

      OrtValueArray output_values(static_cast<int>(output_count));
      {
          std::vector<const char*> output_names_raw_ptr(output_count);
          for (size_t i = 0; i != output_count; ++i) {
              output_names_raw_ptr[i] = output_names[i].c_str();
          }
          Ort::ThrowOnError(Ort::GetApi().Run((OrtSession*)&session_object, nullptr, input_names.data(), input_values.Data(),
                                              static_cast<size_t>(input_values.Length()), output_names_raw_ptr.data(),
                                              output_count, output_values.Data()));
      }

      double per_sample_tolerance;
      double relative_per_sample_tolerance;
      bool post_procesing;
      Status status;
      ASSERT_STATUS_OK(l->GetPerSampleTolerance(&per_sample_tolerance));
      ASSERT_STATUS_OK(l->GetRelativePerSampleTolerance(&relative_per_sample_tolerance));
      ASSERT_STATUS_OK(l->GetPostProcessing(&post_procesing));

      //TODO: if there are no output value files, just skip the validation
      std::unordered_map<std::string, OrtValue*> expected_output_values;
      l->LoadTestData(task_id, holder, expected_output_values, false);

      std::unordered_map<std::string, OrtValue*> name_fetch_output_map;
      std::unordered_map<std::string, const ONNX_NAMESPACE::ValueInfoProto*> name_output_value_info_proto;
      size_t i = 0;
      for (auto& output_name : output_names) {
          // p_fetches is filled in the order of output_names.
          name_fetch_output_map[output_name] = output_values.Get(i);
          const ONNX_NAMESPACE::ValueInfoProto* infoProto = l->GetOutputInfoFromModel(i);
          if (infoProto != nullptr) name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
          i++;
      }

      EXECUTE_RESULT res = EXECUTE_RESULT::SUCCESS;
      for (auto& output : expected_output_values) {
          OrtValue* expected_output_value = output.second;
          const std::string& output_name = output.first;
          auto iter = name_fetch_output_map.find(output_name);
          if (iter == name_fetch_output_map.end()) {
              res = EXECUTE_RESULT::INVALID_GRAPH;
              LOGF_DEFAULT(ERROR, "cannot find %s in the outputs", output_name.c_str());
              break;
          }
          OrtValue* actual_output_value = iter->second;
          std::pair<COMPARE_RESULT, std::string> ret =
                  CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                                  relative_per_sample_tolerance, post_procesing);
          COMPARE_RESULT compare_result = ret.first;
          if (compare_result == COMPARE_RESULT::SUCCESS) {
              const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
              if (v == nullptr) continue;
              ret = VerifyValueInfo(*v, Ort::Unowned<Ort::Value>{actual_output_value});
              compare_result = ret.first;
              if (compare_result != COMPARE_RESULT::SUCCESS) {
                  switch (compare_result) {
                      case COMPARE_RESULT::NOT_SUPPORT:
                          res = EXECUTE_RESULT::NOT_SUPPORT;
                          break;
                      case COMPARE_RESULT::SHAPE_MISMATCH:
                          res = EXECUTE_RESULT::MODEL_SHAPE_MISMATCH;
                          break;
                      case COMPARE_RESULT::TYPE_MISMATCH:
                          res = EXECUTE_RESULT::MODEL_TYPE_MISMATCH;
                          break;
                      default:
                          res = EXECUTE_RESULT::UNKNOWN_ERROR;
                  }
              }
          } else {
              switch (compare_result) {
                  case COMPARE_RESULT::NOT_SUPPORT:
                      res = EXECUTE_RESULT::NOT_SUPPORT;
                      break;
                  case COMPARE_RESULT::RESULT_DIFFERS:
                      res = EXECUTE_RESULT::RESULT_DIFFERS;
                      break;
                  case COMPARE_RESULT::SHAPE_MISMATCH:
                      res = EXECUTE_RESULT::SHAPE_MISMATCH;
                      break;
                  case COMPARE_RESULT::TYPE_MISMATCH:
                      res = EXECUTE_RESULT::TYPE_MISMATCH;
                      break;
                  default:
                      res = EXECUTE_RESULT::UNKNOWN_ERROR;
              }
          }
          ASSERT_EQ(compare_result, COMPARE_RESULT::SUCCESS) << test_case_name << ":output=" << output_name << ":" << ret.second;

          if (compare_result != COMPARE_RESULT::SUCCESS) {
              break;
          }
      }
      for (auto& kvp : expected_output_values) {
          Ort::GetApi().ReleaseValue(kvp.second);
      }
  }
}

//TODO: all providers
::std::vector<::std::basic_string<ORTCHAR_T>> GetParameterStrings(const char* provider_name) {
  // Permanently exclude following tests because ORT support only opset staring from 7,
  // Please make no more changes to the list
  static const ORTCHAR_T* immutable_broken_tests[] =
      {
          ORT_TSTR("AvgPool1d"),
          ORT_TSTR("AvgPool1d_stride"),
          ORT_TSTR("AvgPool2d"),
          ORT_TSTR("AvgPool2d_stride"),
          ORT_TSTR("AvgPool3d"),
          ORT_TSTR("AvgPool3d_stride"),
          ORT_TSTR("AvgPool3d_stride1_pad0_gpu_input"),
          ORT_TSTR("BatchNorm1d_3d_input_eval"),
          ORT_TSTR("BatchNorm2d_eval"),
          ORT_TSTR("BatchNorm2d_momentum_eval"),
          ORT_TSTR("BatchNorm3d_eval"),
          ORT_TSTR("BatchNorm3d_momentum_eval"),
          ORT_TSTR("GLU"),
          ORT_TSTR("GLU_dim"),
          ORT_TSTR("Linear"),
          ORT_TSTR("PReLU_1d"),
          ORT_TSTR("PReLU_1d_multiparam"),
          ORT_TSTR("PReLU_2d"),
          ORT_TSTR("PReLU_2d_multiparam"),
          ORT_TSTR("PReLU_3d"),
          ORT_TSTR("PReLU_3d_multiparam"),
          ORT_TSTR("PoissonNLLLLoss_no_reduce"),
          ORT_TSTR("Softsign"),
          ORT_TSTR("operator_add_broadcast"),
          ORT_TSTR("operator_add_size1_broadcast"),
          ORT_TSTR("operator_add_size1_right_broadcast"),
          ORT_TSTR("operator_add_size1_singleton_broadcast"),
          ORT_TSTR("operator_addconstant"),
          ORT_TSTR("operator_addmm"),
          ORT_TSTR("operator_basic"),
          ORT_TSTR("operator_mm"),
          ORT_TSTR("operator_non_float_params"),
          ORT_TSTR("operator_params"),
          ORT_TSTR("operator_pow"),
      };

  static const ORTCHAR_T* cuda_flaky_tests[] = {
      ORT_TSTR("fp16_inception_v1"),
      ORT_TSTR("fp16_shufflenet"), ORT_TSTR("fp16_tiny_yolov2")};
  static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mlperf_ssd_mobilenet_300"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("zfnet512")};
  static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("test_densenet121"), ORT_TSTR("test_resnet18v2"), ORT_TSTR("test_resnet34v2"), ORT_TSTR("test_resnet50v2"), ORT_TSTR("test_resnet101v2"),
                                                   ORT_TSTR("test_resnet101v2"), ORT_TSTR("test_vgg19"), ORT_TSTR("tf_inception_resnet_v2"), ORT_TSTR("tf_inception_v1"), ORT_TSTR("tf_inception_v3"), ORT_TSTR("tf_inception_v4"), ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.0_224"), ORT_TSTR("tf_mobilenet_v2_1.4_224"), ORT_TSTR("tf_nasnet_large"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("tf_resnet_v1_50"), ORT_TSTR("tf_resnet_v1_101"), ORT_TSTR("tf_resnet_v1_101"),
                                                   ORT_TSTR("tf_resnet_v2_101"), ORT_TSTR("tf_resnet_v2_152"), ORT_TSTR("batchnorm_example_training_mode"), ORT_TSTR("batchnorm_epsilon_training_mode")};

  std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests), std::end(immutable_broken_tests));
  if (strcmp(provider_name, "cuda") == 0) {
    all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
  }
  if (strcmp(provider_name, "dml") == 0) {
    all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
  }
  if (strcmp(provider_name, "dnnl") == 0) {
    // these models run but disabled tests to keep memory utilization low
    // This will be removed after LRU implementation
    all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
  }
#if !defined(__amd64__) && !defined(_M_AMD64)
  //out of memory
  static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mask_rcnn_keras"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("vgg19"), ORT_TSTR("coreml_VGG16_ImageNet")};
  all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

  std::vector<::std::string> v;
  std::vector<std::basic_string<ORTCHAR_T>> paths;
#ifndef NDEBUG
  paths.push_back("/data/models");
#endif
  paths.push_back("/data/onnx");
  while (!paths.empty()) {
    std::basic_string<ORTCHAR_T> node_data_root_path = paths.back();
    paths.pop_back();
    std::basic_string<ORTCHAR_T> my_dir_name = GetLastComponent(node_data_root_path);
    try {
      LoopDir(node_data_root_path, [&](const ORTCHAR_T* filename, OrtFileType f_type) -> bool {
        if (filename[0] == '.') return true;
        if (f_type == OrtFileType::TYPE_DIR) {
          std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path,
                                                                                    filename);
          paths.push_back(p);
          return true;
        }
        std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
        if (!HasExtensionOf(filename_str, ORT_TSTR("onnx"))) return true;

        std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
        if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);
        if (all_disabled_tests.find(test_case_name) != all_disabled_tests.end()) return true;
        std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path,
                                                                                  filename_str);
        std::basic_string<PATH_CHAR_TYPE> r = provider_name;
        r.append("_").append(p);
        v.emplace_back(r);
        return true;
      });
    } catch (std::exception& ex) {
    }  //ignore non-exist dir
  }
  return v;
}

    static std::string MyParamName(const testing::TestParamInfo<::std::basic_string<ORTCHAR_T>>& info) {
        size_t pos = info.param.find(ORT_TSTR("_"));
        std::basic_string<ORTCHAR_T> model_path = info.param.substr(pos + 1);
        std::basic_string<ORTCHAR_T> my_dir_name;
        (void)GetDirNameFromFilePath(model_path,my_dir_name);
        std::basic_string<PATH_CHAR_TYPE> test_case_name = GetLastComponent(my_dir_name);
        if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);
        // Check for invalid characters
        bool is_valid_name = true;
        for (std::string::size_type index = 0; index < test_case_name.size(); ++index) {
            if (!isalnum(test_case_name[index]) && test_case_name[index] != '_') {
                is_valid_name = false;
                break;
            }
        }
        std::ostringstream  oss;
        oss<< info.index;
        if(is_valid_name) {
            oss<< "_";
            oss << test_case_name;
        }
        return oss.str();
    }

INSTANTIATE_TEST_SUITE_P(ModelTests,
                         ModelTest,
                         testing::ValuesIn(GetParameterStrings("cpu")),MyParamName);

}  // namespace test
}  // namespace onnxruntime
