#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/aconv_fast_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class AConvFastLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  AConvFastLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 2, 4, 3)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~AConvFastLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(AConvFastLayerTest, GPUDevice<float>);

TYPED_TEST(AConvFastLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(2);
  //convolution_param->add_pad(1);
  //convolution_param->set_pad_h(2);
  //convolution_param->set_pad_w(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  AConvFastLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(AConvFastLayerTest, TestGradient3x3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();

  AConvParameter* aconv_param = layer_param.mutable_aconv_param();
  aconv_param->set_position_group(2);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(3);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  AConvFastLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(AConvFastLayerTest, TestGradientVariableFilter) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
  AConvParameter* aconv_param = layer_param.mutable_aconv_param();
  aconv_param->set_position_group(1);

  convolution_param->set_kernel_w(3);
  convolution_param->set_kernel_h(2);
  convolution_param->add_stride(1);
  aconv_param->set_type(AConvParameter_FilterType_RANDOM);
  aconv_param->set_pad_w(1);
  aconv_param->set_pad_h(-1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  AConvFastLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
}  // namespace caffe
