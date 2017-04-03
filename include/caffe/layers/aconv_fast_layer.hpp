#ifndef CAFFE_ACONV_FAST_LAYER_HPP_
#define CAFFE_ACONV_FAST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class AConvFastLayer: public BaseConvolutionLayer<Dtype> {
 public:
  explicit AConvFastLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactBottomBlobs() const { return 1; }
  virtual inline int ExactTopBlobs() const { return 1; }

  virtual inline const char* type() const { return "AConvFast"; }
  
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  Blob<Dtype> pos_diff_buffer_;
  Blob<Dtype> pos_diff_temp;	//temporary storage for pos diff sum
  Blob<Dtype> pos_diff_multiplier_;

  int pos_group_offset_; //output channel offset for one pos-group

  int weight_w_;
  int weight_h_;
  int kernel_n_;
  int pos_count_;

  int pad_w_;
  int pad_h_;
  int stride_w_;
  int stride_h_;

  int bottom_width_;
  int bottom_height_;
  int top_width_;
  int top_height_;

  Dtype base_radius_;
  Dtype base_angle_;

  //Regularization
  bool normalize_diff_;

  //
  vector<Blob<Dtype>*> conv_top_vec_;
  vector<Blob<Dtype>*> conv_bottom_vec_;
  Blob<Dtype> conv_top_;
  Blob<Dtype> conv_bottom_;

  Blob<Dtype> fliptop_;

  shared_ptr<CuDNNConvolutionLayer<Dtype> > conv_layer_;

  int wgroup_offset_;
  int cwCount_;
};


}  // namespace caffe

//#define TEST_ACONV_FAST_ENV

#endif  // CAFFE_ACONV_FAST_LAYER_HPP_
