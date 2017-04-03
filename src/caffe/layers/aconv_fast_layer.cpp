#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/aconv_fast_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void AConvFastLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	AConvParameter aconv_param = this->layer_param_.aconv_param();

	CHECK_EQ(conv_param.dilation_size(), 0) << "Dilation parameter is not supported";
	CHECK_EQ(this->group_, 1) << "Group>1 is not supported yet";
	CHECK_EQ(this->channel_axis_, 1) << "Channel axis should be 1";
	//CHECK_EQ(this->stride_.cpu_data()[0], 1) << "Stride>1 is not supported yet";
	//CHECK_EQ(this->stride_.cpu_data()[1], 1) << "Stride>1 is not supported yet";

	base_radius_ = aconv_param.base_radius();
	CHECK_GT(base_radius_, 0);

	base_angle_ = aconv_param.base_angle();
	normalize_diff_ = aconv_param.normalize();

	//initialize pos group
	this->pos_group_ = aconv_param.position_group();

	CHECK_EQ(this->num_output_%this->pos_group_, 0);
	pos_group_offset_ = this->num_output_/this->pos_group_;

	// Handle the parameters: weights and biases.
	// - blobs_[0] holds the filter weights
	// - blobs_[1] holds the biases (optional)
	// - blobs_[2] holds the filter x position
	// - blobs_[3] holds the filter y position
	this->blobs_.resize(4);

	// Setup filter position
	weight_w_ = this->blobs_[0]->width();
	weight_h_ = this->blobs_[0]->height();
	kernel_n_ = weight_w_*weight_h_;
	pos_count_ = this->pos_group_*kernel_n_;

	// Setup for position backprob
	this->blobs_[2].reset(new Blob<Dtype>(this->pos_group_,1,weight_h_,weight_w_));
	this->blobs_[3].reset(new Blob<Dtype>(this->pos_group_,1,weight_h_,weight_w_));
	pos_diff_buffer_.Reshape(this->blobs_[0]->num(), 2*this->blobs_[0]->channels(),weight_h_,weight_w_);

	vector<int> pos_diff_multiplier_shape(1, std::max(this->num_output_, this->channels_));
	pos_diff_multiplier_.Reshape(pos_diff_multiplier_shape);
	caffe_set(pos_diff_multiplier_.count(), Dtype(1), pos_diff_multiplier_.mutable_cpu_data());

	vector<int> pos_diff_temp_shape(1, this->channels_*kernel_n_*2);
	pos_diff_temp.Reshape(pos_diff_temp_shape);

	// Initialize filter position
	Dtype *xpos_data = this->blobs_[2]->mutable_cpu_data();
	Dtype *ypos_data = this->blobs_[3]->mutable_cpu_data();
	Dtype pos_filler_std = aconv_param.pos_filler_std()*base_radius_;

	if(pos_filler_std>0)
	{
		FillerParameter filler_param;
		filler_param.set_mean(0);
		filler_param.set_std(pos_filler_std);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blobs_[2].get());
		filler.Fill(this->blobs_[3].get());
	}

	const Dtype unit_angle = 2.0 * M_PI / (kernel_n_-1);

	for(int g=0;g<this->pos_group_;g++)
	{
		const int idx = g*kernel_n_;

#ifdef TEST_ACONV_FAST_ENV
			//fixed center
			xpos_data[idx] = 0.2 + 0.1*g;
			ypos_data[idx] = 0.2 + 0.1*g;

			for(int i=1;i<kernel_n_;i++)
			{
				xpos_data[idx+i] = nearbyint(base_radius_ * cos((i-1)*unit_angle))+0.2 + 0.1*g;
				ypos_data[idx+i] = nearbyint(base_radius_ * sin((i-1)*unit_angle))+0.2 + 0.1*g;
			}
#else
		//fixed center
		xpos_data[idx] = 0;
		ypos_data[idx] = 0;

		for(int i=1;i<kernel_n_;i++)
		{
			if(pos_filler_std < 0)
			{
				xpos_data[idx+i] = nearbyint(base_radius_ * cos((i-1)*unit_angle+base_angle_));
				ypos_data[idx+i] = nearbyint(base_radius_ * sin((i-1)*unit_angle+base_angle_));
			}
			else
			{
				xpos_data[idx+i] += base_radius_ * cos((i-1)*unit_angle+base_angle_);
				ypos_data[idx+i] += base_radius_ * sin((i-1)*unit_angle+base_angle_);
			}
		}
#endif
	}

	// initialize padding(aconv padding has different meaning with original padding)
	// padding can be a negative value
	int* pad_data = this->pad_.mutable_cpu_data();
	if(aconv_param.type()==AConvParameter_FilterType_SQUARE)
	{
		if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
			CHECK(false) << "kernel_h/w is not valid for SQUARE type kernel";
		}

		pad_data[0] -= weight_h_/2;
		pad_data[1] -= weight_w_/2;

		if (aconv_param.has_pad() || aconv_param.has_pad_h() || aconv_param.has_pad_w())
		{
			CHECK(false) << "aconv.pad is only valid for non-SQUARE type kernel";
		}
	}
	else
	{
		if (conv_param.has_pad_h() || conv_param.has_pad_w() || conv_param.pad_size()>0)
		{
			CHECK(false) << "Use aconv.pad instead of conv.pad when you use non-SQUARE type kernel";
		}

		pad_data[0] = aconv_param.pad();
		pad_data[1] = aconv_param.pad();

		if (aconv_param.has_pad_h() || aconv_param.has_pad_w()) {
			pad_data[0] = aconv_param.pad_h();
			pad_data[1] = aconv_param.pad_w();
		}
	}

	//setup
	const int* stride_data = this->stride_.cpu_data();
	pad_h_ = pad_data[0];
	pad_w_ = pad_data[1];
    stride_h_ = stride_data[0];
    stride_w_ = stride_data[1];

    //
    bottom_height_ = bottom[0]->height();
    bottom_width_ = bottom[0]->width();
    top_height_ = (bottom[0]->height() + 2 * pad_h_ - 1)/ stride_h_ + 1;
    top_width_ = (bottom[0]->width() + 2 * pad_w_ - 1)/ stride_w_ + 1;

    //
    wgroup_offset_ = this->blobs_[0]->count()/this->pos_group_;

	//layer setup
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
	convolution_param->add_kernel_size(1);
	convolution_param->set_bias_term(false);
	convolution_param->set_num_output(this->num_output_/this->pos_group_);
	conv_layer_.reset(new CuDNNConvolutionLayer<Dtype>(layer_param));

	conv_bottom_.Reshape(bottom[0]->num(),this->channels_,top_height_,top_width_);
	conv_top_.Reshape(bottom[0]->num(),this->num_output_/this->pos_group_,top_height_,top_width_);

	conv_bottom_vec_.clear();
	conv_bottom_vec_.push_back(&conv_bottom_);

	conv_top_vec_.clear();
	conv_top_vec_.push_back(&conv_top_);
	conv_layer_->SetUp(conv_bottom_vec_, conv_top_vec_);
	conv_layer_->setDeterministic(0);

	cwCount_ = conv_layer_->blobs()[0]->count();

	// Propagate gradients to the parameters (as directed by backward pass).
	this->param_propagate_down_.resize(this->blobs_.size(), true);

	if (!this->bias_term_) {
	  	this->blobs_[1].reset(new Blob<Dtype>(1,1,1,1));

	  	FillerParameter filler_param;
		filler_param.set_value(0);
		ConstantFiller<Dtype> bias_filler(filler_param);
	    bias_filler.Fill(this->blobs_[1].get());

		this->param_propagate_down_[1]=false;
	}
}


template <typename Dtype>
void AConvFastLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
	this->pos_col_buffer_.Reshape(this->num_,2*this->col_buffer_shape_[0],this->col_buffer_shape_[1],this->col_buffer_shape_[2]);
	fliptop_.ReshapeLike(*top[0]);

	conv_layer_->Reshape(conv_bottom_vec_, conv_top_vec_);
}

template <typename Dtype>
void AConvFastLayer<Dtype>::compute_output_shape() {
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - 1)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void AConvFastLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//NOT_IMPLEMENTED;
}

template <typename Dtype>
void AConvFastLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(AConvFastLayer);
#endif

INSTANTIATE_CLASS(AConvFastLayer);
REGISTER_LAYER_CLASS(AConvFast);

}  // namespace caffe
