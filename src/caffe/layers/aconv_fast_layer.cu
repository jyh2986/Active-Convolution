#include <vector>
#include <math.h>

#include "caffe/layers/aconv_fast_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void copyWeight(const int n, const int out_channels, const int in_channels, int kernel_n,
		const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
	  int c_out = index / in_channels;
	  int c_in = index % in_channels;

	  output[index] = input[(c_out*in_channels + c_in)*kernel_n];
  }
}

template <typename Dtype>
__global__ void copyDiff(const int n, const int out_channels, const int in_channels, int kernel_n,
		const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
	  int c_out = index / in_channels;
	  int c_in = index % in_channels;

	  output[(c_out*in_channels + c_in)*kernel_n] = input[index];
  }
}

template <typename Dtype>
__global__ void mergeOutput(const int n, const int out_channels, const int in_channels, int spatial_dim, int pg,
		const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
	  int num = index / (in_channels * spatial_dim);
	  int idx = index % (in_channels * spatial_dim);
	  int c = idx / spatial_dim;
	  int i = idx % spatial_dim;

	  output[num*out_channels*spatial_dim + (c+pg*in_channels)*spatial_dim + i] += input[index];
  }
}

template <typename Dtype>
__global__ void splitOutput(const int n, const int out_channels, const int in_channels, int spatial_dim, int pg,
		const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
	  int num = index / (in_channels * spatial_dim);
	  int idx = index % (in_channels * spatial_dim);
	  int c = idx / spatial_dim;
	  int i = idx % spatial_dim;

	  output[index] = input[num*out_channels*spatial_dim + (c+pg*in_channels)*spatial_dim + i];
  }
}

template <typename Dtype>
__global__ void multiplyWeight2(const int n, const int kernel_dim, const Dtype* weight,
		const int pos_group_offset_, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
	 int row = index/kernel_dim;
	 int col = index%kernel_dim;

	 Dtype w = weight[index];

	 int idx = 2*row*kernel_dim+col;
	 out[idx] *= w;
	 out[idx+kernel_dim] *= w;
  }
}

template <typename Dtype>
__global__ void applyConstraint2(const int n, int kernel_n,
		const Dtype* xpos_data, const Dtype* ypos_data, Dtype* xpos_diff, Dtype* ypos_diff, const bool normalize) {
  CUDA_KERNEL_LOOP(index, n) {
#ifndef TEST_ACONV_FAST_ENV
	  const int kernel_idx 	= index % kernel_n;	//index of kernel in each group

	  if(kernel_idx == 0){
		  xpos_diff[index] = 0;
	  	  ypos_diff[index] = 0;

	  	  return;
	  }

	  if(normalize) //normalize
	  {
		  const Dtype dx = xpos_diff[index];
		  const Dtype dy = ypos_diff[index];
		  const Dtype dr = sqrt(dx*dx+dy*dy);

		  if(dr!=0)
		  {
			  xpos_diff[index] = dx/dr;
			  ypos_diff[index] = dy/dr;
		  }
	  }
#endif
  }
}

template <typename Dtype>
__global__ void interpolated(const int nthreads, const Dtype* const bottom_data,
    const int num, const int channels, const int bottom_height, const int bottom_width,
    const Dtype* xpos, const Dtype* ypos, const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int top_height, const int top_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int out_spatial_dim = top_height * top_width;
    const int n = index / (channels * out_spatial_dim);
    const int idx = index % (channels * out_spatial_dim);
    const int c = idx / out_spatial_dim;
    const int idx2 = idx % out_spatial_dim;
    const int h = idx2 / top_width;
    const int w = idx2 % top_width;
    const int h_offset = h * stride_h - pad_h;
    const int w_offset = w * stride_w - pad_w;


    const Dtype shiftX = xpos[0];
    const Dtype shiftY = ypos[0];

    Dtype val = 0;

    if(shiftX==0 && shiftY==0)
	{
		const int offset = (n*channels + c) * (bottom_width*bottom_height);
		const Dtype *in =  bottom_data + offset;

		int x,y;

		y = h_offset;
		x = w_offset;
		Dtype q11 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

		val = q11;
	}
    else
    {
		int ix = floor((double)shiftX);
		int iy = floor((double)shiftY);
		Dtype dx = shiftX-ix;
		Dtype dy = shiftY-iy;

		if(dx==0 && dy==0)
		{
			const int offset = (n*channels + c) * (bottom_width*bottom_height);
			const Dtype *in =  bottom_data + offset;

			int x = w_offset + ix;
			int y = h_offset + iy;
			Dtype q11 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

			val = q11;
		}
		else
		{
			int x1 = w_offset + ix;
			int x2 = x1+1;
			int y1 = h_offset + iy;
			int y2 = y1+1;
			int x,y;

			const int offset = (n*channels + c) * (bottom_width*bottom_height);
			const Dtype *in =  bottom_data + offset;

			y = y1;
			x = x1;
			Dtype q11 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

			y = y1;
			x = x2;
			Dtype q21 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

			y = y2;
			x = x1;
			Dtype q12 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

			y = y2;
			x = x2;
			Dtype q22 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

			val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
		}
	}

	top_data[index] = val;
  }
}


template <typename Dtype>
__global__ void interpolatedBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int top_height, const int top_width,
    const Dtype* xpos, const Dtype* ypos, const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int bottom_height, const int bottom_width, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int out_spatial_dim = bottom_height * bottom_width;
    const int n = index / (channels * out_spatial_dim);
    const int idx = index % (channels * out_spatial_dim);
    const int c = idx / out_spatial_dim;
    const int idx2 = idx % out_spatial_dim;
    const int h = idx2 / bottom_width;
    const int w = idx2 % bottom_width;
    const int h_offset = h + pad_h;
   	const int w_offset = w + pad_w;

    const Dtype shiftX = -xpos[0];
    const Dtype shiftY = -ypos[0];

    Dtype val = 0;

	if(shiftX==0 && shiftY==0)
	{

		const int offset = (n*channels + c) * (top_width*top_height);
		const Dtype *in =  top_diff + offset;

		int x,y;

		Dtype q11 = 0;
		y = h_offset;
		x = w_offset;
		if(x%stride_w == 0 && y%stride_h == 0)
		{
			x=x/stride_w;
			y=y/stride_h;
			q11 = (y >= 0 && x >= 0 && y < top_height && x < top_width) ? in[y * top_width + x] : 0;
		}

		val = q11;
	}
	else
	{
		int ix = floor((double)shiftX);
		int iy = floor((double)shiftY);
		Dtype dx = shiftX-ix;
		Dtype dy = shiftY-iy;


		if(dx==0 && dy==0)
		{
			const int offset = (n*channels + c) * (top_width*top_height);
			const Dtype *in =  top_diff + offset;

			Dtype q11 = 0;

			int x = w_offset + ix;
			int y = h_offset + iy;

			if(x%stride_w == 0 && y%stride_h == 0)
			{
				x=x/stride_w;
				y=y/stride_h;
				q11 = (y >= 0 && x >= 0 && y < top_height && x < top_width) ? in[y * top_width + x] : 0;
			}

			val = q11;
		}
		else
		{

			int x1 = w_offset + ix;
			int x2 = x1+1;
			int y1 = h_offset + iy;
			int y2 = y1+1;
			int x,y;

			const int offset = (n*channels + c) * (top_width*top_height);
			const Dtype *in =  top_diff + offset;

			Dtype q11 = 0;
			y = y1;
			x = x1;
			if(x%stride_w == 0 && y%stride_h == 0)
			{
				x=x/stride_w;
				y=y/stride_h;
				q11 = (y >= 0 && x >= 0 && y < top_height && x < top_width) ? in[y * top_width + x] : 0;
			}

			Dtype q21 = 0;
			y = y1;
			x = x2;
			if(x%stride_w == 0 && y%stride_h == 0)
			{
				x=x/stride_w;
				y=y/stride_h;
				q21 = (y >= 0 && x >= 0 && y < top_height && x < top_width) ? in[y * top_width + x] : 0;
			}

			Dtype q12 = 0;
			y = y2;
			x = x1;
			if(x%stride_w == 0 && y%stride_h == 0)
			{
				x=x/stride_w;
				y=y/stride_h;
				q12 = (y >= 0 && x >= 0 && y < top_height && x < top_width) ? in[y * top_width + x] : 0;
			}

			Dtype q22 = 0;
			y = y2;
			x = x2;
			if(x%stride_w == 0 && y%stride_h == 0)
			{
				x=x/stride_w;
				y=y/stride_h;
				q22 = (y >= 0 && x >= 0 && y < top_height && x < top_width) ? in[y * top_width + x] : 0;
			}

			val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
		}
	}

	bottom_diff[index] += val;
	//printf("(%d,%d,%d,%d),(%d,%d):(%f,%f),(%f,%f,%f,%f),(%f),\n",n,c,h,w, y1,x1, shiftX,shiftY, q11,q21,q12,q22, val);
  }
}

template <typename Dtype>
__global__ void backwardPosition(const int nthreads, const Dtype* const bottom_data, const Dtype* const top_diff, const Dtype* const weights,
    const int in_channels, const int out_channels, const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const Dtype shiftX, const Dtype shiftY, const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* xpos_diff, Dtype* ypos_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	//index : C1*C2*H*W
    const int out_spatial_dim = top_height * top_width;
    const int c1 = index / (out_channels * out_spatial_dim);
    const int idx = index % (out_channels * out_spatial_dim);
    const int c2 = idx / out_spatial_dim;
    const int idx2 = idx % out_spatial_dim;
    const int h = idx2 / top_width;
    const int w = idx2 % top_width;

    int ix = floor((double)shiftX);
    int iy = floor((double)shiftY);
    Dtype dx = shiftX-ix;
   	Dtype dy = shiftY-iy;

   	const int h_offset = h * stride_h - pad_h;
   	const int w_offset = w * stride_w - pad_w;

	int x1 = w_offset + ix;
	int x2 = x1+1;
	int y1 = h_offset + iy;
	int y2 = y1+1;
	int x,y;

	const Dtype *in =  bottom_data + c1*bottom_width*bottom_height;

	y = y1;
	x = x1;
	Dtype q11 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

	y = y1;
	x = x2;
	Dtype q21 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

	y = y2;
	x = x1;
	Dtype q12 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;

	y = y2;
	x = x2;
	Dtype q22 = (y >= 0 && x >= 0 && y < bottom_height && x < bottom_width) ? in[y * bottom_width + x] : 0;


	Dtype val_x = (1-dy)*(q21-q11)+dy*(q22-q12);
	Dtype val_y = (1-dx)*(q12-q11)+dx*(q22-q21);

	Dtype err = top_diff[c2*top_width*top_height + h*top_width + w];
	Dtype weight = weights[c2*in_channels + c1];

	xpos_diff[index] = val_x*err*weight;
	ypos_diff[index] = val_y*err*weight;

	//printf("(%d,%d,%d,%d),(%d,%d):(%f,%f),(%f,%f,%f,%f),(%f,%f),(%f,%f,%f)\n",c1,c2,h,w,y1,x1,shiftX,shiftY, q11,q21,q12,q22, val_x,val_y, err, val_x*err*weight, val_y*err*weight);
  }
}

template <typename Dtype>
__global__ void flipTop2(const int n, const int nums, const int channels, const int spatial_dim, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
	  //c_in, c_out, f
	  const int dim = channels*spatial_dim;
	  const int num = index/dim;
	  const int idx = index%dim;
	  const int c = idx/spatial_dim;
	  const int i = idx%spatial_dim;

	  out[(c*nums + num)*spatial_dim + i] = in[index];
  }
}



template <typename Dtype>
void AConvFastLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype *top_data = top[0]->mutable_gpu_data();

	const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* xpos = this->blobs_[2]->gpu_data();
	const Dtype* ypos = this->blobs_[3]->gpu_data();

	Dtype* intp_data = conv_bottom_.mutable_gpu_data();

	//Blob<Dtype>* cwBlob = conv_layer_->blobs()[0].get();	//Conv Weight Blob
	Dtype* cw = conv_layer_->blobs()[0]->mutable_gpu_data();

	caffe_gpu_set(top[0]->count(),(Dtype)0.,top_data);

	for(int g=0;g<this->pos_group_;g++)
	{
		for(int f=0;f<kernel_n_;f++)
		{
		  int count = conv_bottom_.count();
		  interpolated<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom_data, this->num_, this->channels_, bottom_height_, bottom_width_,
				xpos + (g*kernel_n_+f), ypos + (g*kernel_n_+f), stride_h_, stride_w_, pad_h_, pad_w_,
				top_height_, top_width_, intp_data);

		  count = cwCount_;
		  copyWeight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, this->num_output_/this->pos_group_, this->channels_, kernel_n_, weight + g*wgroup_offset_ + f, cw);

		  //forward
		  conv_layer_->Forward(conv_bottom_vec_, conv_top_vec_);

		  //add outputs
		  count = conv_top_.count();
		  mergeOutput<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, this->num_output_, this->num_output_/this->pos_group_, top_height_*top_width_, g, conv_top_.gpu_data(), top_data);
		}
	}

	if (this->bias_term_) {
		for (int n = 0; n < this->num_; ++n) {
		  const Dtype* bias = this->blobs_[1]->gpu_data();
		  this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
		}
	}
}

template <typename Dtype>
void AConvFastLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype *bottom_data = bottom[0]->gpu_data();
	Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype *top_diff = top[0]->gpu_diff();
	Dtype* intp_data = conv_bottom_.mutable_gpu_data();
	const Dtype* intp_diff = conv_bottom_.gpu_diff();

	const Dtype* xpos = this->blobs_[2]->gpu_data();
	const Dtype* ypos = this->blobs_[3]->gpu_data();
	Dtype* xposDiff = this->blobs_[2]->mutable_gpu_diff();
	Dtype* yposDiff = this->blobs_[3]->mutable_gpu_diff();

	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

	Blob<Dtype>* cwBlob = conv_layer_->blobs()[0].get();
	Dtype* cw = cwBlob->mutable_gpu_data();
	Dtype* cwDiff = cwBlob->mutable_gpu_diff();

	Dtype* conv_top_diff = conv_top_.mutable_gpu_diff();

	caffe_gpu_set(bottom[0]->count(),(Dtype)0.,bottom_diff);

	for(int g=0;g<this->pos_group_;g++)
	{
		int count = conv_top_.count();
		splitOutput<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, this->num_output_, this->num_output_/this->pos_group_, top_height_*top_width_, g, top_diff, conv_top_diff);

		for(int f=0;f<kernel_n_;f++)
		{
		  caffe_gpu_set(cwCount_,(Dtype)0.,cwDiff);

		  //Reconstruct weight/bottom
		  int count = conv_bottom_.count();
		  interpolated<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom_data, this->num_, this->channels_, bottom_height_, bottom_width_,
				xpos+(g*kernel_n_+f), ypos+(g*kernel_n_+f), stride_h_, stride_w_, pad_h_, pad_w_,
				top_height_, top_width_, intp_data);

		  count = cwCount_;
		  copyWeight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  	    count, this->num_output_/this->pos_group_, this->channels_, kernel_n_, weight + g*wgroup_offset_ + f, cw);

		  //backward
		  conv_layer_->Backward(conv_top_vec_, propagate_down, conv_bottom_vec_);

		  //bottom diff(accumulate interpolate difference)
		  count = bottom[0]->count();
		  interpolatedBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, intp_diff, this->num_, this->channels_, top_height_, top_width_,
				xpos+(g*kernel_n_+f), ypos+(g*kernel_n_+f), stride_h_, stride_w_, pad_h_, pad_w_,
				bottom_height_, bottom_width_, bottom_diff);

		  //weight diff
		  count = cwCount_;
		  copyDiff<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, this->num_output_/this->pos_group_, this->channels_, kernel_n_, cwDiff, weight_diff + g*wgroup_offset_ + f);
		}
	}

	//Position diff
	if (this->param_propagate_down_[2] && this->param_propagate_down_[3]) {
	  int kernel_dim = this->blobs_[0]->count(1);

	  Dtype* pos_diff = pos_diff_buffer_.mutable_gpu_data();
	  //caffe_gpu_set(pos_diff_buffer_.count(),(Dtype)0.,pos_diff);

	  const Dtype *diff_muliplier = pos_diff_multiplier_.gpu_data();
	  Dtype *diff_temp = pos_diff_temp.mutable_gpu_data();

	  Dtype *fliptop_diff = fliptop_.mutable_gpu_diff();

	  int count = top[0]->count();
	  flipTop2<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, this->num_, this->num_output_, top[0]->count(2), top_diff, fliptop_diff);

	  this->position_gpu_gemm(bottom_data, fliptop_diff, pos_diff, xpos, ypos);

	  count = pos_diff_buffer_.count()/2;
	  multiplyWeight2<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, kernel_dim, weight, kernel_n_,
				pos_diff, pos_diff);

	  const int pos_diff_offset = pos_group_offset_*kernel_dim*2;
	  for (int pg = 0; pg < this->pos_group_; ++pg)
	  {
		  //summation along output channel
		  caffe_gpu_gemv<Dtype>(CblasTrans, pos_group_offset_, kernel_dim*2, 1.,
			  pos_diff + pos_diff_offset*pg, diff_muliplier, 0., diff_temp);

		  //summation along input channel
		  caffe_gpu_gemv<Dtype>(CblasTrans, this->channels_, kernel_n_, 1.,
			  diff_temp, diff_muliplier, 1., xposDiff + kernel_n_*pg);
		  caffe_gpu_gemv<Dtype>(CblasTrans, this->channels_, kernel_n_, 1.,
			  diff_temp+this->channels_*kernel_n_, diff_muliplier, 1., yposDiff + kernel_n_*pg);
	  }

	  // Position Regularization
	  if (this->param_propagate_down_[2] && this->param_propagate_down_[3]) {
		int count = kernel_n_*this->pos_group_;
		applyConstraint2<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, kernel_n_,
					this->blobs_[2]->gpu_data(), this->blobs_[3]->gpu_data(), xposDiff, yposDiff, normalize_diff_);
	  }
	}

    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(AConvFastLayer);

}  // namespace caffe
