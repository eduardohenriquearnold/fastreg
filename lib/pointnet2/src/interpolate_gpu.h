#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H

#include <torch/serialize/tensor.h>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void nn_ddim_wrapper_fast(int b, int n, int m, int d, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

void nn_ddim_kernel_launcher_fast(int b, int n, int m, int d, const float *unknown,
	const float *known, float *dist2, int *idx, cudaStream_t stream);

void nn_ddim_grad_wrapper_fast(int b, int n, int m, int d, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor idx_tensor, at::Tensor grad_dist2_tensor, at::Tensor grad_unknown_tensor, at::Tensor grad_known_tensor);

void nn_ddim_grad_kernel_launcher_fast(int b, int n, int m, int d, const float *unknown_tensor,
	const float *known_tensor, const int *idx, const float *grad_dist2_tensor, float *grad_unknown_tensor, float *grad_known_tensor, cudaStream_t stream);

void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown,
	const float *known, float *dist2, int *idx, cudaStream_t stream);


void three_interpolate_wrapper_fast(int b, int c, int m, int n, at::Tensor points_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor);

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
    const float *points, const int *idx, const float *weight, float *out, cudaStream_t stream);


void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m, at::Tensor grad_out_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_points_tensor);

void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, const float *grad_out, 
    const int *idx, const float *weight, float *grad_points, cudaStream_t stream);

#endif
