// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Check that skinny scaled matmuls are sent down the #iree_gpu.pipeline<VectorDistribute> pipeline.

// CHECK-DAG: #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options =
// CHECK-DAG: #[[$F32_ATTENTION_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 =

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @skinny_scaled_matmul() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x512x16xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x16xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x512xf8E8M0FNU>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf8E8M0FNU>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1024xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 512, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x512x16xi8>> -> tensor<4x512x16xi8>
  %6 = iree_tensor_ext.bitcast %5 : tensor<4x512x16xi8> -> tensor<4x512x32xf4E2M1FN>
  %7 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1024, 512, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x16xi8>> -> tensor<1024x512x16xi8>
  %8 = iree_tensor_ext.bitcast %7 : tensor<1024x512x16xi8> -> tensor<1024x512x32xf4E2M1FN>
  %9 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x512xf8E8M0FNU>> -> tensor<4x512xf8E8M0FNU>
  %10 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf8E8M0FNU>> -> tensor<1024x512xf8E8M0FNU>
  %11 = tensor.empty() : tensor<4x1024xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %13 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%6, %8, %9, %10 : tensor<4x512x32xf4E2M1FN>, tensor<1024x512x32xf4E2M1FN>, tensor<4x512xf8E8M0FNU>, tensor<1024x512xf8E8M0FNU>) outs(%12 : tensor<4x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %14 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %15 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %16 = arith.mulf %14, %15 : f32
    %17 = arith.addf %out, %16 : f32
    linalg.yield %17 : f32
  } -> tensor<4x1024xf32>
  iree_tensor_ext.dispatch.tensor.store %13, %4, offsets = [4, 1024], sizes = [4, 1024], strides = [1, 1] : tensor<4x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1024xf32>>
  return
}
// CHECK-LABEL: @skinny_scaled_matmul

// Small f32 attention keeps the reduction config instead of falling back to
// generic Distribute, which miscompiles this ROCm e2e shape.

// CHECK-LABEL: func.func @attention_f32_3x3x4()
//  CHECK-SAME: attributes {translation_info = #[[$F32_ATTENTION_TRANSLATION]]}
//       CHECK: iree_linalg_ext.online_attention
//  CHECK-SAME: lowering_config = #iree_gpu.lowering_config

#pipeline_layout_attention = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @attention_f32_3x3x4() {
  %cst = arith.constant 0.5 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_attention) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_attention) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_attention) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout_attention) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x4xf32>>
  %q = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [3, 3, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4xf32>> -> tensor<3x3x4xf32>
  %k = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4xf32>> -> tensor<3x3x4xf32>
  %v = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [3, 3, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4xf32>> -> tensor<3x3x4xf32>
  %empty = tensor.empty() : tensor<3x3x4xf32>
  %empty_red = tensor.empty() : tensor<3x3xf32>
  %att:3 = iree_linalg_ext.online_attention
      {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> ()>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>]}
      ins(%q, %k, %v, %cst : tensor<3x3x4xf32>, tensor<3x3x4xf32>, tensor<3x3x4xf32>, f32)
      outs(%empty, %empty_red, %empty_red : tensor<3x3x4xf32>, tensor<3x3xf32>, tensor<3x3xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<3x3x4xf32>, tensor<3x3xf32>, tensor<3x3xf32>
  iree_tensor_ext.dispatch.tensor.store %att#0, %3, offsets = [0, 0, 0], sizes = [3, 3, 4], strides = [1, 1, 1] : tensor<3x3x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x4xf32>>
  return
}
