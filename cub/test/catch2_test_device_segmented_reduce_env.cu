// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_reduce_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Min, device_segmented_reduce_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Max, device_segmented_reduce_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMin, device_segmented_reduce_argmin);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMax, device_segmented_reduce_argmax);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Test data: 3 segments: {8,6,7,5, 3,0,9, 1,2} with offsets {0,4,7,9}

#if TEST_LAUNCH == 0

TEST_CASE("Device segmented sum works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

#endif

TEST_CASE("Device segmented sum uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Sum(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device fixed-size segmented reduce works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedReduce::Reduce(
            d_in.begin(), d_out.begin(), num_segments, segment_size, ::cuda::std::plus<>{}, 0));

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented sum works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented min works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<int> expected{6, 0};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented max works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<int> expected{8, 5};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented argmin works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::host_vector<cuda::std::pair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].first == 1);
  REQUIRE(h_out[0].second == 6);
  REQUIRE(h_out[1].first == 2);
  REQUIRE(h_out[1].second == 0);
}

TEST_CASE("Device fixed-size segmented argmax works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::host_vector<cuda::std::pair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].first == 0);
  REQUIRE(h_out[0].second == 8);
  REQUIRE(h_out[1].first == 0);
  REQUIRE(h_out[1].second == 5);
}

TEST_CASE("Device fixed-size segmented reduce uses custom stream", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop};

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedReduce::Reduce(
            d_in.begin(), d_out.begin(), num_segments, segment_size, ::cuda::std::plus<>{}, 0, env));

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device fixed-size segmented sum uses custom stream", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop};

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, segment_size, env));

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device fixed-size segmented min uses custom stream", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop};

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, segment_size, env));

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{6, 0};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device fixed-size segmented max uses custom stream", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop};

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, segment_size, env));

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{8, 5};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device fixed-size segmented argmin uses custom stream", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop};

  REQUIRE(
    cudaSuccess == cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, segment_size, env));

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::host_vector<cuda::std::pair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].first == 1);
  REQUIRE(h_out[0].second == 6);
  REQUIRE(h_out[1].first == 2);
  REQUIRE(h_out[1].second == 0);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device fixed-size segmented argmax uses custom stream", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop};

  REQUIRE(
    cudaSuccess == cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, segment_size, env));

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::host_vector<cuda::std::pair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].first == 0);
  REQUIRE(h_out[0].second == 8);
  REQUIRE(h_out[1].first == 0);
  REQUIRE(h_out[1].second == 5);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

template <int BlockThreads>
struct reduce_tuning : cub::detail::reduce::tuning<reduce_tuning<BlockThreads>>
{
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct fn
  {
    struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>
    {
      struct ReducePolicy
      {
        static constexpr int VECTOR_LOAD_LENGTH = 1;

        static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BLOCK_REDUCE_WARP_REDUCTIONS;

        static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;

        static constexpr int ITEMS_PER_THREAD = 1;
        static constexpr int BLOCK_THREADS    = BlockThreads;
      };

      using SingleTilePolicy      = ReducePolicy;
      using SegmentedReducePolicy = ReducePolicy;
    };

    using MaxPolicy = Policy500;
  };
};

struct get_scan_tuning_query_t
{};

struct scan_tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_scan_tuning_query_t&) const noexcept
  {
    return *this;
  }

  // Make sure this is not used
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct fn
  {};
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device segmented sum can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  // We are expecting that `scan_tuning` is ignored
  auto env = cuda::execution::__tune(reduce_tuning<target_block_size>{}, scan_tuning{});

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}
