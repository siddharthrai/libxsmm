/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#if defined(USE_LIBXSMM_JIT)
#include <libxsmm.h>
#endif
#include "utils.h"
#include "rtm.h"

template <typename T>
class EmbeddingBagImpl
{
public:
  EmbeddingBagImpl(long M, long E) : M(M), E(E)
  {
#ifdef USE_LIBXSMM_JIT
    _ld = E;
    libxsmm_meltw_unary_shape unary_shape_f32 = libxsmm_create_meltw_unary_shape( E, 0, _ld, _ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltw_unary_shape unary_shape_f16 = libxsmm_create_meltw_unary_shape( E, 0, _ld, _ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltw_binary_shape binary_shape_f32 = libxsmm_create_meltw_binary_shape( E, 1, _ld, _ld, _ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);

    if (sizeof(T) == 4) {
      kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD, unary_shape_f32, (sizeof(long) == 8) ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES );
    } else {
      kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD, unary_shape_f16, (sizeof(long) == 8) ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES );
    }
    kernel1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR, unary_shape_f32, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    kernel2 = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MULADD, binary_shape_f32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0 );
#else
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);
#endif

  }

  ~EmbeddingBagImpl()
  {
    my_free(weight_);
    weight_ = 0;
  }

  void init(T low = -0.1, T high = 0.1)
  {
    init_random(M * E, weight_, low, high);
  }

#ifdef USE_LIBXSMM_JIT
  void forward(long N, long NS, const long *offsets, const long *indices, T *output_)
  {
    #pragma omp for nowait
    for (int n = 0; n < N; n++)
    {
      libxsmm_meltw_unary_param params;
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      unsigned long long __n = end-start;

      params.in.primary = (void*)weight_;
      params.in.secondary = (void*)&indices[start];
      params.in.tertiary = &__n;
      params.out.primary = (void*)((T*)output_ + n * E);
      kernel( &params );
    }
  }
#else
  void forward(long N, long NS, const long *offsets, const long *indices, T *output_)
  {
    T(*__restrict weight)[E] = (T(*)[E])weight_;
    T(*__restrict output)[E] = (T(*)[E])output_;

#pragma omp for nowait
    for (long n = 0; n < N; n++)
    {
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
#pragma omp simd
      for (long v = 0; v < E; v++)
        output[n][v] = 0;
      for (long s = start; s < end; s++)
      {
        auto ind = indices[s];
#pragma omp simd
        for (long v = 0; v < E; v++)
        {
          output[n][v] += weight[ind][v];
        }
      }
    }
  }
#endif

#ifdef USE_LIBXSMM_JIT
  void backward(long N, long NS, const T *gradout_, const long *offsets, const long *indices, T *values_)
  {
  }
#else
  void backward(long N, long NS, const T *gradout_, const long *offsets, const long *indices, T *values_)
  {
  }
#endif

#ifdef USE_LIBXSMM_JIT
  void update(long NS, const T *grads_, const long *indices, float lr, long M, int use_rtm)
  {
  }
#else
  void update(long NS, const T *grads_, const long *indices, float lr, long M, int use_rtm)
  {
  }
#endif

  T *weight_;
  long M;
  long E;

#ifdef USE_LIBXSMM_JIT
  int _ld;
  libxsmm_meltwfunction_unary kernel;
  libxsmm_meltwfunction_unary kernel1;
  libxsmm_meltwfunction_binary kernel2;
#endif
};

