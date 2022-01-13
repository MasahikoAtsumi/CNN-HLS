#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

#include "./func/slide_func.h"
#include "./func/conv.h"
#include "./func/maxpool.h"
#include "./func/relu.h"
#include "./func/load_wb.h"

typedef float DTYPE;
static const std::size_t max_size = 5*5*128*128;
// #define HLS
void cnn_unit(
  int const static_parameters[8],
  #ifdef HLS
  DTYPE const in_data[max_size],
  DTYPE out_data[max_size],
  DTYPE const weight_data[max_size],
  DTYPE const bias_data[max_size]
  #else
  DTYPE* in_data,
  DTYPE* out_data,
  DTYPE* weight_data,
  DTYPE* bias_data
  #endif
)
{

  // static paremeter 0:opecode, 1:inchs, 2:in_size, 3:ochs, 4:out_size, 5:k_size, 6:stride, 7:padding
  const int opecode  = static_parameters[0];
  const int inchs    = static_parameters[1];
  const int in_size  = static_parameters[2];
  const int ochs     = static_parameters[3];
  const int out_size = static_parameters[4];
  const int k_size   = static_parameters[5];
  const int stride   = static_parameters[6];
  const int pad      = static_parameters[7];

  int unroll_och = 8;
  int unroll_y   = 2;
  int unroll_x   = 4;

  // ================================================================
  // below code should be fixed for HLS
  // do not use "%"
  int iteration_x = 0;
  if(out_size%(unroll_x) == 0) iteration_x = out_size/(unroll_x);
  else iteration_x = out_size/(unroll_x) + 1;

  int iteration_och = 0;
  if(ochs%unroll_och == 0) iteration_och = ochs/unroll_och;
  else iteration_och = ochs/unroll_och + 1;

  int ofst_unroll_dst_y = int(out_size/unroll_y) + out_size%unroll_y;
  int ofst_unroll_in_y  = ofst_unroll_dst_y*stride;
  int iteration_y = 0;
  if(out_size%unroll_y == 0) iteration_y = out_size/unroll_y;
  else iteration_y = out_size/unroll_y + 1;
  // ================================================================

  // dynamic parameter
  int dynamic_parameters[8] = {0}; // och, in_y, in_x, inch, ky, kx, dst_y, dst_x
  bool flag_list[2]         = {0}; // bias_flag, end_flag

  #ifdef HLS
  DTYPE in_buffer[28*28*128]       = {0};
  DTYPE acc_buffer[32*2*4]         = {0};
  DTYPE out_buffer[28*28*128]      = {0};
  DTYPE weight_buffer[5*5*128*128] = {0};
  DTYPE bias_buffer[128]           = {0};
  #else
  DTYPE *in_buffer     = (DTYPE *)malloc(28*28*128*sizeof(DTYPE));
  DTYPE *acc_buffer    = (DTYPE *)malloc(32*4*4*sizeof(DTYPE));
  DTYPE *out_buffer    = (DTYPE *)malloc(28*28*128*sizeof(DTYPE));;
  DTYPE *weight_buffer = (DTYPE *)malloc(5*5*128*128*sizeof(DTYPE));;
  DTYPE *bias_buffer   = (DTYPE *)malloc(128*sizeof(DTYPE));
  #endif

  // laod data
  switch (opecode){
    case 2:load_wb(in_data, in_buffer, ochs*out_size*out_size);
    break;
    case 1:load_wb(in_data, in_buffer, ochs*in_size*in_size  );
    break;
    case 0:load_wb(in_data, in_buffer, inchs*in_size*in_size );
    break;
}

  if(opecode == 0){
    load_wb(weight_data, weight_buffer, ochs*inchs*k_size*k_size);
    load_wb(bias_data,   bias_buffer,   ochs);
  }

  // init state
  int state = 1;
  int max_loop = 3*28*28*128*128*5*5;

  for( int id_loop = 0; id_loop < max_loop; id_loop++ ){
    if(flag_list[1] == 1) break; // end flag
    switch (state) {
      // sliding operation
      case 0:
      slide_func(
        // static paremeter
        k_size, in_size, out_size, ochs, inchs, stride,
        unroll_och, iteration_och, iteration_y, unroll_x, iteration_x,
        // dynamic parameter
        dynamic_parameters,
        flag_list
      );
      state = 1;
      break;

      // convolutional operation
      case 1:
      switch (opecode) {
        case 0:
        conv(
          // static paremeter
          k_size, in_size, out_size, ochs, inchs, stride,
          unroll_och, unroll_y, ofst_unroll_in_y, ofst_unroll_dst_y, unroll_x,
          // dynamic parameter
          dynamic_parameters, flag_list,
          // features
          in_buffer, acc_buffer, out_buffer,
          // weight and bias
          weight_buffer,bias_buffer
        );
        break;

        case 1:
        maxpool(
          // static paremeter
          k_size, in_size, out_size, ochs, stride,
          unroll_och, unroll_y, ofst_unroll_in_y, ofst_unroll_dst_y, unroll_x,
          // dynamic parameter
          dynamic_parameters, flag_list,
          // features
          in_buffer, acc_buffer, out_buffer
        );
        break;

        case 2:
        relu(
          // static paremeter
          out_size, ochs,
          unroll_och, unroll_y, ofst_unroll_in_y, ofst_unroll_dst_y, unroll_x,
          // dynamic parameter
          dynamic_parameters,
          // features
          in_buffer, out_buffer
        );
        break;
      }
      state = 0;
      break;
    }
  }
  // wb to host
  load_wb(out_buffer, out_data, out_size*out_size*ochs);

  // free mem
  #ifdef HLS
  #else
  free(in_buffer);
  free(out_buffer);
  free(weight_buffer);
  free(bias_buffer);
  #endif
}
