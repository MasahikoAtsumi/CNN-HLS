#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>
#include <list>

// #define HLS
#include "./func/slide_func.h"
#include "./func/conv.h"
#include "./func/w_load_wb.h"
#include "./func/line_load.h"
#include "./func/out_line_load.h"
#include "./func/out_line_wb.h"
#include "./func/get_out_index.h"

typedef float DTYPE;
#define line_row  7
#define line_col  28
#define max_Ksize 7
#define max_Osize 128
#define max_size 5*5*128*128
#define max_loop 28*28*128*128*7*7


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

  int unroll_och    = 8;
  int unroll_y      = 1;
  int unroll_x      = 8;
  int iteration_och = 0;
  int iteration_y   = 0;
  int iteration_x   = 0;
  int in_st_point   = 0;
  int out_st_point  = 0;
  int w_st_point    = 0;
  int in_tgt_pad    = k_size - stride;

  int out_tgt_row[1]  = {0};
  int out_tgt_col[1]  = {0};
  int out_ld_chunk[1] = {0};
  int out_wb_chunk[1] = {0};
  int l_point[1]      = {0};
  int r_point[1]      = {0}; 

  // ================================================================
  // below code should be fixed for HLS
  // do not use "%"
  if(out_size%(unroll_x) == 0) iteration_x = out_size/(unroll_x);
  else                         iteration_x = out_size/(unroll_x) + 1;
  
  if(ochs%unroll_och == 0) iteration_och = ochs/unroll_och;
  else                     iteration_och = ochs/unroll_och + 1;

  if(out_size%unroll_y == 0) iteration_y = out_size/unroll_y;
  else                       iteration_y = out_size/unroll_y + 1;
  // ================================================================

  // dynamic parameter
  int dynamic_parameters[8] = {0}; // och, in_y, in_x, inch, ky, kx, dst_y, dst_x
  bool flag_list[6]         = {0}; // bias_flag, end_flag, acc_flag, line_flag, weight_flag, out_flag

#ifdef HLS
  DTYPE in_buffer[line_row][line_col]                 = {0};
  DTYPE out_buffer[unroll_och][uncroll_x]             = {0};
  DTYPE weight_buffer[max_Osize][max_Ksize*max_Ksize] = {0};
  DTYPE bias_buffer[max_Ksize]                        = {0};
#else
  DTYPE **in_buffer     = (DTYPE**)calloc(line_row, sizeof(DTYPE*));
  for (int i = 0; i < line_row;   i++) {in_buffer[i] = (DTYPE*)calloc(line_col, sizeof(DTYPE));}
  DTYPE **out_buffer    = (DTYPE**)calloc(max_Osize, sizeof(DTYPE));
  for (int i = 0; i < max_Osize;  i++) {out_buffer[i] = (DTYPE*)calloc(unroll_x*unroll_y, sizeof(DTYPE));}
  DTYPE **weight_buffer = (DTYPE**)calloc(max_Osize, sizeof(DTYPE*));
  for (int i = 0; i < max_Osize; i++) {weight_buffer[i] = (DTYPE*)calloc(max_Ksize*max_Ksize, sizeof(DTYPE));}
  DTYPE *bias_buffer    = (DTYPE *)calloc(max_Osize, sizeof(DTYPE));
#endif

  // laod data
  switch (opecode){
  case 2:line_load(in_data, in_buffer, 1,      out_size, l_point, 0);
    break;
  case 1:line_load(in_data, in_buffer, k_size, in_size,  l_point, 0);
    break;
  case 0:line_load(in_data, in_buffer, k_size, in_size,  l_point, 0);
    break;
  }
  if(opecode == 0){
    w_load_wb(weight_data, weight_buffer, ochs, k_size*k_size, inchs, 0);
    // load bias
    for(int id_bias = 0; id_bias < ochs; id_bias++){
      bias_buffer[id_bias] = bias_data[id_bias];
    }
  }

  // ========== main operation ==========
  while (flag_list[1] != 1) {
    switch (opecode) {
    case 0:
      conv(// static paremeter
	   k_size, in_size, out_size, ochs, stride,
	   unroll_och, unroll_y, unroll_x, r_point[0],
	   // dynamic parameter
	   dynamic_parameters, flag_list,
	   // features
	   in_buffer, out_buffer,
 	   // weight
	   weight_buffer
	   );
      break;
    }

    // sliding operaiton
    slide_func(// static paremeter
	       k_size, in_size, out_size, ochs, inchs, stride,
	       unroll_och, iteration_och, iteration_y, unroll_x, iteration_x,
	       // dynamic parameter
	       dynamic_parameters,
	       flag_list
	       );

    // operation for input feature
    if(flag_list[3] == 1){
      // update r_point
      for(int id_r_point=0; id_r_point < stride; id_r_point++){ r_point[0]++;if(r_point[0] == line_row){r_point[0] = 0;} }
      if(flag_list[4] == 1){
	if(opecode == 0){w_load_wb(weight_data, weight_buffer, ochs, k_size*k_size, inchs, dynamic_parameters[3]);}
	//update line buffer
	l_point[0]  = 0;
	r_point[0]  = 0;
	in_st_point = dynamic_parameters[3]*in_size*in_size;
	line_load(in_data, in_buffer, k_size, in_size, l_point, in_st_point);
	flag_list[4] = 0; // reset flag
      }
      else{	  
	// update line buffer
	in_st_point = (dynamic_parameters[3]*in_size + (in_tgt_pad+dynamic_parameters[1]) )*in_size;
	line_load(in_data, in_buffer, stride, in_size, l_point, in_st_point);
      }
      flag_list[3] = 0; // reset flag
    }

    //operation for output feature
    if(flag_list[5] == 1){
      get_out_index(dynamic_parameters, out_tgt_col, out_tgt_row, out_ld_chunk, out_wb_chunk, out_size, unroll_x, unroll_y, iteration_x);
      out_line_wb(out_buffer, out_data,   ochs, out_wb_chunk[0], out_tgt_row[0],        out_tgt_col[0],        out_size);
      out_line_load(out_data, out_buffer, ochs, out_ld_chunk[0], dynamic_parameters[6], dynamic_parameters[7], out_size);  
      flag_list[5] = 0; // reset flag
    }
  }

  // add bias
  for(int id_och = 0; id_och < ochs; id_och++){
    for(int id_bias = 0; id_bias < out_size*out_size; id_bias++){
      out_data[id_och*out_size*out_size + id_bias] += bias_data[id_och];
    }
  }

  // free mem
#ifdef HLS
#else
  free(in_buffer);
  free(out_buffer);
  free(weight_buffer);
  free(bias_buffer);
#endif
}
