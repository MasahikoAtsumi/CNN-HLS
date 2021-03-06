#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

typedef float DTYPE;
#define line_row  7
#define line_col  28
#define max_Ksize 7
#define max_Osize 128
#define u_Xsize 8

void conv(// static paremeter
	  int k_size,
	  int in_size,
	  int out_size,
	  int ochs,
	  int stride,
	  int unroll_och,
	  int unroll_y,
	  int unroll_x,
	  int line_id,

	  // dynamic parameter
	  int  *dy_param, // 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	  bool *flag_list,
	  // features
	  DTYPE **in_buffer,
	  DTYPE **out_buffer,
	  
	  // weight
	  DTYPE **weight_buffer
	  )
{
  int target_row = 0;
  // dy_param 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
  for(int id_uoch = 0; id_uoch < unroll_och; id_uoch++){
    for(int id_ux = 0; id_ux < unroll_x; id_ux++){
      int u_in_x = dy_param[2] + dy_param[5] + id_ux*stride;
      int u_och  = dy_param[0] + id_uoch;
      if( (u_in_x < in_size) && (u_och < ochs) ){
	if(line_id + dy_param[4] > line_row-1){target_row = line_id + dy_param[4] - line_row;}
	else{target_row = line_id + dy_param[4];}
	// write to accumulator
	out_buffer[u_och][id_ux] += in_buffer[target_row][u_in_x]*weight_buffer[u_och][dy_param[4]*k_size + dy_param[5]];
      }
    }
  }
}
