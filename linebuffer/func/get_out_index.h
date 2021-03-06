#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

void get_out_index(int *dynamic_parameters,
		   int *out_tgt_col,
		   int *out_tgt_row,
		   int *out_ld_chunk,
		   int *out_wb_chunk,
		   int out_size,
		   int unroll_x,
		   int unroll_y,
		   int iteration_x)
{
  out_tgt_col[0] = dynamic_parameters[7] - unroll_x;
  if(out_tgt_col[0] < 0){out_tgt_col[0] = (iteration_x-1)*unroll_x;}
  out_tgt_row[0] = dynamic_parameters[6];
  if(out_tgt_col[0] == (iteration_x-1)*unroll_x){
    out_tgt_row[0] = dynamic_parameters[6] - unroll_y;
    if(out_tgt_row[0] < 0){out_tgt_row[0] = out_size - unroll_y;}
  }
  if(out_tgt_col[0]+unroll_x-1 > out_size-1){out_wb_chunk[0]=out_size%unroll_x;}
  else{out_wb_chunk[0] = unroll_x;}
  if(dynamic_parameters[7]+unroll_x-1 > out_size-1){out_ld_chunk[0]=out_size%unroll_x;}
  else{out_ld_chunk[0] = unroll_x;}
}
