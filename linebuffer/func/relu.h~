#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

typedef float DTYPE;

void relu(
	// static paremeter
	int out_size,
	int ochs,
	int unroll_och,
	int unroll_y,
	int ofst_unroll_in_y,
	int ofst_unroll_dst_y,
	int unroll_x,

	// dynamic parameter
	int  *dy_param, // 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	// features
	DTYPE const *in_buffer,
	DTYPE *out_buffer
)
{
	// dy_param 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	// parallel multiplication
	for(int id_uoch = 0; id_uoch < unroll_och; id_uoch++){
#pragma HLS PIPELINE
		for(int id_ux = 0; id_ux < unroll_x; id_ux++){
			int u_in_x = dy_param[2] + id_ux  ;
			int u_och  = dy_param[0] + id_uoch;

			if( (u_in_x < out_size) && (u_och < ochs) ){

				//  index
				int id_in_above = ( (u_och)*out_size + (                  dy_param[1]) )*out_size + u_in_x;
				int id_in_below = ( (u_och)*out_size + ( ofst_unroll_in_y+dy_param[1]) )*out_size + u_in_x;
				// write from accumulator
				if(in_buffer[id_in_above] > 0){out_buffer[id_in_above] = in_buffer[id_in_above];}
				else{out_buffer[id_in_above] = 0;}
				if( ofst_unroll_in_y+dy_param[1] < out_size ){
					if(in_buffer[id_in_below] > 0){out_buffer[id_in_below] = in_buffer[id_in_below];}
					else{out_buffer[id_in_below] = 0;}
				}
			}
		}
	}

}
