#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

typedef float DTYPE;

void maxpool(
	// static paremeter
	int k_size,
	int in_size,
	int out_size,
	int ochs,
	int stride,
	int unroll_och,
	int unroll_y,
	int ofst_unroll_in_y,
	int ofst_unroll_dst_y,
	int unroll_x,

	// dynamic parameter
	int  *dy_param, // 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	bool *flag_list,
	// features
	DTYPE const *in_buffer,
	DTYPE *acc_buffer,
	DTYPE *out_buffer
)
{
	/***
	Abstract of Unroll method

	intput feature              0 1 2 3  <- unroll_x
	 __________                 __________     _
	|  				| <- unroll_y0   |_|_|_|_|  ...  |<- unroll_y
	|_________|                |_________     _|
	|					|	<- unroll_y1
	|_________|

	***/

	// dy_param 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	// parallel multiplication
	for(int id_uoch = 0; id_uoch < unroll_och; id_uoch++){
#pragma HLS PIPELINE
		for(int id_ux = 0; id_ux < unroll_x; id_ux++){
			int u_in_x = dy_param[2] + dy_param[5] + id_ux*stride;
			int u_och  = dy_param[0] + id_uoch;
			if( (u_in_x < in_size) && (u_och < ochs) ){
				//  index
				int id_in_above = ( (u_och)*in_size + (                  dy_param[1]+dy_param[4]) )*in_size + u_in_x;
				int id_in_below = ( (u_och)*in_size + ( ofst_unroll_in_y+dy_param[1]+dy_param[4]) )*in_size + u_in_x;

				// write to accumulator
				acc_buffer[id_uoch*unroll_y*unroll_x + 0*unroll_x + id_ux] = std::max(acc_buffer[id_uoch*unroll_y*unroll_x + 0*unroll_x + id_ux] , in_buffer[id_in_above]);
				if( ofst_unroll_in_y+dy_param[1]+dy_param[4] < in_size ){
						acc_buffer[id_uoch*unroll_y*unroll_x + 1*unroll_x + id_ux] = std::max(acc_buffer[id_uoch*unroll_y*unroll_x + 1*unroll_x + id_ux], in_buffer[id_in_below]);
				}
			}
		}
	}

	// flush acc
	if( flag_list[0] == 1){
		for(int id_uoch = 0; id_uoch < unroll_och; id_uoch++){
#pragma HLS UNROLL
			for(int id_ux = 0; id_ux < unroll_x; id_ux++){
				int u_dst_x = dy_param[7] + id_ux;
				int u_och   = dy_param[0] + id_uoch;

				if( (u_och < ochs) && (u_dst_x < out_size)){
					int id_out_above = ( (u_och*out_size) +                     dy_param[6] )*out_size + u_dst_x;
					int id_out_below = ( (u_och*out_size) + ofst_unroll_dst_y + dy_param[6] )*out_size + u_dst_x;
					// write from accumulator
					out_buffer[id_out_above] = acc_buffer[id_uoch*unroll_y*unroll_x + 0*unroll_x + id_ux];
					if( ofst_unroll_dst_y + dy_param[6] < out_size ){
						out_buffer[id_out_below] = acc_buffer[id_uoch*unroll_y*unroll_x + 1*unroll_x + id_ux];
					}
				}
				acc_buffer[id_uoch*unroll_y*unroll_x + 0*unroll_x + id_ux] = 0;
				acc_buffer[id_uoch*unroll_y*unroll_x + 1*unroll_x + id_ux] = 0;
			}
		}
		flag_list[0]  = 0; //  flag |_
	}

}
