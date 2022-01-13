#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

typedef float DTYPE;

void operation(
	    // static paremeter
			int k_size,
		  int in_size,
			int out_size,
		  int ochs,
		  int inchs,
		  int stride,
			int unroll_och,
			int unroll_x,

	    // dynamic parameter
			int  *dynamic_parameters, // 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
			bool *flag_list,
			// features
			DTYPE const *in_buffer,
			DTYPE *acc_buffer,
			DTYPE *out_buffer,
			// weight and bias
			DTYPE const *weight_buffer,
			DTYPE const *bias_buffer
	    )
{
	// 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	// parallel multiplication
	#ifdef p_conv
	for(int id_uoch = 0; id_uoch < unroll_och; id_uoch++){
		for(int id_ux = 0; id_ux < unroll_x; id_ux++){

			int u_in_x = dynamic_parameters[2]+dynamic_parameters[5] + id_ux*stride;
			int u_och  = dynamic_parameters[0] + id_uoch;
			if( (u_in_x < in_size) && (u_och < ochs) ){
				int in_id      = ( (dynamic_parameters[3])*in_size + (dynamic_parameters[1]+dynamic_parameters[4]) )*in_size + u_in_x;
				int weight_id  = ( (u_och*inchs + dynamic_parameters[3])*k_size + (dynamic_parameters[4]))*k_size + (dynamic_parameters[5]);
				acc_buffer[id_uoch*unroll_x + id_ux] += in_buffer[in_id]*weight_buffer[weight_id];
			}
		}
	}

	// add bias
	if( flag_list[0] == 1){
		for(int id_uoch = 0; id_uoch < unroll_och; id_uoch++){
			for(int id_ux = 0; id_ux < unroll_x; id_ux++){
				int u_dst_x = dynamic_parameters[7] + id_ux;
				int u_och   = dynamic_parameters[0] + id_uoch;
				if( (u_och < ochs) && (u_dst_x < out_size)){
					int u_out_id = ( (u_och*out_size) + dynamic_parameters[6] )*out_size + u_dst_x;
					out_buffer[u_out_id] += acc_buffer[id_uoch*unroll_x + id_ux] + bias_buffer[u_och];
					acc_buffer[id_uoch*unroll_x + id_ux] = 0; // flush acc
				}
			}
		}
		flag_list[0]  = 0; // bias flag |_
	}
}
