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
	int out_id     = dynamic_parameters[0]*out_size*out_size + dynamic_parameters[6]*out_size + dynamic_parameters[7];
	int in_id      = ( (dynamic_parameters[3])*in_size + (dynamic_parameters[1]+dynamic_parameters[4]) )*in_size + (dynamic_parameters[2]+dynamic_parameters[5]);
	int weight_id  = ( (dynamic_parameters[0]*inchs + dynamic_parameters[3])*k_size + (dynamic_parameters[4]))*k_size + (dynamic_parameters[5]);
	out_buffer[out_id] += in_buffer[in_id]*weight_buffer[weight_id];

	if( flag_list[0] == 1){
		out_buffer[out_id] += bias_buffer[dynamic_parameters[0]];
		flag_list[0]        = 0; // bias flag |_
	}
}
