#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <random>

#include "./load_from_txt.h"
#include "cnn_unit.cpp"

typedef float DTYPE;
using namespace std;

// #define HLS

// static const std::size_t max_size = 25*128*128;
void cnn_unit(
	int   const static_parameters[8],
	#ifdef HLS
	DTYPE const in_data[max_size],
  DTYPE out_data[max_size],
  DTYPE const weight_data[max_size],
  DTYPE const bias_data[max_size]
	#else
	DTYPE* in_data,
	DTYPE* out_data,
	DTYPE* weight,
	DTYPE* bias
	#endif
);

int main()
{
	int in_parameters[]     = { 1, 1, 3, 3};
	int out_parameters[]    = { 1, 1, 1, 1};
	int weight_parameters[] = { 1, 1, 3, 3};
	int bias_parameters[]   = { 1, 1,  1,  1};
	int const static_parameters[] = {0, 1, 3, 1, 1, 3, 2, 0};

	#ifdef HLS
	DTYPE stream_in[in_parameters[0]*in_parameters[1]*in_parameters[2]*in_parameters[3]]              = {0};
	DTYPE stream_out[out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3]]         = {0};
	DTYPE answer[out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3]]             = {0};
	DTYPE weight[weight_parameters[0]*weight_parameters[1]*weight_parameters[2]*weight_parameters[3]] = {0};
	DTYPE bias[bias_parameters[0]*bias_parameters[1]*bias_parameters[2]*bias_parameters[3]]           = {0};
	#else
	DTYPE *stream_in  = (DTYPE *)malloc(in_parameters[0]*in_parameters[1]*in_parameters[2]*in_parameters[3]*sizeof(DTYPE) );
	DTYPE *stream_out = (DTYPE *)malloc(out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3]*sizeof(DTYPE) );
	DTYPE *answer     = (DTYPE *)malloc(out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3]*sizeof(DTYPE) );
	DTYPE *weight     = (DTYPE *)malloc(weight_parameters[0]*weight_parameters[1]*weight_parameters[2]*weight_parameters[3]*sizeof(DTYPE) );
	DTYPE *bias       = (DTYPE *)malloc(bias_parameters[0]*bias_parameters[1]*bias_parameters[2]*bias_parameters[3]*sizeof(DTYPE) );
	#endif

	int out_val = out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3];
	int error = 0;
	int test_num = 100;

	// load in data and inference----------------------------------------------------
	for( int id_in = 0; id_in < test_num; id_in++){
		std::random_device rd;
	  std::default_random_engine eng(rd());
	  std::uniform_real_distribution<> distr(0.01, -0.01);

		DTYPE ans   = 0;
		int int_val = 0;
		for(int kernel_id = 0; kernel_id < 9; kernel_id++){
				// generate feature
				DTYPE float_val      = distr(eng);
				stream_in[kernel_id] = float_val;
				//generate weight
				DTYPE weight_val  = distr(eng);
				weight[kernel_id] = weight_val;
				ans += weight_val*float_val;
				// printf("float:%f, int:%d\n" ,float_val, int_val);
				// printf("tmp_val: %f\n", ans);
			}
			DTYPE bias_val  = distr(eng);
			bias[0] = bias_val;
			ans += bias_val;
			answer[0] = ans;

	  // Inference ----------------------------------------------------
	  // printf("start inference\n");
	  cnn_unit( static_parameters, stream_in, stream_out, weight, bias);
	  // printf("end\n");

	  // verify output ------------------------------------------------
	  for(int x = 0; x < out_val; x++){
			if(( (answer[x] > stream_out[x]) && (answer[x] - stream_out[x] > 0.000001) ) ||
	       ( (answer[x] < stream_out[x]) && (stream_out[x] - answer[x] > 0.000001) ))
	    {
	      error = 1;
	      printf( "RESULT idx=%d bench=%f test=%f\n", x, answer[x], stream_out[x]);
	    }
	  }

	  if( error){
	    printf("TEST ERROR\n");
			break;
	  }
	} // --- end for loop

	if(error){
		return 1;
	}
	else{
		printf("TEST PASS\n");
		return 0;
	}

	// free mem
	#ifdef HLS
	#else
	free(stream_in);
	free(stream_out);
	free(answer);
	free(weight);
	free(bias);
	#endif

}
