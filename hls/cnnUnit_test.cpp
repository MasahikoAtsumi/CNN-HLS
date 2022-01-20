#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <glob.h>
#include <random>

#include "./load_from_txt.h"
#include "cnn_unit.cpp"

typedef float DTYPE;
using namespace std;

// #define HLS
// #define fivebyfive
// #define threebythree
// #define relu
// #define maxpool

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
	#ifdef fivebyfive
	int in_parameters[]     = { 1,  3, 28, 28};
	int out_parameters[]    = { 1, 64, 24, 24};
	int weight_parameters[] = {64,  3,  5,  5};
	int bias_parameters[]   = { 1,  1,  1, 64};
	int const static_parameters[] = { 0,  3, 28, 64, 24, 5, 1, 0};
	#endif
	#ifdef threebythree
	int in_parameters[]     = { 1,  3, 28, 28};
	int out_parameters[]    = { 1, 64, 13, 13};
	int weight_parameters[] = {64,  3,  3,  3};
	int bias_parameters[]   = { 1,  1,  1, 64};
	int const static_parameters[] = { 0, 3,  28, 64, 13, 3, 2, 0};
	#endif
	#ifdef relu
	int in_parameters[]     = { 1, 64, 13, 13};
	int out_parameters[]    = { 1, 64, 13, 13};
	int weight_parameters[] = {64,  3,  3,  3};
	int bias_parameters[]   = { 1,  1,  1, 64};
	int const static_parameters[] = { 2,  1,  1, 64, 13, 1, 1, 0};
	#endif
	#ifdef maxpool
	int in_parameters[]     = { 1, 64, 13, 13};
	int out_parameters[]    = { 1, 64,  6, 6};
	int weight_parameters[] = {64,  3,  3,  3};
	int bias_parameters[]   = { 1,  1,  1, 64};
	int const static_parameters[] = { 1,  1, 13, 64, 6, 3, 2, 0};
	#endif

	#ifdef HLS
	DTYPE weight[weight_parameters[0]*weight_parameters[1]*weight_parameters[2]*weight_parameters[3]] = {0};
	DTYPE bias[bias_parameters[0]*bias_parameters[1]*bias_parameters[2]*bias_parameters[3]]           = {0};
	#else
	DTYPE *weight     = (DTYPE *)malloc(weight_parameters[0]*weight_parameters[1]*weight_parameters[2]*weight_parameters[3]*sizeof(DTYPE) );
	DTYPE *bias       = (DTYPE *)malloc(bias_parameters[0]*bias_parameters[1]*bias_parameters[2]*bias_parameters[3]*sizeof(DTYPE) );
	#endif

	int out_val  = out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3];
	int error    = 0;
	int test_num = 1;

  // load parameters ----------------------------------------------------
	#ifdef fivebyfive
	char weight_file[256] = "path\\fivebyfive\\txt\\features.0.weight.txt";
	char bias_file[256]   = "path\\fivebyfive\\txt\\features.0.bias.txt";
	#else
	char weight_file[256] = "path\\threebythree\\txt\\features.0.weight.txt";
	char bias_file[256]   = "path\\threebythree\\txt\\features.0.bias.txt";
	#endif

	#ifdef HLS
	DTYPE stream_in[in_parameters[0]*in_parameters[1]*in_parameters[2]*in_parameters[3]]      = {0};
	DTYPE stream_out[out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3]] = {0};
	DTYPE answer[out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3]]     = {0};
	#else
	DTYPE *stream_in  = (DTYPE *)calloc(in_parameters[0]*in_parameters[1]*in_parameters[2]*in_parameters[3],     sizeof(DTYPE) );
	DTYPE *stream_out = (DTYPE *)calloc(out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3], sizeof(DTYPE) );
	DTYPE *answer     = (DTYPE *)calloc(out_parameters[0]*out_parameters[1]*out_parameters[2]*out_parameters[3], sizeof(DTYPE) );
	#endif

	load_from_txt( weight, weight_parameters, weight_file);
	load_from_txt( bias,   bias_parameters,   bias_file  );

	// load in data and inference----------------------------------------------------
	for( int id_in = 0; id_in < test_num; id_in++){
		string id_num = std::to_string(id_in);
		#ifdef fivebyfive
		string dir_path("path\\fivebyfive\\iotxt\\");
		#endif

		#ifdef threebythree
		string dir_path("path\\threebythree\\iotxt\\");
		#endif

		#ifdef maxpool
		string dir_path("path\\maxpool\\iotxt\\");
		#endif

		#ifdef relu
		string dir_path("\\Users\\masah\\project\\cnn_test\\python\\relu\\iotxt\\");
		#endif

		string in_name("/in_target_");
		string out_name("/out_target_");
		string ex_name(".txt");
		string in_file = dir_path + id_num + in_name + id_num + ex_name;
		string out_file = dir_path + id_num + out_name + id_num + ex_name;

		load_from_txt( stream_in, in_parameters,  in_file.c_str() );
		load_from_txt( answer,    out_parameters, out_file.c_str());

	  // Inference ----------------------------------------------------
	  printf("start inference\n");
		cnn_unit( static_parameters, stream_in, stream_out, weight, bias);
	  printf("end\n");

	  // verify output ------------------------------------------------
	  for(int x = 0; x < out_val; x++){
			// printf( "RESULT idx=%d bench=%f test=%f\n", x, answer[x], stream_out[x]);
			if(( (answer[x] > stream_out[x]) && (answer[x] - stream_out[x] > 0.001) ) ||
	       ( (answer[x] < stream_out[x]) && (stream_out[x] - answer[x] > 0.001) ))
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

	#ifdef HLS
	#else
	free(weight);
	free(bias);
	free(stream_in);
	free(stream_out);
	free(answer);
	#endif

}
