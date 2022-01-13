#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

typedef float DTYPE;

void load_wb(
	DTYPE const *stream_in,
	DTYPE *stream_out,
	int data_size)
{
	for(int lw_id = 0; lw_id < data_size; lw_id++){
		stream_out[lw_id] = stream_in[lw_id];
	}
}
