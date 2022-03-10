#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>
#include <list>

#define line_row  7
#define line_col  28
typedef float DTYPE;

void out_line_load(
		   DTYPE *stream_in,
		   DTYPE **stream_out,
		   int d0_size,
		   int d1_size,
		   int target_row,
		   int target_col,
                   int out_size)
{
  for(int id_d0 = 0; id_d0 < d0_size; id_d0++){
    for(int id_d1 = 0; id_d1 < d1_size; id_d1++){
      stream_out[id_d0][id_d1] = stream_in[( (id_d0)*out_size + target_row)*out_size + target_col + id_d1];
    }
  }
}
