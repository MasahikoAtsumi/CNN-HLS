#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>
#include <list>

#define line_row  7
#define line_col  28
typedef float DTYPE;

void line_load(DTYPE const *stream_in,
	       DTYPE **stream_out,
	       int d0_size,
	       int d1_size,
	       int *l_point,
	       int st_point)
{
  for(int id_d0 = 0; id_d0 < d0_size; id_d0++){
    // get line index
    for(int id_d1 = 0; id_d1 < d1_size; id_d1++){
      // stream_out[l_point][id_d1] = 0;
      stream_out[l_point[0]][id_d1] = stream_in[st_point + d1_size*id_d0 + id_d1];
    }
    // update stack
    if(l_point[0] == line_row-1){l_point[0] = 0;}
    else{l_point[0]++;}
  }
}
