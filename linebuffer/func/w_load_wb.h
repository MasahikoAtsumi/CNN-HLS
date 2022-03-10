#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

typedef float DTYPE;

void w_load_wb(
	       DTYPE const *stream_in,
	       DTYPE **stream_out,
	       int d0_size,
	       int d1_size,
	       int inchs,
	       int inch)
{

  for(int id_d0 = 0; id_d0 < d0_size; id_d0++){
    for(int id_d1 = 0; id_d1 < d1_size; id_d1++){
      stream_out[id_d0][id_d1] = stream_in[( id_d0*inchs + inch )*d1_size + id_d1];
    }
  }

}
