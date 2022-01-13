#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <cstring>

typedef float DTYPE;

void load_from_txt( DTYPE *stream_in, int* parameters, const char* path){

  FILE *fp;
  printf("TXT PATH: %s\n", path);

  if( (fp = fopen( path, "r")) == NULL){
    fprintf(stderr, "CAN'T OPEN DATA FILE\n");
    exit(-1);
  }

  for( int och = 0; och < parameters[0]; och++){
    for( int inch = 0; inch < parameters[1]; inch++){
      for( int y = 0; y < parameters[2]; y++){
        for( int x = 0; x < parameters[3]; x++){
          DTYPE imgval;
          fscanf( fp, "%f", &imgval);
          stream_in[parameters[1]*parameters[2]*parameters[3]*och  +
          parameters[2]*parameters[3]*inch +
          parameters[3]*y    +x] = imgval;
        }
      }
    }
  }
  fclose(fp);
}
