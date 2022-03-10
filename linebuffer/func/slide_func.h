#include <stdint.h>
#include <algorithm>
#include <string>
#include <cstring>

// #define print

void slide_func(
	// static paremeter
	int k_size,
	int in_size,
	int out_size,
	int ochs,
	int inchs,
	int stride,
	int unroll_och,
	int iteration_och,
	int iteration_y,
	int unroll_x,
	int iteration_x,

	// dynamic parameter
	int  *dy_param, // 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
	bool *flag_list // 0:bias_flag, 1:end_flag, 2:acc_flag, 3:line_flag, 4:weight_flag
)
{
  if ( dy_param[5] == k_size - 1){
    dy_param[5] = 0; // flush kx

    if ( dy_param[4] == k_size - 1){
      dy_param[4] = 0; // flush ky

      if(  dy_param[0] >= (iteration_och - 1)*unroll_och){
	dy_param[0]  = 0; 
        flag_list[5] = 1;
	    
	if ( dy_param[7] >= (iteration_x - 1)*unroll_x){
	  dy_param[2] = 0;
	  dy_param[7] = 0;

	  if ( dy_param[6] >= (iteration_y - 1) ){
	    dy_param[1] = 0;
	    dy_param[6] = 0;
	    flag_list[3] = 1;
	    
	    if ( dy_param[3] == inchs - 1){
	      dy_param[3]  = 0; // flush inch
	      flag_list[1] = 1; // end_flag
	    }
	    else{
	      dy_param[3]++; // inch++
	      flag_list[4] = 1; // weight_flag
	    }
	  }
	 
	  else{
	    dy_param[1] += stride; // y+=stride
	    dy_param[6]++; // dst_y++
	    flag_list[3] = 1; // line_flag
	  }
	}
	else{
	  dy_param[2]+= stride*unroll_x; // x+=stride
	  dy_param[7]+= unroll_x; // dst_x
	}
      }
      else{
	dy_param[0]+=unroll_och; // och++
      }
    }
    else{
      dy_param[4]++; // ky++
    }
  }
  else{
    dy_param[5]++; // kx++
  }

  if(dy_param[4] == k_size - 1 && dy_param[5] == k_size - 1){flag_list[2] = 1;} // acc flag
  // if(dy_param[3] == inchs - 1  && dy_param[4] == k_size - 1 && dy_param[5] == k_size - 1){flag_list[0] = 1;}  // bias flag

  // 0:och,1:in_y, 2:in_x, 3:inch, 4:ky, 5:kx, 6:dst_y, 7:dst_x
  #ifdef print
  printf("och:   %d\n", dy_param[0]);
  printf("in y:  %d\n", dy_param[1]);
  printf("in x:  %d\n", dy_param[2]);
  printf("inch:  %d\n", dy_param[3]);
  printf("ky:    %d\n", dy_param[4]);
  printf("kx:    %d\n", dy_param[5]);
  printf("dst y: %d\n", dy_param[6]);
  printf("dst x: %d\n", dy_param[7]);
  printf("end  : %d\n", flag_list[1]);
  printf("\n");
  #endif
}
