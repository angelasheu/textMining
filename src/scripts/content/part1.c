#include<stdio.h>
#include<omp.h>
#include<string.h>
#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

void insertMatrixValues(float* pad, float* in, int total_pad_x, int kernel_padside, int data_size_X, int data_size_Y) {
    // total_pad_x = Total padding for row
    // kernel_padside = Total padding due to kernel overflow on one side of the matrix 
    int start_x = kernel_padside + kernel_padside*(data_size_X + total_pad_x);
    int offset = data_size_X + total_pad_x;

    for (int y = 0; y < data_size_Y; y++) {
        memcpy(pad + start_x + y * offset, in + y*data_size_X, sizeof(float)*data_size_X);
    }
/*
    for (int y = 0; y < data_size_Y; y++) {
        for (int x = 0; x < data_size_X; x++) {
            pad[start_x + x + y * offset] = in[x + y*data_size_X];
        }
    }*/
}

void insertKernelValues(float* pad, float* kernel, int pad_width) {
    for (int y = 0; y < KERNY; y++) {
        for (int x = 0; x < KERNX; x++) {
            pad[x + y * (pad_width)] = kernel[x + y*KERNX]; 
        }
    }
}


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int size_kernel = KERNX * KERNY;
    float* flipped = (float*)calloc(size_kernel, sizeof(float));
    for (int i = 0; i < size_kernel; i++) {
        *(flipped + (size_kernel - 1) - i) = *(kernel+ i);
    }
    kernel = flipped;

   int pdata_size_X = (data_size_X + 3) & ~0x03;
   int pdata_size_Y = (data_size_Y + 3) & ~0x03;
   int matrix_size = (pdata_size_X + KERNX/2*2) * (pdata_size_Y + KERNY/2*2); 
   int total_pad_x = pdata_size_X - data_size_X + KERNX/2*2;
   float* padded = (float*)calloc(matrix_size, sizeof(float));
   int start_x = KERNX/2 + KERNX/2*(data_size_X + total_pad_x); 
   insertMatrixValues(padded, in, pdata_size_X - data_size_X + KERNX/2*2, KERNX/2, data_size_X, data_size_Y);
   in = padded;
   __m128 in_vector;
   __m128 in_vector2;
   __m128 in_vector3;
   __m128 in_vector4;

   __m128 kern_vector;

   __m128 out_vector;
   __m128 out_vector2;
   __m128 out_vector3;
   __m128 out_vector4;


   int dest_index;
   int dest_index2;
   int dest_index3;
   int dest_index4;

   int x_offset = data_size_X + total_pad_x;
   int yj_sum;
   int kern_offset;
   int round_down = data_size_X >= 0 ? (data_size_X / 4) * 4 : ((data_size_X - 4 + 1) / 4) * 4;


   for(int y = 0; y < data_size_Y; y++){ 
		for(int x = 0; x < data_size_X/16*16; x+=16){
            dest_index = x + y*data_size_X;
            dest_index2 = x+4 + y*data_size_X;
            dest_index3 = x+8 + y*data_size_X;
            dest_index4 = x+12 + y*data_size_X;

            out_vector = _mm_setzero_ps();
            out_vector2 = _mm_setzero_ps();
            out_vector3 = _mm_setzero_ps();
            out_vector4 = _mm_setzero_ps();
			for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ 
                yj_sum = y+j;
                kern_offset = (j+kern_cent_Y)*KERNX;
				for(int i = -kern_cent_X; i <= kern_cent_X; i++){ 
                    kern_vector = _mm_set1_ps(*(kernel + (i+kern_cent_X) + kern_offset)); 

                    in_vector = _mm_loadu_ps(in + (x+i+start_x) + yj_sum*(x_offset)); 
                    in_vector2 = _mm_loadu_ps(in + (x+i+4+start_x) + yj_sum*(data_size_X + total_pad_x)); 
                    in_vector3 = _mm_loadu_ps(in + (x+i+8+start_x) + yj_sum*(data_size_X + total_pad_x)); 
                    in_vector4 = _mm_loadu_ps(in + (x+i+12+start_x) + yj_sum*(data_size_X + total_pad_x)); 
                    
                    out_vector = _mm_add_ps(out_vector, _mm_mul_ps(kern_vector, in_vector));
                    out_vector2 = _mm_add_ps(out_vector2, _mm_mul_ps(kern_vector, in_vector2));
                    out_vector3 = _mm_add_ps(out_vector3, _mm_mul_ps(kern_vector, in_vector3));
                    out_vector4 = _mm_add_ps(out_vector4, _mm_mul_ps(kern_vector, in_vector4));
				}
			}
            _mm_storeu_ps(out + dest_index, out_vector);
            _mm_storeu_ps(out + dest_index2, out_vector2);
            _mm_storeu_ps(out + dest_index3, out_vector3);
            _mm_storeu_ps(out + dest_index4, out_vector4);
		}
      for (int x = data_size_X/16*16; x < data_size_X/4*4; x+=4) {
            dest_index = x + y*data_size_X;

            out_vector = _mm_setzero_ps();
			for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ 
                yj_sum = y+j;
                kern_offset = (j+kern_cent_Y)*KERNX;
				for(int i = -kern_cent_X; i <= kern_cent_X; i++){ 
                    kern_vector = _mm_set1_ps(*(kernel + (i+kern_cent_X) + kern_offset)); 
                    in_vector = _mm_loadu_ps(in + (x+i+start_x) + yj_sum*(x_offset)); 
                    out_vector = _mm_add_ps(out_vector, _mm_mul_ps(kern_vector, in_vector));
				}
			}
            _mm_storeu_ps(out + dest_index, out_vector);
      }
      for (int x = data_size_X/4*4; x < data_size_X; x++) {
          dest_index = x + y*data_size_X;
          for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
              kern_offset = (j+kern_cent_Y)*KERNX;
              for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
                  out[dest_index] += kernel[(i+kern_cent_X) + kern_offset] * in[(x+i+start_x) + (y+j)*(x_offset)];
              }
          }
      }
	}
    free(flipped);
    free(padded);
	return 1; 
}