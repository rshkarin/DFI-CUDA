#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <tiffio.h>
#include <cufft.h>

#define PI 3.14159265358979323846

//CUDA_LAUNCH_BLOCKING=0 ./dfi
//CUDA_VISIBLE_DEVICES="0,1,2"

int tiff_decode_complex(TIFF *tif, cufftComplex *buffer, int offset)
{
  const int strip_size = TIFFStripSize(tif);
  const int n_strips = TIFFNumberOfStrips(tif);

  printf("\nstrip_size:%d ; n_strips:%d\n", strip_size, n_strips);
 
  int result = 0;
  
  int width, height;
  
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  
  float *temp_buffer = (float *)malloc(strip_size);
  
  for (int strip = 0; strip < n_strips; strip++) {
    result = TIFFReadEncodedStrip(tif, strip, temp_buffer, strip_size);

    if (result == -1)
      return 0;

    int pixels_per_strip = strip_size / sizeof(float);
    int rows = pixels_per_strip / width;

    for(int i = 0; i < rows; ++i) {
      for(int j = offset; j < width; ++j) {
	buffer[strip * (pixels_per_strip - rows * offset) + i * (width-offset) + j - offset].x = temp_buffer[i * width + j];
	buffer[strip * (pixels_per_strip - rows * offset) + i * (width-offset) + j - offset].y = 0.0;
      }
    }
  }
  
  return 1;
}

cufftComplex *tiff_read_complex(const char *filename,
				int center_pos,
                                int *bits_per_sample,
                                int *samples_per_pixel,
                                int *width,
                                int *height)
{
  TIFF *image;
  int *bytes_per_strip;
  
  // Create the TIFF file
  if((image = TIFFOpen(filename, "r")) == NULL){
    printf("Could not open %s for reading\n", filename);
    return NULL;
  }

  // Get TIFF Attributes
  TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
  TIFFGetField(image, TIFFTAG_IMAGEWIDTH, width);
  TIFFGetField(image, TIFFTAG_IMAGELENGTH, height);

  if (*samples_per_pixel > 1) {
    printf("%s has %i samples per pixel (%i bps)",
           filename, *samples_per_pixel, *bits_per_sample);
  }

  int offset = 0;
  if (center_pos != -1) {
    offset = *width - (*width - center_pos) * 2;
    *width = (*width - center_pos) * 2;
  }

  cufftComplex *buffer =
  (cufftComplex *)malloc((*height) * (*width) * sizeof(cufftComplex));
  
  if (!tiff_decode_complex(image, buffer, offset)) {
    goto error_close;
  }
  
  TIFFClose(image);
  return buffer;
  
error_close:
  TIFFClose(image);
  return NULL;
}

void tiff_write_complex(const char *filename, cufftComplex *data, int width, int height)
{
  TIFF *image;
  
  // Open the TIFF file
  if((image = TIFFOpen(filename, "w")) == NULL){
    printf("Could not open output.tif for writing\n");
    return;
  }
  
  // Set TIFF Attributes
  TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, 1);
  
  // Scanline sizew
  int scanlineSize = TIFFScanlineSize(image);
  
  for (int y = 0; y < height; y++) {
    float *vec_data = (float *)malloc(scanlineSize);
    
    for (int x = 0; x < width; x++) {
      vec_data[x] = fabsf((float)data[x + y * width].x);
    }

    TIFFWriteScanline(image, vec_data, y, 0);
  }
  
  TIFFClose(image);
}

__global__ void kernel_zero_ifftshift(cufftComplex *input,
                                      int in_rows,
                                      int in_cols,
                                      cufftComplex *output,
                                      int out_rows,
                                      int out_cols)
{
  unsigned long elemID = (blockIdx.x * blockDim.x + threadIdx.x);
  int in_half_size = floor(in_cols/2.);
  int cur_row = ceil((float)(elemID + 1) /
                     ((in_cols & 1)?(in_half_size + 1):(in_half_size)));
  
  unsigned long index = elemID + in_half_size * cur_row;
  
  if ((index < in_rows * in_cols) && (index >= 0)) {
    unsigned long first_index = (cur_row - 1) * in_cols;
    unsigned long last_index = (cur_row * in_cols) - 1;
    
    unsigned long first_zero_index = (cur_row - 1) * out_cols;
    unsigned long last_zero_index = (cur_row * out_cols) - 1;
    
    int inner_index = index - first_index;

    output[first_zero_index + inner_index - in_half_size] = input[index];
    
    if (((index != last_index) && (in_cols & 1)) || !(in_cols & 1)) {
      output[last_zero_index - (in_half_size - 1) + (inner_index - in_half_size)] =
      input[first_index + inner_index - in_half_size];
    }
  }
}

__global__ void kernel_fftshift(cufftComplex *input,
                                int in_rows,
                                int in_cols,
                                cufftComplex *output,
                                int out_rows,
                                int out_cols)
{
  unsigned long elemID = (blockIdx.x * blockDim.x + threadIdx.x);
  int in_half_size = floor(in_cols/2.);
  int cur_row = floor((float)(elemID) /
                      ((in_cols & 1)?(in_half_size + 1):(in_half_size)));
  
  unsigned long index = elemID + in_half_size * cur_row;
  
  if ((index < in_rows * in_cols) && (index >= 0)) {
    unsigned long first_index = cur_row * in_cols;
    unsigned long last_index = ((cur_row + 1) * in_cols) - 1;
    
    unsigned long first_zero_index = cur_row * out_cols;
    unsigned long last_zero_index = ((cur_row + 1) * out_cols) - 1;
    
    int inner_index = index - first_index;
    
    output[last_zero_index - ((in_cols & 1)?(in_half_size):(in_half_size - 1)) +
           inner_index] = input[index];
    
    if (((inner_index != in_half_size) &&
         (in_cols & 1)) || !(in_cols & 1)) {
      output[first_zero_index + inner_index] =
      input[first_index + inner_index +
            ((in_cols & 1)?(in_half_size + 1):(in_half_size))];
    }
  }
}

/*
__global__ void linear_interp(cufftComplex *input, cufftComplex *output,
float norm_ratio,
float in_rows_first_val, float in_rows_last_val,
float in_cols_first_val, float in_cols_last_val,
int output_rows, int output_cols,
int input_rows, int input_cols)
{
long index = (blockIdx.x * blockDim.x + threadIdx.x) - 1;
if (index < output_cols * output_rows) {

// indices of cartesian projection plane
float cart_row = index/(float)output_cols;
float cart_col = index%output_cols;

//range values along interp_cols and interp_rows
float cartesian_center = output_cols/2.;

//range values along interp_cols and interp_rows
float cartesian_x = (cartesian_center - cart_col) * norm_ratio;
float cartesian_y = (cartesian_center - cart_row) * norm_ratio;

//get angle in radians
float atan_val = atan2f(cartesian_y, cartesian_x);
//transform from radians to degree
float deg = atan_val * (in_rows_last_val/PI);
//since, negative angle cannot be used for index calculations
float theta_value = (deg < 0) ? in_rows_last_val + deg : deg;

//get angle on source image and check the border values
float angular_index = theta_value;
if (angular_index < 1 || angular_index > input_rows) {
angular_index = 1;
}
//get hypotenuse, if hypotenuse > PI then go next iteration
float sqrt_value = sqrt(pow(cartesian_x, 2) + pow(cartesian_y, 2));
if (sqrt_value <= PI) {
//set the sign of hypotenuse
sqrt_value = sqrt_value * ((atan_val > 0)?1:((atan_val < 0)?-1:0));
//get radius on source image and check the border values
float sino_normalized_center = (input_cols/2.0) * norm_ratio;
float radius_index = ((sqrt_value + sino_normalized_center) / norm_ratio) - 1;
if (radius_index < 1 || radius_index > input_cols) {
radius_index = 1;
}
//calculate index on source image
long source_index = (input_cols * floorf(angular_index) + floorf(radius_index));

//checking border values on indices
if (angular_index == input_rows) {
angular_index = angular_index + 1;
source_index = source_index - input_rows;
}
float angular_ratio = (angular_index - floorf(angular_index));
if (radius_index == input_cols) {
radius_index = radius_index + 1;
source_index = source_index - 1;
}
float radius_ratio = (radius_index - floorf(radius_index));
//get indices of interpolation
int cur_idx = source_index - 1;
int cur_next_idx = cur_idx + 1;
int cur_max_idx = cur_idx + (input_cols + 1);
//get values at indices
float input_re_val = input[cur_idx].x;
float input_im_val = input[cur_idx].y;
float input_next_re_val = input[cur_next_idx].x;
float input_next_im_val = input[cur_next_idx].y;
float input_max_re_val = input[cur_max_idx].x;
float input_max_im_val = input[cur_max_idx].y;
//interpolate real and imaginary values
//output[index].x = input[cur_idx].x;
//output[index].y = input[cur_idx].y;

output[index].x =
(input_re_val * (1 - radius_ratio) + input_next_re_val * (radius_ratio)) * (1 - angular_ratio) +
(input_re_val * (1 - radius_ratio) + input_max_re_val * (radius_ratio)) * (angular_ratio);
output[index].y =
(input_im_val * (1 - radius_ratio) + input_next_im_val * (radius_ratio)) * (1 - angular_ratio) +
(input_im_val * (1 - radius_ratio) + input_max_im_val * (radius_ratio)) * (angular_ratio);

}
}
}
*/

__global__ void linear_interp(cufftComplex *input, cufftComplex *output,
float norm_ratio,
                              float in_rows_first_val, float in_rows_last_val,
                              float in_cols_first_val, float in_cols_last_val,
int interp_rows, int interp_cols,
                              int in_rows, int in_cols)
{
  long index = (blockIdx.x * blockDim.x + threadIdx.x) - 1;

  if (index < interp_cols * interp_rows) {
    int cur_row = ceil(((float)index)/interp_cols); /* row s0 */
    int cur_col = index - cur_row * interp_cols; /* col s1*/

    //range values along interp_cols and interp_rows
    float omega_y =
    (interp_rows/2. * norm_ratio) -
    ((cur_row*interp_cols + cur_col)*norm_ratio/interp_cols);

    float omega_x =(((cur_row*interp_cols + cur_col)%interp_cols)*norm_ratio) -
    (interp_rows/2. * norm_ratio);

    //get angle in radians
    float atan_val = atan2f(omega_y, omega_x);

    //transfrom from radiangs to degree a - n * floor(a / n);
    float a = (atan_val * (in_rows_last_val/PI));
    float b = in_rows_last_val * floorf(a/in_rows_last_val);
    float theta_i = a - b;

    //get angle on source image and check the border values
    float s_val = 1 + ((theta_i - in_rows_first_val)/(in_rows_last_val - in_rows_first_val)) * (in_rows - 1);

    if (s_val < 1 || s_val > in_rows) {
      s_val = 1;
    }

    //get hypotenuse, if hypotenuse > PI then go next iteration
    float sqrt_val = sqrt(pow(omega_x, 2) + pow(omega_y, 2));

    if (sqrt_val <= PI) {
      //set the sign of hypotenuse
      float omega_si = sqrt_val * ((atan_val > 0)?1:((atan_val < 0)?-1:1));

      //get radius on source image and check the border values
      //float t_val = 1 +
      //((omega_si - ((-((float)in_cols)/2)*norm_ratio/dx))/
      // ((((((float)in_cols)/2) - 1)*norm_ratio/dx) -
      // ((-((float)in_cols)/2)*norm_ratio/dx))) * (in_cols - 1);
      float half_sin_height = (float)in_cols / 2;
      float t_val = 1 + (omega_si + half_sin_height * norm_ratio)/(((half_sin_height - 1) * norm_ratio) + half_sin_height * norm_ratio) * (in_cols - 1);

      if (t_val < 1 || t_val > in_cols) {
        t_val = 1;
      }

      //calculate index on source image
      long source_index = (floorf(t_val) + floorf(s_val - 1) * in_cols);

      //checking border values on indices
      if (s_val == in_cols) {
        s_val = s_val + 1;
        source_index = source_index - in_rows;
      }
      float s = (s_val - floorf(s_val));

      if (t_val == in_rows) {
        t_val = t_val + 1;
        source_index = source_index - 1;
      }
      float t = (t_val - floorf(t_val));

      //get indices of interpolation
      int cur_idx = source_index - 1;
      int cur_next_idx = source_index + 1 - 1; //ndx + 1
      int cur_max_idx = source_index +
      (in_cols + 1) - 1; //ndx + (nrows + 1)

      //get values at indices
      float input_re_val = input[cur_idx].x;
      float input_im_val = input[cur_idx].y;

      float input_next_re_val = input[cur_next_idx].x;
      float input_next_im_val = input[cur_next_idx].y;

      float input_max_re_val = input[cur_max_idx].x;
      float input_max_im_val = input[cur_max_idx].y;

      //interpolate real and imaginary values
      output[index].x =
      (input_re_val * (1 - t) + input_next_re_val * (t)) * (1 - s) +
      (input_re_val * (1 - t) + input_max_re_val * (t)) * (s);

      output[index].y =
      (input_im_val * (1 - t) + input_next_im_val * (t)) * (1 - s) +
      (input_im_val * (1 - t) + input_max_im_val * (t)) * (s);
    }
  }
}


__global__ void crop_data(cufftComplex *input, cufftComplex *output,
                          int crop_side_length, int original_side_length, float lt_offset, float rb_offset)
{
  long index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= 0 && index < crop_side_length * crop_side_length) {
    long in_index = original_side_length * lt_offset +
                    lt_offset +
                    (ceil(index/(float)crop_side_length))*(crop_side_length + lt_offset + rb_offset) +
                    index%crop_side_length;

    output[index].x = input[in_index].x;
    output[index].y = input[in_index].y;
  }
}

__global__ void shift2d(cufftComplex *input,
                        cufftComplex *output,
                        int rows,
                        int cols,
                        int isInverse)
{
  long index = blockIdx.x * blockDim.x + threadIdx.x;
  if ((index < cols * rows) && (index >= 0)) {
    int cur_row = (int)(index/cols);
    int cur_col = index - cur_row * cols;
    
    int p_rows = (isInverse)?floorf(rows/2.):ceilf(rows/2.);
    int p_cols = (isInverse)?floorf(cols/2.):ceilf(cols/2.);
    
    int p_rows_offset = (rows & 1)?(rows - p_rows):p_rows;
    int p_cols_offset = (cols & 1)?(cols - p_cols):p_cols;
    
    int x_out = (cur_col > (p_cols - 1))?(cur_col - p_cols):
    (p_cols_offset + cur_col);
    int y_out = (cur_row > (p_rows - 1))?(cur_row - p_rows):
    (p_rows_offset + cur_row);
    
    output[x_out + y_out * cols] = input[index];
  }
}

void dfi_process_sinogram(const char *tiff_input, const char *tiff_output, int center_rotation)
{
  cudaSetDevice(1);

  if(!tiff_input) {
    printf("The filename of input is not valid. (pointer tiff_input = %p)", tiff_input);
    return;
  }
  
  if(!tiff_output) {
    printf("The filename of output is not valid. (pointer tiff_output = %p)", tiff_output);
    return;
  }
  
  ///////////////
  /* PREPARING */
  ///////////////
  
  /* Input a slice properties */
  int bits_per_sample;
  int samples_per_pixel;
  int theta_size;
  int slice_size;
  
  /* Read the slice */
  cufftComplex *data_tiff = tiff_read_complex(tiff_input,
					      center_rotation,
                                              &bits_per_sample,
                                              &samples_per_pixel,
                                              &slice_size,
                                              &theta_size);

  printf("\nFile:%s , theta_size:%d , slice_size:%d , bits_per_sample:%d , samples_per_pixel:%d\n", tiff_input, theta_size, slice_size, bits_per_sample, samples_per_pixel);
  
  cufftComplex *d_data_tiff;
  cudaMalloc((void**)&d_data_tiff,
             slice_size * theta_size * sizeof(cufftComplex));
  cudaMemcpy(d_data_tiff,
             data_tiff,
             slice_size * theta_size * sizeof(cufftComplex),
             cudaMemcpyHostToDevice);

  /*
//write original data
tiff_write_complex("resources/initial-sinogram.tif", data_tiff, slice_size, theta_size);
*/
  
  ////////////////////////
  /* DATA PREPROCESSING */
  ////////////////////////
  
  /* Start timer */
  timeval global_tim;
  gettimeofday(&global_tim, NULL);
  double t1_global = global_tim.tv_sec + (global_tim.tv_usec/1000000.0), t2_global = 0.0;

  /* Reconstruction properties */
  int min_theta = 0;
  int max_theta = theta_size - 1;
  int size_zeropad_s = pow(2, ceil(log2((float)slice_size)));
  float d_omega_s = 2 * PI / size_zeropad_s; //normalized ratio [0; 2PI]
  int size_s = size_zeropad_s;



  timeval zeropad_shift_tim;
  gettimeofday(&zeropad_shift_tim, NULL);
  double t1_zeropad_shift = zeropad_shift_tim.tv_sec + (zeropad_shift_tim.tv_usec/1000000.0), t2_zeropad_shift = 0.0;

    /************************/
   /* Zeropad the sinogram */
  /************************/
  cufftComplex *d_zeropad_sinogram;
  cudaMalloc((void**)&d_zeropad_sinogram,
             size_zeropad_s * theta_size * sizeof(cufftComplex));
  cudaMemset(d_zeropad_sinogram, 0, size_zeropad_s * theta_size * sizeof(cufftComplex));

  int nThreads = ceil(slice_size/2.) * theta_size;
  dim3 blockSize(ceil(slice_size/2.), 1, 1);
  dim3 gridSize(ceil(nThreads/blockSize.x),1,1);

  kernel_zero_ifftshift<<<gridSize, blockSize>>>(d_data_tiff, theta_size, slice_size,
                                                 d_zeropad_sinogram, theta_size, size_zeropad_s);

  gettimeofday(&zeropad_shift_tim, NULL);
  t2_zeropad_shift = zeropad_shift_tim.tv_sec+(zeropad_shift_tim.tv_usec/1000000.0);
  printf("\n(Zeropad + Shift) %.6lf seconds elapsed\n", t2_zeropad_shift-t1_zeropad_shift);

  /*
//write zeropadded data
cufftComplex *zeropadded_egg_data = (cufftComplex *)malloc(theta_size * size_zeropad_s * sizeof(cufftComplex));
cudaMemcpy(zeropadded_egg_data,
d_zeropad_sinogram, theta_size * size_zeropad_s * sizeof(cufftComplex),
cudaMemcpyDeviceToHost);
tiff_write_complex("resources/zeropad-sinogram.tif", zeropadded_egg_data, size_zeropad_s, theta_size);
*/

  timeval onedim_fft_tim;
  gettimeofday(&onedim_fft_tim, NULL);
  double t1_onedim_fft = onedim_fft_tim.tv_sec + (onedim_fft_tim.tv_usec/1000000.0), t2_onedim_fft = 0.0;

    /******************/
   /* Perform 1D FFT */
  /******************/
  cufftHandle plan;
  cufftPlan1d(&plan, size_zeropad_s, CUFFT_C2C, theta_size);
  cufftExecC2C(plan, d_zeropad_sinogram, d_zeropad_sinogram, CUFFT_FORWARD);

  gettimeofday(&onedim_fft_tim, NULL);
  t2_onedim_fft = onedim_fft_tim.tv_sec+(onedim_fft_tim.tv_usec/1000000.0);
  printf("\n(1D FFT) %.6lf seconds elapsed\n", t2_onedim_fft-t1_onedim_fft);

  /*
//write 1d fft data
cufftComplex *one_dim_fft_egg_data = (cufftComplex *)malloc(theta_size * size_zeropad_s * sizeof(cufftComplex));
cudaMemcpy(one_dim_fft_egg_data,
d_zeropad_sinogram, theta_size * size_zeropad_s * sizeof(cufftComplex),
cudaMemcpyDeviceToHost);
tiff_write_complex("resources/fourier-sinogram.tif", one_dim_fft_egg_data, size_zeropad_s, theta_size);
*/

  timeval onedim_shift_tim;
  gettimeofday(&onedim_shift_tim, NULL);
  double t1_onedim_shift = onedim_shift_tim.tv_sec + (onedim_shift_tim.tv_usec/1000000.0), t2_onedim_shift = 0.0;

    /*****************************/
   /* Perform 1D shift sinogram */
  /*****************************/
  cufftComplex *d_fourier_sinogram;
  cudaMalloc((void**)&d_fourier_sinogram,
             size_zeropad_s * theta_size * sizeof(cufftComplex));
  cudaMemset(d_fourier_sinogram, 0, size_zeropad_s * theta_size * sizeof(cufftComplex));

  nThreads = ceil(size_zeropad_s/2.) * theta_size;
  
  dim3 fourierBlockSize(ceil(size_zeropad_s/2.), 1, 1);
  dim3 fourierGridSize((nThreads % blockSize.x == 0)?
                       (nThreads/blockSize.x):
                       ((nThreads + blockSize.x - 1) / blockSize.x), 1, 1);
  
  kernel_fftshift<<<fourierGridSize, fourierBlockSize>>>(
                                                         d_zeropad_sinogram, theta_size, size_zeropad_s,
                                                         d_fourier_sinogram, theta_size, size_zeropad_s);

  gettimeofday(&onedim_shift_tim, NULL);
  t2_onedim_shift = onedim_shift_tim.tv_sec+(onedim_shift_tim.tv_usec/1000000.0);
  printf("\n(1D Fftshift) %.6lf seconds elapsed\n", t2_onedim_shift-t1_onedim_shift);

/*
//write fftshift data
cufftComplex *fftshift_egg_data = (cufftComplex *)malloc(theta_size * size_zeropad_s * sizeof(cufftComplex));
cudaMemcpy(fftshift_egg_data,
d_fourier_sinogram, theta_size * size_zeropad_s * sizeof(cufftComplex),
cudaMemcpyDeviceToHost);
tiff_write_complex("resources/shifted-fourier-sinogram.tif", fftshift_egg_data, size_zeropad_s, theta_size);
*/

  timeval interp_tim;
  gettimeofday(&interp_tim, NULL);
  double t1_interp = interp_tim.tv_sec + (interp_tim.tv_usec/1000000.0), t2_interp = 0.0;

  /**************************/
 /* Perform interpolation */
/**************************/
  float norm_ratio = d_omega_s;
  
  float in_rows_first_val = min_theta;
  float in_rows_last_val = max_theta;
  
  float in_cols_first_val = (-size_zeropad_s/2)*norm_ratio;
  float in_cols_last_val = (size_zeropad_s/2-1)*norm_ratio;
  
  int interp_rows = size_s;
  int interp_cols = interp_rows;
  
  cufftComplex *d_interp_result;
  cudaMalloc((void**)&d_interp_result,
             interp_cols * interp_rows * sizeof(cufftComplex));
  cudaMemset(d_interp_result, 0, interp_cols * interp_rows * sizeof(cufftComplex));
  
  nThreads = interp_cols * interp_rows;
  dim3 interpBlockSize(ceil(interp_cols/2.0), 1, 1);
  dim3 interpGridSize(ceil(nThreads/(float)interpBlockSize.x), 1, 1);
  
  linear_interp<<<interpGridSize, interpBlockSize>>>(d_fourier_sinogram, d_interp_result,
                                                     norm_ratio,
                                                     min_theta, max_theta,
                                                     in_cols_first_val, in_cols_last_val,
                                                     interp_rows, interp_cols,
                                                     theta_size, size_zeropad_s);

  gettimeofday(&interp_tim, NULL);
  t2_interp = interp_tim.tv_sec+(interp_tim.tv_usec/1000000.0);
  printf("\n(Interpolation) %.6lf seconds elapsed\n", t2_interp-t1_interp);

/*
// write interpolated data
cufftComplex *d_interp_result_data = (cufftComplex *)malloc(interp_rows * interp_cols * sizeof(cufftComplex));
cudaMemcpy(d_interp_result_data,
d_interp_result, interp_rows * interp_cols * sizeof(cufftComplex),
cudaMemcpyDeviceToHost);
tiff_write_complex("resources/interpolated-spectrum.tif", d_interp_result_data, interp_cols, interp_rows);
*/

  timeval twodim_ifft_tim;
  gettimeofday(&twodim_ifft_tim, NULL);
  double t1_twodim_ifft = twodim_ifft_tim.tv_sec + (twodim_ifft_tim.tv_usec/1000000.0), t2_twodim_ifft = 0.0;

    /********************/
   /* Perform 2D IFFT */
  /********************/
  cufftHandle plan2Difft;
  cufftPlan2d(&plan2Difft, interp_rows, interp_cols, CUFFT_C2C);
  cufftExecC2C(plan2Difft, d_interp_result, d_interp_result, CUFFT_INVERSE);

  gettimeofday(&twodim_ifft_tim, NULL);
  t2_twodim_ifft = twodim_ifft_tim.tv_sec+(twodim_ifft_tim.tv_usec/1000000.0);
  printf("\n(2D IFFT) %.6lf seconds elapsed\n", t2_twodim_ifft-t1_twodim_ifft);

/*
// write 2d-ifft data
cufftComplex *d_ifft_result_data = (cufftComplex *)malloc(interp_rows * interp_cols * sizeof(cufftComplex));
cudaMemcpy(d_ifft_result_data, d_interp_result, interp_rows * interp_cols * sizeof(cufftComplex),
cudaMemcpyDeviceToHost);
tiff_write_complex("resources/2d-invfft-spectrum.tif", d_ifft_result_data, interp_cols, interp_rows);
*/

  timeval twodim_shift_tim;
  gettimeofday(&twodim_shift_tim, NULL);
  double t1_twodim_shift = twodim_shift_tim.tv_sec + (twodim_shift_tim.tv_usec/1000000.0), t2_twodim_shift = 0.0;

    /********************/
   /* Perform 2D shift */
  /********************/
  cufftComplex *d_shifted_result;
  cudaMalloc((void**)&d_shifted_result,
             interp_cols * interp_rows * sizeof(cufftComplex));
  shift2d<<<interpGridSize, interpBlockSize>>>(d_interp_result, d_shifted_result, interp_rows, interp_cols, 0);

  gettimeofday(&twodim_shift_tim, NULL);
  t2_twodim_shift = twodim_shift_tim.tv_sec+(twodim_shift_tim.tv_usec/1000000.0);
  printf("\n(2D Shift) %.6lf seconds elapsed\n", t2_twodim_shift-t1_twodim_shift);

/*
// write 2d shifted data
cufftComplex *d_shifted_result_data = (cufftComplex *)malloc(interp_rows * interp_cols * sizeof(cufftComplex));
cudaMemcpy(d_shifted_result_data, d_shifted_result, interp_rows * interp_cols * sizeof(cufftComplex),
cudaMemcpyDeviceToHost);
tiff_write_complex("resources/2d-shifted-spectrum.tif", d_shifted_result_data, interp_cols, interp_rows);
*/

  timeval crop_tim;
  gettimeofday(&crop_tim, NULL);
  double t1_crop = crop_tim.tv_sec + (crop_tim.tv_usec/1000000.0), t2_crop = 0.0;

    /****************/
   /* Perform crop */
  /****************/
  cufftComplex *cropped_result;
  cudaMalloc((void**)&cropped_result, slice_size * slice_size * sizeof(cufftComplex));

  nThreads = slice_size * slice_size;
  dim3 cropBlockSize(ceil(slice_size/2.0), 1, 1);
  dim3 cropGridSize(ceil(nThreads/(float)ceil(slice_size/2.0)), 1, 1);

  float lt_offset = 0, rb_offset = 0;

  int dif_sides = interp_cols - slice_size;
  if (dif_sides%2) {
     lt_offset = floor(dif_sides / 2.0);
     rb_offset = ceil(dif_sides / 2.0);
  }
  else {
     lt_offset = rb_offset = dif_sides / 2.0;
  }

  crop_data<<<cropGridSize, cropBlockSize>>>(d_shifted_result, cropped_result, slice_size, interp_cols, lt_offset, rb_offset);

  gettimeofday(&crop_tim, NULL);
  t2_crop = crop_tim.tv_sec+(crop_tim.tv_usec/1000000.0);
  printf("\n(Crop target image) %.6lf seconds elapsed\n", t2_crop-t1_crop);


  /* Stop timer */
  gettimeofday(&global_tim, NULL);
  t2_global = global_tim.tv_sec+(global_tim.tv_usec/1000000.0);
  printf("\n(Total time) %.6lf seconds elapsed\n", t2_global-t1_global);

  //write crop output
  cufftComplex *h_crop_result =
  (cufftComplex *)malloc(slice_size * slice_size * sizeof(cufftComplex));
  cudaMemcpy(h_crop_result, cropped_result, slice_size * slice_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  
  tiff_write_complex(tiff_output, h_crop_result, slice_size, slice_size);
  printf("Last cuda errors: %s\n", cudaGetErrorString(cudaGetLastError()));
}

int main(int argc, char **argv) {
  dfi_process_sinogram("../resources/sino-egg.tif", "../resources/out-sino.tif", 461);
}
