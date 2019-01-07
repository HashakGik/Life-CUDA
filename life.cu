#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <fcntl.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
   Conway's Game of Life running on nVidia GPUs.

   Each GPU thread updates the state of a single cell in the playing field, then the current generation is
   displayed in Linux's framebuffer (this means the program must be run outside X Window System).
   Since this program is just a demonstration, the data model is NOT decoupled from its representation.
*/



/**
   Cell drawing function

   Updates the state of a single cell and draws it on screen.
   Each GPU thread updates a single cell (the mapping is linear, so even if the field's size exceeds the
   number of computation units of the GPU, the pixels assigned to the same thread are far away from each other),
   based on the number of neighbors.
   Since the number of neighbors is only read (and this function is called AFTER the neighbors array is written)
   and each thread writes only its cell, there are no race conditions.

   @param field Array of cell states. 1: live cell, 0: dead cell.
   @param neighbors Array of neighbors' count (e.g. neighbors[42] = 4 means cell 42 has 4 neighbors).
   @param screen Framebuffer (the function assumes the buffer uses 32 bit colors).
   @param w Width of the framebuffer.
   @param h Height of the framebuffer.
*/
__global__ void updateCell(char *field, char *neighbors, char *screen, int w, int h)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x; /* Pick the cell based on thread and block IDs. */
  
  if (i < w * h)
  {
    if ((field[i] != 0 && (neighbors[i] == 3 || neighbors[i] == 2)) || /* Survive rule: S23 */
	(field[i] == 0 && neighbors[i] == 3))                            /* Birth rule: B3    */
    {
      /* Live cell */
      field[i] = 1;
      screen[4 * i] =  0xff;     /* blue  */
      screen[4 * i + 1] =  0xff; /* green */
      screen[4 * i + 2] =  0xff; /* red   */
      screen[4 * i + 3] =  0xff; /* alpha */
    }
    else
    {
      /* Dead cell */
      field[i] = 0;
      screen[4 * i] =  0x00;
      screen[4 * i + 1] =  0x00;
      screen[4 * i + 2] =  0x00;
      screen[4 * i + 3] =  0x00;
    }
  }
}

/**
   Neighbors counting function

   Checks the eight neighbors' state and counts the live ones.
   Each GPU thread writes the neighbors count for a single cell while the field is not modified, so there
   are no race conditions.

   @param field Array of cell states. 1: live cell, 0: dead cell.
   @param neighbors Array of neighbors' count (e.g. neighbors[42] = 4 means cell 42 has 4 neighbors).
   @param w Width of the framebuffer.
   @param h Height of the framebuffer.
 */
__global__ void countNeighbors(char *field, char *neighbors, int w, int h)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int x = i % w;
  int y = i / w * w; /* Rounding to the closest multiple of w. */
  int xp1 = (i + 1) % w;
  int xm1 = (i + w - 1) % w;
  int yp1 = ((i + w) % (h * w)) / w * w;
  int ym1 = ((i - w + h * w) % (h * w)) / w * w;
   
  neighbors[i] = 0;
  if (field[xm1 + ym1] != 0)
    neighbors[i]++;
  if (field[xm1 + y] != 0)
    neighbors[i]++;
  if (field[xm1 + yp1] != 0)
    neighbors[i]++;
  if (field[x + ym1] != 0)
    neighbors[i]++;
  if (field[x + yp1] != 0)
    neighbors[i]++;
  if (field[xp1 + ym1] != 0)
    neighbors[i]++;
  if (field[xp1 + y] != 0)
    neighbors[i]++;
  if (field[xp1 + yp1] != 0)
    neighbors[i]++;
}

/**
   Game of Life wrapper function.

   Initializes the playing field with a random initial state and iterates over Life's generations.
   This function takes care of synchronizing GPU threads so that no race conditions are triggered.

   @param screen Framebuffer.
   @param w Width of the framebuffer.
   @param h Height of the framebuffer.
 */
void life(char *screen, int w, int h)
{
  char field[w * h];
  
  /* Generate a random initial playing field. */
  srand(time(NULL));
  for (int i = 0; i < w * h; i++)
    field[i] = (rand() % 100 < 75)? 1: 0;

  /* Find the maximum number of threads for the current GPU. */
  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  char *d_field, *d_screen, *d_neighbors;
  int maxthds = properties.maxThreadsPerBlock;

  /*
     Copy memory to the GPU's internal RAM. Since this process is slow, it's done only once and the state
     of the playing field is effectively kept in GPU's memory, updating the main RAM only to display the
     result on screen.
  */
  cudaMalloc((void **) &d_field, sizeof(char) * w * h);
  cudaMalloc((void **) &d_neighbors, sizeof(char) * w * h);
  cudaMalloc((void **) &d_screen, sizeof(char) * w * h * 4);
  cudaMemcpy(d_field, field, sizeof(char) * w * h, cudaMemcpyHostToDevice);
  cudaMemcpy(d_screen, screen, sizeof(char) * w * h * 4, cudaMemcpyHostToDevice);
  
  
  while (1) /* Maybe a termination condition would have been more elegant... */
  {
    /*
       Count the neighbors, synchronize the threads (in order to avoid race conditions),
       then update the state of each cell.
    */
    countNeighbors<<<(w * h + maxthds - 1) / maxthds, maxthds>>>(d_field, d_neighbors, w, h);
    cudaDeviceSynchronize();
    updateCell<<<(w * h + maxthds - 1) / maxthds, maxthds>>>(d_field, d_neighbors, d_screen, w, h);

    /*
       Synchronize the threads again and finally copy the framebuffer from GPU's internal memory to main RAM
       (and therefore display the result).
    */
    cudaDeviceSynchronize();
    cudaMemcpy(screen, d_screen, sizeof(char) * w * h * 4, cudaMemcpyDeviceToHost);
  }
  
  
  /* Free the GPU's internal memory. Since the loop condition is never false, this cleanup is never performed. */
  cudaFree(d_field);
  cudaFree(d_neighbors);
  cudaFree(d_screen);
  
  /* printf("%s\n", cudaGetErrorString(cudaGetLastError())); Debug message. */
}


/* Semplice funzione main che controlla il framebuffer. Copiata da: https://stackoverflow.com/a/1830865

/**
   Program entrypoint.

   Initializes the main framebuffer (/dev/fb0), as suggested on: https://stackoverflow.com/a/1830865
   then passes it to the life() function.

   @return 0 if no error occurred, 1 otherwise.
 */
int main()
{
  struct fb_var_screeninfo screen_info;
  struct fb_fix_screeninfo fixed_info;
  char *buffer = NULL;
  size_t buflen;
  int fd = -1;
  int r = 1;

  fd = open("/dev/fb0", O_RDWR);
  if (fd >= 0)
  {
    if (!ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) &&
        !ioctl(fd, FBIOGET_FSCREENINFO, &fixed_info))
    {
      buflen = screen_info.yres_virtual * fixed_info.line_length;
      buffer = (char *) mmap(NULL, buflen,
                       PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
      if (buffer != MAP_FAILED)
      {
	life(buffer, screen_info.xres_virtual, screen_info.yres_virtual); /* Start the game. */
	r = 0;
      }
      else
      {
	perror("mmap");
      }
    }
    else
      {
	perror("ioctl");
      }
   }
   else
   {
       perror("open");
   }

   if (buffer && buffer != MAP_FAILED)
     munmap(buffer, buflen);
   if (fd >= 0)
     close(fd);

   return r;
}
