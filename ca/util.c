#include <time.h>
#include <sys/time.h>


double mysecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
