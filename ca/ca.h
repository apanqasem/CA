/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void check_ca_conversion(DATA_TYPE *A, DATA_TYPE *ca, unsigned int items, int fields, int sparsity);
void convert_aos_to_ca(void *aos_data, DATA_TYPE *ca, unsigned items, int fields, int sparsity);

/* #ifndef TILESIZE */
/* #define TILESIZE 4 */
/* #endif */

#define TILE TILESIZE
