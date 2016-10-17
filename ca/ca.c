#include <stdio.h>
#include <ca.h>

unsigned int calc_tile_size(unsigned int items, int fields, int sparsity) {
  return TILE;
}
void check_ca_conversion(DATA_TYPE *A, DATA_TYPE *ca, unsigned int items, int fields, int sparsity) {

  fprintf(stderr, "checking CA conversion...\n");

  int tile = calc_tile_size(items, fields, sparsity);
  unsigned int items_per_ref_grp = items/sparsity;
  int num_ref_groups = 0;
  int base_top = 0;
  int base_bottom = 0;
  int tiles = 0;
  for (int i = 0, j = 0, k = 0; i < items; i++, k++) {
    int ca_index;
    if ((i != 0) && (i % tile == 0))
      tiles++; 
    if ((i != 0) && (i % items_per_ref_grp == 0)) {
      num_ref_groups++;
      j = 0; 
      k = 0;
      base_bottom = num_ref_groups;
    }

    if (k == num_ref_groups) {
      base_top = i;
    }    
    if (k < num_ref_groups) {
      ca_index = base_bottom + k * items_per_ref_grp; 
      ca_index = ca_index + (ca_index / tile) * (fields - 1) * tile ;
#ifdef DEBUG 
      printf ("[%d]->[%d]\t\t%3.2f\t%3.2f\n", i, ca_index, A[i], ca[ca_index]);
#endif
    }
    else {
      ca_index = base_top + j * items_per_ref_grp; 
      ca_index = ca_index + (ca_index / tile) * (fields - 1) * tile ;
#ifdef DEBUG 
      printf ("[%d]->[%d]\t\t%3.2f\t%3.2f\n", i, ca_index, A[i], ca[ca_index]);
#endif
      j++;
    }

    int errors = 0;
    if (A[i] != ca[ca_index]) {
      errors++;
      printf("AoS[%d] %3.2f \t CA[%d] %3.2f\n",  i, A[i], ca_index, ca[ca_index]);
    }
  } 
  return;
}

void convert_aos_to_ca(void *aos_data, DATA_TYPE *ca, unsigned int items, int fields, int sparsity) {


  int tile = calc_tile_size(items, fields, sparsity);
  unsigned int items_per_ref_grp = items/sparsity;
  unsigned int ref_group_reset = items_per_ref_grp/tile;

  int ref_grp_count = 0;
  for (int j = 0, t = 0; j < items * fields; j += tile * fields, t++)
    for (int i = 0, m = j; i < tile; i++, m++) {
      if (t == ref_group_reset) {
	ref_grp_count++;
	t = 0;
      }
      int aos_index = ((t * tile + i) * sparsity) + ref_grp_count;
      for (int p = 0; p < fields; p++) {
	ca[m + tile * p] = *((DATA_TYPE *) ((DATA_TYPE *) aos_data + (aos_index * fields)) + p);
      }
#ifdef DEBUG
      printf("%d\t%d\t%d\n", j, m, aos_index);
#endif
    }
}

#if 0

void print_da(DATA_TYPE *A, DATA_TYPE *B) {
  for (int i = 0; i < N * N; i += N) {
    for (int j = i; j < i + N; j++) {
      printf("%3.2f\t", A[j]);
    }
    printf("\n");
  }
  return;
}

void print_ca(DATA_TYPE *ca) {
  for (int i = 0; i < N * N * FIELDS; i += (TILE * FIELDS)) {
     for (int j = i; j < i + TILE; j++) {
       printf("[%d] %3.2f\t", i, ca[j]);
     }
     printf("\n");
  }
  return;
}

#endif
