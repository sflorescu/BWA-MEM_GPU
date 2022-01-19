#ifndef __SEED_GEN_H__
#define __SEED_GEN_H__


#include <vector_types.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include "../bwamem.h"

typedef uint32_t bwtint_t_gpu;

typedef struct {
	bwtint_t_gpu primary; // S^{-1}(0), or the primary index of BWT
	bwtint_t_gpu L2[5];
	bwtint_t_gpu seq_len; // sequence length
	bwtint_t_gpu bwt_size; // size of bwt, about seq_len/4
	uint32_t *bwt; // BWT
	int sa_intv;
	bwtint_t_gpu n_sa;
	bwtint_t_gpu *sa;
} bwt_t_gpu;

#ifdef __cplusplus
extern "C" {
#endif

mem_seed_v *seed_gpu(int argc, char **argv, int n_reads);

#ifdef __cplusplus
}
#endif

#endif