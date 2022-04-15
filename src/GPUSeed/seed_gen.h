#ifndef __SEED_GEN_H__
#define __SEED_GEN_H__

#include <stdbool.h>

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

typedef struct {
	int64_t rbeg;
	int32_t qbeg, len;
	int score;
} mem_seed_t; // unaligned memory

typedef struct { size_t n, m; mem_seed_t *a; int seed_counter; } mem_seed_v;

#ifdef __cplusplus
extern "C" {
#endif

void bwt_destroy_gpu(bwt_t_gpu *bwt);
void bwt_restore_sa_gpu(const char *fn, bwt_t_gpu *bwt);
bwt_t_gpu *bwt_restore_bwt_gpu(const char *fn);
mem_seed_v *seed_gpu(const char *read_file_name, int n_reads, int64_t n_processed, bwt_t_gpu *bwt);

#ifdef __cplusplus
}
#endif

#endif