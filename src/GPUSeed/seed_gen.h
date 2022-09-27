#ifndef __SEED_GEN_H__
#define __SEED_GEN_H__

#include <stdbool.h>
#include"../bwa.h"

//typedef uint32_t bwtint_t_gpu;
typedef uint64_t bwtint_t_gpu;

/*typedef struct {
	bwtint_t_gpu primary; // S^{-1}(0), or the primary index of BWT
	bwtint_t_gpu L2[5];
	bwtint_t_gpu seq_len; // sequence length
	bwtint_t_gpu bwt_size; // size of bwt, about seq_len/4
	uint32_t *bwt; // BWT
	int sa_intv;
	bwtint_t_gpu n_sa;
	bwtint_t_gpu *sa;
} bwt_t_gpu;*/

typedef struct {
	bwtint_t_gpu primary; // S^{-1}(0), or the primary index of BWT
	bwtint_t_gpu *L2;
	bwtint_t_gpu seq_len; // sequence length
	bwtint_t_gpu bwt_size; // size of bwt, about seq_len/4
	uint32_t *bwt; // BWT
	int sa_intv;
	bwtint_t_gpu n_sa;
	// sa: 32 + (pack_size) bits
	uint32_t *sa;
	uint32_t *sa_upper_bits;
	uint8_t pack_size; // size of the pack of bits
} bwt_t_gpu;

typedef struct {
        int64_t offset;
        int32_t len;
        int32_t n_ambs;
        uint32_t gi;
        int32_t is_alt;
        char *name, *anno;
} bntann2_t;

typedef struct {
        int64_t offset;
        int32_t len;
        char amb;
} bntamb2_t;

typedef struct {
        int64_t l_pac;
        int32_t n_seqs;
        uint32_t seed;
        bntann2_t *anns; // n_seqs elements
        int32_t n_holes;
        bntamb2_t *ambs; // n_holes elements
        FILE *fp_pac;
} bntseq2_t;

typedef struct {
	int64_t rbeg;
	int32_t qbeg, len;
	int score;
} mem_seed_t; // unaligned memory

typedef struct { size_t n, m; mem_seed_t *a; int seed_counter; } mem_seed_v;

typedef struct {
	bwtint_t_gpu *rbeg;
	int2 *qbeg;
	uint32_t *score;
	uint32_t *n_ref_pos_fow_rev_results;
	uint32_t *n_ref_pos_fow_rev_prefix_sums;
	uint64_t file_bytes_skip;
} mem_seed_v_gpu;

typedef struct {
	char *read_file;
	char *query_file;
	bwt_t_gpu *bwt;
	bwt_t_gpu bwt_gpu;
	uint2 *pre_calc_seed_intervals;
	int pre_calc_seed_intervals_flag;
	int pre_calc_seed_len;
	int min_seed_size;
	int is_smem;
	uint64_t file_bytes_skip;	
} gpuseed_storage_vector;



#ifdef __cplusplus
extern "C" {
#endif

void bwt_destroy_gpu(bwt_t_gpu *bwt);
void bwt_restore_sa_gpu(const char *fn, bwt_t_gpu *bwt);
bwt_t_gpu *bwt_restore_bwt_gpu(const char *fn);
bwt_t_gpu gpu_cpy_wrapper(bwt_t_gpu *bwt);
void pre_calc_seed_intervals_wrapper(uint2 *pre_calc_seed_intervals, int pre_calc_seed_len, bwt_t_gpu bwt_gpu);
void free_gpuseed_data(gpuseed_storage_vector *gpuseed_data);
mem_seed_v_gpu *seed_gpu(gpuseed_storage_vector *gpuseed_data);

#ifdef __cplusplus
}
#endif

#endif