/* The MIT License

   Copyright (c) 2008 Genome Research Ltd (GRL).

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* Contact: Heng Li <lh3@sanger.ac.uk> */

#ifndef BWT_BNTSEQ_H
#define BWT_BNTSEQ_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <zlib.h>
#include "../GASAL2/include/gasal.h"
#include "../GASAL2/include/args_parser.h"
#include "../GASAL2/include/host_batch.h"
#include "../GASAL2/include/gasal_align.h"
#include "../GASAL2/include/ctors.h"
#include "../GASAL2/include/interfaces.h"

#ifndef BWA_UBYTE
#define BWA_UBYTE
typedef uint8_t ubyte_t;
#endif

typedef struct {
	int64_t offset;
	int32_t len;
	int32_t n_ambs;
	uint32_t gi;
	int32_t is_alt;
	char *name, *anno;
} bntann1_t;

typedef struct {
	int64_t offset;
	int32_t len;
	char amb;
} bntamb1_t;

typedef struct {
	int64_t l_pac;
	int32_t n_seqs;
	uint32_t seed;
	bntann1_t *anns; // n_seqs elements
	int32_t n_holes;
	bntamb1_t *ambs; // n_holes elements
	FILE *fp_pac;
} bntseq_t;

extern unsigned char nst_nt4_table[256];

typedef struct {
		gasal_gpu_storage_t *gpu_storage;
		int batch_size;
		int batch_start;
		int id;
		int is_active;
		int no_extend;
		//int32_t *max_score, *read_start, *read_end, *ref_start, *ref_end;
		int n_query_batch, n_target_batch, n_seqs;
}gpu_batch;

/* 
    In this data structure, all sequences that must be aligned must be processed in two steps: Left alignment, and right alignment. 
    The best optimization would be to sort the alignment lengths, to process alignments with similar lengths in the same warp (avoid thread divergence)
    But doing so, one would need to sort either in advance, or to keep track of every sequence number.
    This would be needed to find back which sequence aligment is which (number, left/right) to sum the partial results.

    To keep the programming model simple, I decided to split the calculation into two batches: long, and short.
    When computing a seed, usually it's not centered, there's one side that is longer than the other.
    So keeping the sequences ordered, I can decided on which batch to put the "long" part, and I'll put the "short" one on the other.
    The difference between shortest and longest will be reduced by half. (At worst, a seed is in the middle, making both ends the same length).
    You could see it as a "free optimization" (at least, with very little cost in terms of bookkeeping).

    The good news is, I don't need to actually care which one is the left and which one is the right. I'll just add the partial results, period.

    Note that, for the moment, I can program it freely as "short=left" and "long=right" if needed. But the names would be confusing.
*/
/*
// unused. instantiate 2 gpu_batch instead.
typedef struct {
		gasal_gpu_storage_t *gpu_storage_short;
		gasal_gpu_storage_t *gpu_storage_long;
		int batch_size;
		int batch_start;
		int is_active;
		int no_extend;
		//int32_t *max_score, *read_start, *read_end, *ref_start, *ref_end;
		int n_query_batch, n_target_batch, n_seqs;
} gpu_batch_asym_t;
*/

#ifdef __cplusplus
extern "C" {
#endif

	void bns_dump(const bntseq_t *bns, const char *prefix);
	bntseq_t *bns_restore(const char *prefix);
	bntseq_t *bns_restore_core(const char *ann_filename, const char* amb_filename, const char* pac_filename);
	void bns_destroy(bntseq_t *bns);
	int64_t bns_fasta2bntseq(gzFile fp_fa, const char *prefix, int for_only);
	int bns_pos2rid(const bntseq_t *bns, int64_t pos_f);
	int bns_cnt_ambi(const bntseq_t *bns, int64_t pos_f, int len, int *ref_id);
	const uint8_t *bns_get_seq(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len);
	const uint8_t *bns_fetch_seq(const bntseq_t *bns, const uint8_t *pac, int64_t *beg, int64_t mid, int64_t *end, int *rid);
	void bns_get_seq_gpu(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len, gpu_batch *curr_gpu_batch);
	void bns_fetch_seq_gpu(const bntseq_t *bns, const uint8_t *pac, int64_t *beg, int64_t mid, int64_t *end, int *rid, gpu_batch *curr_gpu_batch);
	int bns_intv2rid(const bntseq_t *bns, int64_t rb, int64_t re);

	void bns_fetch_seq_uint8(uint8_t *res, const bntseq_t *bns, const uint8_t *pac, int64_t *beg, int64_t mid, int64_t *end, int *rid);
	void bns_get_seq_uint8(uint8_t *res, int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len);

#ifdef __cplusplus
}
#endif

static inline int64_t bns_depos(const bntseq_t *bns, int64_t pos, int *is_rev)
{
	return (*is_rev = (pos >= bns->l_pac))? (bns->l_pac<<1) - 1 - pos : pos;
}

#endif
