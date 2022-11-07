#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include "cub/cub.cuh"
#include "seed_gen.h"
#include "nvToolsExt.h"

#define gpuErrchk(err) if (err != cudaSuccess) { \
	fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
	exit(EXIT_FAILURE); \
}

double realtime_gpu()
{
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return tp.tv_sec + tp.tv_usec * 1e-6;
}

//#define OCC_INTERVAL 0x100

//#define OCC_INTERVAL 0x80

#define OCC_INTERVAL 0x40

#define OCC_INTV_SHIFT 6

//#define OCC_INTERVAL 0x20

__constant__ bwtint_t_gpu L2_gpu[5];
__constant__ uint32_t ascii_to_dna_table[8];


/* retrieve a character from the $-removed BWT string. Note that
 * bwt_t_gpu::bwt is not exactly the BWT string and therefore this macro is
 * called bwt_B0 instead of bwt_B */

#define bwt_bwt1(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4) + 4])

#define bwt_bwt(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4) + 4 + (k)%OCC_INTERVAL/16])

#define bwt_B0(b, k) (bwt_bwt(b, k)>>((~(k)&0xf)<<1)&3)

#define bwt_occ_intv(b, k) ((b).bwt + (k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4))


//#define bwt_occ_intv(b, k) ((b).bwt + ((k)>>7<<4))

//#define bwt_bwt(b, k) ((b).bwt[((k)>>7<<4) + sizeof(bwtint_t_gpu) + (((k)&0x7f)>>4)])



__device__ inline uint pop_count_partial(uint32_t word, uint8_t c, uint32_t mask_bits) {

	 word =  word & ~((1<<(mask_bits<<1)) - 1);
	 uint odd  = ((c&2)? word : ~word) >> 1;
	 uint even = ((c&1)? word : ~word);
	 uint mask = odd & even & 0x55555555;
	 return (c == 0) ? __popc(mask) - mask_bits : __popc(mask);

}

__device__ inline uint pop_count_partial_64(bwtint_t_gpu word, uint8_t c, bwtint_t_gpu mask_bits) {

	 word = ((c&2)? word : ~word) >> 1 & ((c&1)? word : ~word) & 0x5555555555555555ull;
	 word =  word & ~((1ull<<(mask_bits<<1)) - 1);
	 return (c == 0) ? __popc(word) - mask_bits : __popc(word);

}

__device__ inline uint pop_count_full(uint32_t word, uint8_t c) {

	 uint odd  = ((c&2)? word : ~word) >> 1;
	 uint even = ((c&1)? word : ~word);
	 uint mask = odd & even & 0x55555555;
	 return __popc(mask);
}

__device__ inline uint pop_count_full_64(bwtint_t_gpu word, uint8_t c) {

	// reduce nucleotide counting to bits counting
	word = ((c&2)? word : ~word) >> 1 & ((c&1)? word : ~word) & 0x5555555555555555ull;
	return __popcll(word);
}


__device__ inline int __occ_aux(uint64_t y, int c)
{
	// reduce nucleotide counting to bits counting
	y = ((c&2)? y : ~y) >> 1 & ((c&1)? y : ~y) & 0x5555555555555555ull;
	// count the number of 1s in y
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}

__device__ inline bwtint_t_gpu bwt_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, uint8_t c)
{
	bwtint_t_gpu n, l;

	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];
	if (k == (bwtint_t_gpu)(-1)) return 0;
	if (k >= bwt.primary) --k; // because $ is not in bwt

	// retrieve Occ at k/OCC_INTERVAL
	n = bwt_occ_intv(bwt, k)[c];
	uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;

	if (k_words > 0) n += pop_count_full( bwt_str_k.x, c );
	if (k_words > 1) n += pop_count_full( bwt_str_k.y, c );
	if (k_words > 2) n += pop_count_full( bwt_str_k.z, c );

	//n += pop_count_partial_64( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 31);
	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 15);
	return n;
}


__device__ inline void bwt_occ4_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, bwtint_t_gpu cnt[4])
{


	if (k == (bwtint_t_gpu)(-1)) {
		cnt[0] = 0;
		cnt[1] = 0;
		cnt[2] = 0;
		cnt[3] = 0;
		return;
	}
	if (k >= bwt.primary) --k; // because $ is not in bwt

	// retrieve Occ at k/OCC_INTERVAL
	uint4 count = ((uint4*)bwt_occ_intv(bwt,k))[0];
	uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;

	uint8_t c;

	for (c = 0; c < 4; c++){
		bwtint_t_gpu n;
		n = c == 0 ? count.x : (c == 1 ? count.y : (c == 2 ? count.z : count.w));
		if (k_words > 0) n += pop_count_full( bwt_str_k.x, c );
		if (k_words > 1) n += pop_count_full( bwt_str_k.y, c );
		if (k_words > 2) n += pop_count_full( bwt_str_k.z, c );

		//n += pop_count_partial_64( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 31);
		n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 15);
		cnt[c] = n;
	}

	return;

}


__device__ inline void find_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c, bwtint_t_gpu *occ_l, bwtint_t_gpu *occ_u) {

	 if (c > 3) {
	 	*occ_l = l - L2_gpu[c];
	 	*occ_u = u - L2_gpu[c];

	 }
	 if (l == bwt.seq_len) {
	 	*occ_l = L2_gpu[c+1] - L2_gpu[c];
	 	*occ_u = bwt_occ_gpu(bwt, u, c);

	 }
	 if (u == bwt.seq_len) {
	 	*occ_l = bwt_occ_gpu(bwt, l, c);
	 	*occ_u = L2_gpu[c+1] - L2_gpu[c];

	 }
	 if (l == (bwtint_t_gpu)(-1)) {
	 	*occ_l = 0;
	 	*occ_u = bwt_occ_gpu(bwt, u, c);

	 }
	 if (u == (bwtint_t_gpu)(-1)) {
	 	*occ_l = bwt_occ_gpu(bwt, l, c);
	 	*occ_u = 0;

	 }

	 if (l >= bwt.primary) --l;
	 if (u >= bwt.primary) --u;

	 bwtint_t_gpu kl = l / OCC_INTERVAL;
	 bwtint_t_gpu ku = u /OCC_INTERVAL;

	 uint4 bwt_str_l = ((uint4*)(&bwt_bwt1(bwt,l)))[0];
	 uint4 bwt_str_u = (kl == ku) ? bwt_str_l : ((uint4*)(&bwt_bwt1(bwt,u)))[0];

	 uint32_t l_words = (l&(OCC_INTERVAL-1)) >> 4;
	 uint32_t u_words = (u&(OCC_INTERVAL-1)) >> 4;

	 *occ_l = bwt_occ_intv(bwt, l)[c];
	 if (l_words > 0) *occ_l += pop_count_full( bwt_str_l.x, c );
	 if (l_words > 1) *occ_l += pop_count_full( bwt_str_l.y, c );
	 if (l_words > 2) *occ_l += pop_count_full( bwt_str_l.z, c );

	 *occ_u = (kl == ku) ? *occ_l : bwt_occ_intv(bwt, u)[c];
	 uint32_t startm = (kl == ku) ? l_words : 0;

	 // sum up all the pop-counts of the relevant masks
	 if (u_words > 0 && startm == 0) *occ_u += pop_count_full( bwt_str_u.x, c );
	 if (u_words > 1 && startm <= 1) *occ_u += pop_count_full( bwt_str_u.y, c );
	 if (u_words > 2 && startm <= 2) *occ_u += pop_count_full( bwt_str_u.z, c );

	 *occ_l += pop_count_partial( l_words <= 1 ? (l_words == 0 ? bwt_str_l.x : bwt_str_l.y) : (l_words == 2 ? bwt_str_l.z : bwt_str_l.w), c,  (~l) & 15);
	 *occ_u += pop_count_partial( u_words <= 1 ? (u_words == 0 ? bwt_str_u.x : bwt_str_u.y) : (u_words == 2 ? bwt_str_u.z : bwt_str_u.w), c,  (~u) & 15);


	 //return make_uint4(occ_l >> 32, occ_l, occ_u >> 32, occ_u);
	 return;


}

__device__ inline void find_occ4_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, bwtint_t_gpu tk[4], bwtint_t_gpu tl[4]) {


	 bwtint_t_gpu _l, _u;
	 _l = l - (l >= bwt.primary);
	 _u = u - (u >= bwt.primary);

	 if (_l/OCC_INTERVAL != _u/OCC_INTERVAL || l == (bwtint_t_gpu)(-1) || u == (bwtint_t_gpu)(-1)) {
	 	bwt_occ4_gpu(bwt, l, tk);
	 	bwt_occ4_gpu(bwt, u, tl);
	 	return;
	 }


	 if (l >= bwt.primary) --l;
	 if (u >= bwt.primary) --u;

	 uint4 count = ((uint4*)bwt_occ_intv(bwt,l))[0];
	 uint4 bwt_str_l = ((uint4*)(&bwt_bwt1(bwt,l)))[0];
	 uint4 bwt_str_u = bwt_str_l;

	 uint32_t l_words = (l&(OCC_INTERVAL-1)) >> 4;
	 uint32_t u_words = (u&(OCC_INTERVAL-1)) >> 4;

	 uint8_t c;

	 for (c = 0; c < 4; c++){
		 bwtint_t_gpu occ_l, occ_u;
		 occ_l = c == 0 ? count.x : (c == 1 ? count.y : (c == 2 ? count.z : count.w));
		 if (l_words > 0) occ_l += pop_count_full( bwt_str_l.x, c );
		 if (l_words > 1) occ_l += pop_count_full( bwt_str_l.y, c );
		 if (l_words > 2) occ_l += pop_count_full( bwt_str_l.z, c );

		 occ_u = occ_l;

		 // sum up all the pop-counts of the relevant masks
		 if (u_words > 0 && l_words == 0) occ_u += pop_count_full( bwt_str_u.x, c );
		 if (u_words > 1 && l_words <= 1) occ_u += pop_count_full( bwt_str_u.y, c );
		 if (u_words > 2 && l_words <= 2) occ_u += pop_count_full( bwt_str_u.z, c );

		 occ_l += pop_count_partial( l_words <= 1 ? (l_words == 0 ? bwt_str_l.x : bwt_str_l.y) : (l_words == 2 ? bwt_str_l.z : bwt_str_l.w), c,  (~l) & 15);
		 occ_u += pop_count_partial( u_words <= 1 ? (u_words == 0 ? bwt_str_u.x : bwt_str_u.y) : (u_words == 2 ? bwt_str_u.z : bwt_str_u.w), c,  (~u) & 15);

		 tk[c] = occ_l;
		 tl[c] = occ_u;

	 }
	 return;

}

__device__ inline uint4 find_occ_gpu_print(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c)
{
	bwtint_t_gpu occ_l = 0, occ_u = 0; //occ_l_1 = 0, occ_l_2 = 0, occ_l_3 = 0, occ_u_1 = 0;

	bwtint_t_gpu l_ret, u_ret;

	//if (c > 3)  return make_uint4((l - L2_gpu[c]) >> 32, l - L2_gpu[c], (u - L2_gpu[c]) >> 32, u - L2_gpu[c]);
	if (c > 3) {
		l_ret = l - L2_gpu[c];
		u_ret = u - L2_gpu[c]; 
		printf("l_ret = %lu\n", l_ret);
		printf("u_ret = %lu\n", u_ret);
		return make_uint4(l_ret >> 32, l_ret, u_ret >> 32, u_ret);
	}
	//if (l == bwt.seq_len) return make_uint4 ((L2_gpu[c+1] - L2_gpu[c]) >> 32, L2_gpu[c+1] - L2_gpu[c], bwt_occ_gpu(bwt, u, c) >> 32, bwt_occ_gpu(bwt, u, c));
	if (l == bwt.seq_len) {
		l_ret = L2_gpu[c+1] - L2_gpu[c];
		u_ret = bwt_occ_gpu(bwt, u, c);
		printf("l_ret = %lu\n", l_ret);
		printf("u_ret = %lu\n", u_ret);

		return make_uint4(l_ret >> 32, l_ret, u_ret >> 32, u_ret);
	}
	//if (u == bwt.seq_len) return make_uint4 (bwt_occ_gpu(bwt, l, c) >> 32, bwt_occ_gpu(bwt, l, c), (L2_gpu[c+1] - L2_gpu[c]) >> 32, L2_gpu[c+1] - L2_gpu[c]);
	if (u == bwt.seq_len) {
		l_ret = bwt_occ_gpu(bwt, l, c);
		u_ret = L2_gpu[c+1] - L2_gpu[c];
		printf("l_ret = %lu\n", l_ret);
		printf("u_ret = %lu\n", u_ret);

		return make_uint4(l_ret >> 32, l_ret, u_ret >> 32, u_ret);
	}
	//if (l == (bwtint_t_gpu)(-1)) return make_uint4 (0, 0, bwt_occ_gpu(bwt, u, c) >> 32, bwt_occ_gpu(bwt, u, c));
	if (l == (bwtint_t_gpu)(-1)) {
		l_ret = 0;
		u_ret = bwt_occ_gpu(bwt, u, c);
		printf("l_ret = %lu\n", l_ret);
		printf("u_ret = %lu\n", u_ret);

		return make_uint4(l_ret >> 32, l_ret, u_ret >> 32, u_ret);
	}
	//if (u == (bwtint_t_gpu)(-1)) return make_uint4 (bwt_occ_gpu(bwt , l, c) >> 32, bwt_occ_gpu(bwt , l, c), 0, 0);
	if (u == (bwtint_t_gpu)(-1)) {
		l_ret = bwt_occ_gpu(bwt, l, c);
		u_ret = 0;
		printf("l_ret = %lu\n", l_ret);
		printf("u_ret = %lu\n", u_ret);

		return make_uint4(l_ret >> 32, l_ret, u_ret >> 32, u_ret);
	}
	if (l >= bwt.primary) --l;
	if (u >= bwt.primary) --u;

	bwtint_t_gpu kl = l / OCC_INTERVAL;
	bwtint_t_gpu ku = u /OCC_INTERVAL;

	uint4 bwt_str_l = ((uint4*)(&bwt_bwt1(bwt,l)))[0];
	uint4 bwt_str_u = (kl == ku) ? bwt_str_l : ((uint4*)(&bwt_bwt1(bwt,u)))[0];

	uint32_t l_words = (l&(OCC_INTERVAL-1)) >> 4;
	uint32_t u_words = (u&(OCC_INTERVAL-1)) >> 4;

	occ_l = bwt_occ_intv(bwt, l)[c];
	if (l_words > 0) occ_l += pop_count_full( bwt_str_l.x, c );
	if (l_words > 1) occ_l += pop_count_full( bwt_str_l.y, c );
	if (l_words > 2) occ_l += pop_count_full( bwt_str_l.z, c );

	occ_u = (kl == ku) ? occ_u + occ_l : bwt_occ_intv(bwt, u)[c];
	uint32_t startm = (kl == ku) ? l_words : 0;

	// sum up all the pop-counts of the relevant masks
	if (u_words > 0 && startm == 0) occ_u += pop_count_full( bwt_str_u.x, c );
	if (u_words > 1 && startm <= 1) occ_u += pop_count_full( bwt_str_u.y, c );
	if (u_words > 2 && startm <= 2) occ_u += pop_count_full( bwt_str_u.z, c );

	occ_l += pop_count_partial( l_words <= 1 ? (l_words == 0 ? bwt_str_l.x : bwt_str_l.y) : (l_words == 2 ? bwt_str_l.z : bwt_str_l.w), c,  (~l) & 15);
	occ_u += pop_count_partial( u_words <= 1 ? (u_words == 0 ? bwt_str_u.x : bwt_str_u.y) : (u_words == 2 ? bwt_str_u.z : bwt_str_u.w), c,  (~u) & 15);

	printf("l_occ = %lu\n", occ_l);
	printf("u_occ = %lu\n", occ_u);

	return make_uint4(occ_l >> 32, occ_l, occ_u >> 32, occ_u);
}


#define THREADS_PER_SEED 4


#if(THREADS_PER_SEED==8)
__device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, int tid) {
	if (k > bwt.primary) --k; // because $ is not in bwt
	if (k == bwt.primary) return 0;


	//uint4 count = ((uint4*)bwt_occ_intv(bwt,k))[0];
	//uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
	uint32_t bin_reg = bwt_occ_intv(bwt,k)[tid&7];
	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;


	uint8_t c;

	c = (bin_reg>>((~(k)&0xf)<<1))&3;

	c = __shfl_sync(0xFFFFFFFF, c, k_words + 4, 8);

	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];


	bwtint_t_gpu n = bin_reg;

	n = __shfl_sync(0xFFFFFFFF, n, c, 8);

	//n = bwt_occ_intv(bwt, k)[c];
	uint32_t bwt_1 = __shfl_sync(0xFFFFFFFF, bin_reg, 5, 8);
	uint32_t bwt_2 = __shfl_sync(0xFFFFFFFF, bin_reg, 6, 8);
	uint32_t bwt_3 = __shfl_sync(0xFFFFFFFF, bin_reg, 7, 8);

	if (k_words > 0) n += pop_count_full( bin_reg, c );
	if (k_words > 1) n += pop_count_full( bwt_1, c );
	if (k_words > 2) n += pop_count_full( bwt_2, c );

	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bin_reg : bwt_1) : (k_words == 2 ? bwt_2 : bwt_3), c,  (~k) & 15);

	return L2_gpu[c] + n;
}
#endif

#if(THREADS_PER_SEED==4)
__device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, int tid) {
 	if (k > bwt.primary) --k; // because $ is not in bwt
 	if (k == bwt.primary) return 0;


 	//uint4 count = ((uint4*)bwt_occ_intv(bwt,k))[0];
 	//uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
 	uint2 bin_reg = ((uint2*)bwt_occ_intv(bwt,k))[tid&3];
 	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;


 	uint8_t c;

 	if (k_words == 0) c = (bin_reg.x>>((~(k)&0xf)<<1))&3;
 	if (k_words == 1) c = (bin_reg.y>>((~(k)&0xf)<<1))&3;
 	if (k_words == 2) c = (bin_reg.x>>((~(k)&0xf)<<1))&3;
 	if (k_words == 3) c = (bin_reg.y>>((~(k)&0xf)<<1))&3;

 	c = __shfl_sync(0xFFFFFFFF, c, (k_words == 0 || k_words == 1) ? 2 : 3, 4);

 	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];


 	bwtint_t_gpu n = c == 0 ? bin_reg.x : (c == 1 ? bin_reg.y : (c == 2 ? bin_reg.x : bin_reg.y));

 	n = __shfl_sync(0xFFFFFFFF, n, (c == 0 || c == 1) ? 0 : 1, 4);

 	//n = bwt_occ_intv(bwt, k)[c];
 	uint32_t bwt_1 = __shfl_sync(0xFFFFFFFF, bin_reg.x, 3, 4);
 	uint32_t bwt_2 = __shfl_sync(0xFFFFFFFF, bin_reg.y, 3, 4);

 	if (k_words > 0) n += pop_count_full( bin_reg.x, c );
 	if (k_words > 1) n += pop_count_full( bin_reg.y, c );
 	if (k_words > 2) n += pop_count_full( bwt_1, c );

 	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bin_reg.x : bin_reg.y) : (k_words == 2 ? bwt_1 : bwt_2), c,  (~k) & 15);

 	return L2_gpu[c] + n;
 }

#endif

#if(THREADS_PER_SEED==2)
 __device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, int tid) {
 	if (k > bwt.primary) --k; // because $ is not in bwt
 	if (k == bwt.primary) return 0;


 	//uint4 count = ((uint4*)bwt_occ_intv(bwt,k))[0];
 	//uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
 	uint4 bin_reg = ((uint4*)bwt_occ_intv(bwt,k))[tid&1];
 	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;


 	uint8_t c;

 	if (k_words == 0) c = (bin_reg.x>>((~(k)&0xf)<<1))&3;
 	if (k_words == 1) c = (bin_reg.y>>((~(k)&0xf)<<1))&3;
 	if (k_words == 2) c = (bin_reg.z>>((~(k)&0xf)<<1))&3;
 	if (k_words == 3) c = (bin_reg.w>>((~(k)&0xf)<<1))&3;

 	c = __shfl_sync(0xFFFFFFFF, c, 1, 2);

 	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];


 	bwtint_t_gpu n = c == 0 ? bin_reg.x : (c == 1 ? bin_reg.y : (c == 2 ? bin_reg.z : bin_reg.w));

 	n = __shfl_sync(0xFFFFFFFF, n, 0, 2);

 	//n = bwt_occ_intv(bwt, k)[c];
 	if (k_words > 0) n += pop_count_full( bin_reg.x, c );
 	if (k_words > 1) n += pop_count_full( bin_reg.y, c );
 	if (k_words > 2) n += pop_count_full( bin_reg.z, c );

 	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bin_reg.x : bin_reg.y) : (k_words == 2 ? bin_reg.z : bin_reg.w), c,  (~k) & 15);

 	return L2_gpu[c] + n;
 }
#endif

#if(THREADS_PER_SEED==1)
 __device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k) {
 	if (k > bwt.primary) --k; // because $ is not in bwt
 	if (k == bwt.primary) return 0;


 	uint4 count = ((uint4*)bwt_occ_intv(bwt,k))[0];
 	uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
 	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;

 	uint8_t c;
 	if (k_words == 0) c = (bwt_str_k.x>>((~(k)&0xf)<<1))&3;
 	if (k_words == 1) c = (bwt_str_k.y>>((~(k)&0xf)<<1))&3;
 	if (k_words == 2) c = (bwt_str_k.z>>((~(k)&0xf)<<1))&3;
 	if (k_words == 3) c = (bwt_str_k.w>>((~(k)&0xf)<<1))&3;

 	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];



 	bwtint_t_gpu n = c == 0 ? count.x : (c == 1 ? count.y : (c == 2 ? count.z : count.w));


 	if (k_words > 0) n += pop_count_full( bwt_str_k.x, c );
 	if (k_words > 1) n += pop_count_full( bwt_str_k.y, c );
 	if (k_words > 2) n += pop_count_full( bwt_str_k.z, c );

 	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 15);

 	return L2_gpu[c] + n;
 }
#endif

#define THREADS_PER_SMEM 1


__global__ void seeds_to_threads(int2 *final_seed_read_pos_fow_rev, bwtint_t_gpu *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t n_smems_sum_fow_rev, uint32_t *final_seed_scores_gpu) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= (n_smems_sum_fow_rev)*THREADS_PER_SMEM) return;

        int n_seeds = n_seeds_fow_rev_scan[tid+1] - n_seeds_fow_rev_scan[tid];//n_seeds_fow[tid];
        int2 seed_read_pos = seed_read_pos_fow_rev[tid];

        //uint32_t intv_l = seed_intervals_fow_rev[tid].x;
	bwtint_t_gpu intv_l, intv_u;
	intv_u = (( ((bwtint_t_gpu) seed_intervals_fow_rev[tid].x) & 0x1) << 32) | ((bwtint_t_gpu) seed_intervals_fow_rev[tid].y);	
	intv_l = intv_u - (bwtint_t_gpu) (seed_intervals_fow_rev[tid].x >> 1) + 1;


        uint32_t offset = n_seeds_fow_rev_scan[tid];
        int idx = tid%THREADS_PER_SMEM;
        int i;
        for(i = 0; (i + idx) < n_seeds; i+=THREADS_PER_SMEM) {
        	seed_sa_idx_fow_rev_gpu[offset + i + idx] = intv_l + i + idx;
        	final_seed_read_pos_fow_rev[offset + i + idx] = seed_read_pos;
			if (i == 0) final_seed_scores_gpu[offset + i + idx] = n_seeds;
        }

        return;

}

__global__ void seeds_to_threads_mem(int2 *final_seed_read_pos_fow_rev, bwtint_t_gpu *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t n_smems_sum_fow_rev, uint32_t *final_seed_scores_gpu) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= (n_smems_sum_fow_rev)*THREADS_PER_SMEM) return;

        int n_seeds = n_seeds_fow_rev_scan[tid+1] - n_seeds_fow_rev_scan[tid];//n_seeds_fow[tid];
        int2 seed_read_pos = seed_read_pos_fow_rev[tid];
        // uint32_t is_rev_strand = (seed_read_pos.y) >> 31; // not needed
        
	//uint32_t intv_l = seed_intervals_fow_rev[tid].x;
        bwtint_t_gpu intv_l, intv_u;
	intv_u = (( ((bwtint_t_gpu) seed_intervals_fow_rev[tid].x) & 0x1) << 32) | ((bwtint_t_gpu) seed_intervals_fow_rev[tid].y);	
	intv_l = intv_u - (bwtint_t_gpu) (seed_intervals_fow_rev[tid].x >> 1) + 1;
	
	uint32_t offset = n_seeds_fow_rev_scan[tid];
        //uint2 next_seed_interval = make_uint2(1,0);
        uint2 next_seed_interval = make_uint2(0,0);
	uint32_t next_seed_sa_intv;

        	int p = 1;
        	while (seed_read_pos.x == seed_read_pos_fow_rev[tid+p].x && tid + p < n_smems_sum_fow_rev){
        		next_seed_interval = seed_intervals_fow_rev[tid+p];

			// shift by 1 to take only the sa_intv
			//next_seed_sa_intv = next_seed_interval.x >> 1;

        		//if (next_seed_interval.y - next_seed_interval.x + 1 > 0) break;
        		if ((next_seed_interval.x >> 1) > 0) break;
        		p++;

        	}


        int i = 0;
        int seed_count = 0;

	// compute next_seed_interval l and u
	bwtint_t_gpu next_seed_interval_l, next_seed_interval_u;
	next_seed_interval_u = (( ((bwtint_t_gpu) next_seed_interval.x) & 0x1) << 32) | ((bwtint_t_gpu) next_seed_interval.y);	
	next_seed_interval_l = next_seed_interval_u - (bwtint_t_gpu) (next_seed_interval.x >> 1) + 1;
	
        for(i = 0; seed_count < n_seeds; i++) {
        //for(i = 0; seed_count < n_seeds ; i++) {
        	//if (((intv_l + i) < next_seed_interval.x) || ((intv_l + i) > next_seed_interval.y)){
        	if (((intv_l + i) < next_seed_interval_l) || ((intv_l + i) > next_seed_interval_u)){
        		seed_sa_idx_fow_rev_gpu[offset + seed_count] = intv_l + i;
        		final_seed_read_pos_fow_rev[offset + seed_count] = seed_read_pos;
				if (seed_count == 0) final_seed_scores_gpu[offset + seed_count] = n_seeds;
        		seed_count++;
			// debugging
			// printf("final_seed_pos.x = %d\n", final_seed_read_pos_fow_rev[offset + seed_count].x);
			// printf("final_seed_pos.y = %d\n", final_seed_read_pos_fow_rev[offset + seed_count].y);
        	}
        }

        return;

}

#if(THREADS_PER_SEED==1)
__global__ void locate_seeds_gpu(bwtint_t_gpu *seed_ref_pos_fow_rev_gpu, bwt_t_gpu bwt, uint32_t n_seeds_sum_fow_rev) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= n_seeds_sum_fow_rev) return;

        	//uint32_t sa_idx = seed_ref_pos_fow_rev_gpu[tid];
        	bwtint_t_gpu sa_idx = seed_ref_pos_fow_rev_gpu[tid];
        	if (sa_idx == UINT_MAX) return;
        	int itr = 0, mask = bwt.sa_intv - 1;
        	// while(sa_idx % bwt.sa_intv){
        	while(sa_idx & mask){
        		itr++;
        		sa_idx = bwt_inv_psi_gpu(bwt, sa_idx);
        	}
        	//seed_ref_pos_fow_rev_gpu[tid] = bwt.sa[sa_idx/bwt.sa_intv] + itr;
		int idx = sa_idx/bwt.sa_intv;
		uint64_t bits_to_append = bwt.sa_upper_bits[bwt.pack_size * idx / 32];
		// for general case: mask instead of 0x1
		bits_to_append = ((bits_to_append >> (idx & 31)) & 0x1) << 32;
		
        	seed_ref_pos_fow_rev_gpu[tid] = (((bwtint_t_gpu) bwt.sa[idx]) | bits_to_append) + itr;

        return;

}
#endif

#if(THREADS_PER_SEED > 1)
__global__ void locate_seeds_gpu(bwtint_t_gpu *seed_ref_pos_fow_rev_gpu, bwt_t_gpu bwt, uint32_t n_tasks) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= (n_tasks<<((int)log2((double)THREADS_PER_SEED)))) return;

        	//uint32_t sa_idx = seed_ref_pos_fow_rev_gpu[tid];
        	bwtint_t_gpu sa_idx = seed_ref_pos_fow_rev_gpu[tid>>((int)log2((double)THREADS_PER_SEED))];
        	if (sa_idx == UINT_MAX) return;
        	int itr = 0, mask = bwt.sa_intv - 1;
        	// while(sa_idx % bwt.sa_intv){
        	while(sa_idx & mask){
        		itr++;
        		sa_idx = bwt_inv_psi_gpu(bwt, sa_idx, tid);
        		sa_idx = __shfl_sync(0xFFFFFFFF, sa_idx, (THREADS_PER_SEED>>1), THREADS_PER_SEED);
        	}
        	//seed_ref_pos_fow_rev_gpu[tid] = bwt.sa[sa_idx/bwt.sa_intv] + itr;
        	if((tid&(THREADS_PER_SEED - 1)) == 0){
        		int idx = sa_idx/bwt.sa_intv;
        		uint64_t bits_to_append = bwt.sa_upper_bits[bwt.pack_size * idx / 32];
        		// for general case: mask instead of 0x1
        		bits_to_append = ((bits_to_append >> (idx & 31)) & 0x1) << 32;

        		seed_ref_pos_fow_rev_gpu[tid>>((int)log2((double)THREADS_PER_SEED))] = (((bwtint_t_gpu) bwt.sa[idx]) | bits_to_append) + itr;
        	}

        return;

}
#endif
__global__ void locate_seeds_gpu_wrapper(int2 *final_seed_read_pos_fow_rev, bwtint_t_gpu *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev,uint32_t *n_smems_sum_fow_rev, uint32_t *n_seeds_sum_fow_rev, bwt_t_gpu bwt, uint32_t *final_seed_scores_gpu) {


	int BLOCKDIM =128;
	int N_BLOCKS = (n_smems_sum_fow_rev[0]*THREADS_PER_SMEM  + BLOCKDIM - 1)/BLOCKDIM;

	n_seeds_fow_rev_scan[n_smems_sum_fow_rev[0]] = n_seeds_sum_fow_rev[0];

	seeds_to_threads<<<N_BLOCKS, BLOCKDIM>>>(final_seed_read_pos_fow_rev, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev, seed_read_pos_fow_rev, n_smems_sum_fow_rev[0], final_seed_scores_gpu);

	bwtint_t_gpu *seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;


	N_BLOCKS = ((n_seeds_sum_fow_rev[0]*THREADS_PER_SEED)  + BLOCKDIM - 1)/BLOCKDIM;

	locate_seeds_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_ref_pos_fow_rev_gpu, bwt, n_seeds_sum_fow_rev[0]);
}


__global__ void locate_seeds_gpu_wrapper_mem(int2 *final_seed_read_pos_fow_rev, bwtint_t_gpu *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev,uint32_t *n_smems_sum_fow_rev, uint32_t *n_seeds_sum_fow_rev, uint32_t *n_ref_pos_fow_rev, bwt_t_gpu bwt, uint32_t *final_seed_scores_gpu) {


	int BLOCKDIM =128;
	int N_BLOCKS = (n_smems_sum_fow_rev[0]*THREADS_PER_SMEM  + BLOCKDIM - 1)/BLOCKDIM;

	n_seeds_fow_rev_scan[n_smems_sum_fow_rev[0]] = n_seeds_sum_fow_rev[0];

	seeds_to_threads_mem<<<N_BLOCKS, BLOCKDIM>>>(final_seed_read_pos_fow_rev, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev, seed_read_pos_fow_rev, n_smems_sum_fow_rev[0], final_seed_scores_gpu);

	bwtint_t_gpu *seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;


	N_BLOCKS = (n_seeds_sum_fow_rev[0]  + BLOCKDIM - 1)/BLOCKDIM;

	locate_seeds_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_ref_pos_fow_rev_gpu, bwt, n_seeds_sum_fow_rev[0]);


}


__global__ void count_seed_intervals_gpu(uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan, uint32_t* n_ref_pos_fow_rev,  uint32_t n_smems_max, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int thread_read_num = tid/n_smems_max;
	int offset_in_read = tid - (thread_read_num*n_smems_max);
	if(offset_in_read >= n_smems_fow_rev[thread_read_num]) return;
	int intv_idx = n_smems_fow_rev_scan[thread_read_num];
	//int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
	int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].x >> 1;
	n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals;
	if (n_intervals > 0)  atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals);

	return;

}

__global__ void count_seed_intervals_gpu_mem(uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow,  uint32_t *n_smems_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,uint32_t *n_ref_pos_fow_rev, uint32_t n_smems_max, int n_tasks) {

	 int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	 //if (tid >= 2*n_tasks) return;
	 if (tid >= n_tasks) return;

	 if (tid < n_tasks) {
		 int thread_read_num = tid/n_smems_max;
		 int offset_in_read = tid - (thread_read_num*n_smems_max);
		 if(offset_in_read >= n_smems_fow[thread_read_num]) return;
		 int intv_idx = n_smems_fow_rev_scan[thread_read_num];
		 //int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
		 int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].x >> 1;
		 int n_intervals_to_add = n_intervals;
		 int next_n_intervals = 0;
		 if (n_intervals > 0) {
			 int seed_read_pos_x = seed_read_pos_fow_rev[intv_idx + offset_in_read].x;
			 int p = 1;
			 while (seed_read_pos_x == seed_read_pos_fow_rev[intv_idx + offset_in_read + p].x && offset_in_read + p < n_smems_fow[thread_read_num]) {
				 //next_n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read + p].y - seed_intervals_fow_rev[intv_idx + offset_in_read + p].x + 1;
				 next_n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read + p].x >> 1;
				 if (next_n_intervals > 0) {
					 n_intervals_to_add = n_intervals - next_n_intervals;
					 break;
				 }

				 p++;
			 }
			 atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals_to_add);
		 }
		 n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals_to_add;
	 }
	return;

}

__global__ void filter_seed_intervals_gpu(uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev_scan, uint32_t n_smems_max, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	//if (tid >= 2*n_tasks) return;
	if (tid >= n_tasks) return;

	if (tid < n_tasks) {
		int thread_read_num = tid/n_smems_max;
		int offset_in_read = tid - (thread_read_num*n_smems_max);
		if(offset_in_read >= n_smems_fow[thread_read_num] || offset_in_read == 0) return;
		int intv_idx = n_smems_fow_rev_scan[thread_read_num];
		int seed_begin_pos = seed_read_pos_fow_rev[intv_idx + offset_in_read].x;
		int comp_seed_begin_pos = seed_read_pos_fow_rev[intv_idx + offset_in_read - 1].x;
		if(seed_begin_pos == comp_seed_begin_pos ) {
			//seed_intervals_fow_rev[intv_idx + offset_in_read - 1]= make_uint2 (1, 0);
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1] = make_uint2 (0, 0);

			//seed_read_pos_fow[intv_idx + offset_in_read].y =  -1;
		}
	}
	return;

}

__global__ void filter_seed_intervals_gpu_mem(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev_scan, uint32_t n_smems_max, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	//if (tid >= 2*n_tasks) return;
	if (tid >= n_tasks) return;
	if (tid < n_tasks) {
		int thread_read_num = tid/n_smems_max;
		int offset_in_read = tid - (thread_read_num*n_smems_max);
		if(offset_in_read >= n_smems_fow[thread_read_num] || offset_in_read == 0) return;
		int intv_idx = n_smems_fow_rev_scan[thread_read_num];
		if(offset_in_read == n_smems_fow[thread_read_num] - 1){
			seed_intervals_fow_rev[intv_idx + offset_in_read] = seed_intervals_fow_rev_compact[intv_idx + offset_in_read];
		}
		int seed_begin_pos = seed_read_pos_fow_rev_compact[intv_idx + offset_in_read].x;
		int comp_seed_begin_pos = seed_read_pos_fow_rev_compact[intv_idx + offset_in_read - 1].x;
		//if(seed_begin_pos == comp_seed_begin_pos && ((seed_intervals_fow_rev_compact[intv_idx + offset_in_read].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read].x + 1) == (seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].x + 1))) {
		uint32_t seed_sa_intv = seed_intervals_fow_rev_compact[intv_idx + offset_in_read].x >> 1; 
		uint32_t comp_seed_sa_intv = seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].x >> 1; 
		if(seed_begin_pos == comp_seed_begin_pos && (seed_sa_intv == comp_seed_sa_intv)) {
			//seed_intervals_fow_rev[intv_idx + offset_in_read - 1] =  make_uint2 (1, 0);			//seed_read_pos_fow[intv_idx + offset_in_read].y =  -1;
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1] =  make_uint2 (0, 0);			//seed_read_pos_fow[intv_idx + offset_in_read].y =  -1;
		} else {
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1] = seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1];
		}
	}

	return;

}

__global__ void filter_seed_intervals_gpu_wrapper(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev,  int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev, uint32_t* n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,  uint32_t* n_ref_pos_fow_rev, uint32_t *n_smems_max, uint32_t *n_smems_sum_fow_rev, void *cub_sort_temp_storage, size_t cub_sort_storage_bytes, int total_reads, int n_bits_max_read_size, int is_smem) {

	// uint32_t n_smems_max_val = n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1];
	uint32_t n_smems_max_val = n_smems_max[0];
	int n_tasks = n_smems_max_val*total_reads;
	//n_smems_sum_fow_rev[0] = n_smems_sum_fow_rev[0]/2; // divided by 2 because of flags
	int BLOCKDIM = 128;
	//int N_BLOCKS = (2*n_tasks + BLOCKDIM - 1)/BLOCKDIM;
	int N_BLOCKS = (n_tasks + BLOCKDIM - 1)/BLOCKDIM;

	filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);
	//cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact, (uint64_t*)seed_read_pos_fow_rev, (uint64_t*)seed_intervals_fow_rev_compact, (uint64_t*)seed_intervals_fow_rev,  n_smems_sum_fow_rev[0], total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size);
	count_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, n_smems_fow_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val, n_tasks);

}

__global__ void filter_seed_intervals_gpu_wrapper_mem(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan, uint32_t *n_ref_pos_fow_rev, uint32_t *n_smems_max, uint32_t *n_smems_sum_fow_rev, int total_reads) {

	// uint32_t n_smems_max_val = n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1];
	uint32_t n_smems_max_val = n_smems_max[0];
	int n_tasks = n_smems_max_val*total_reads;
	//n_smems_sum_fow_rev[0] = n_smems_sum_fow_rev[0]/2; // divided by 2 because of flags
	int BLOCKDIM = 128;
	int N_BLOCKS = (n_tasks + BLOCKDIM - 1)/BLOCKDIM;

	//filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);

	filter_seed_intervals_gpu_mem<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, seed_intervals_fow_rev, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);
	count_seed_intervals_gpu_mem<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val, n_tasks);

}


// is_back = 0
__device__ inline void bwt_extend_fow_gpu(const bwt_t_gpu bwt, const bwtint_t_gpu ik_k, const bwtint_t_gpu ik_l, const uint32_t ik_s, bwtint_t_gpu ok_k[4], bwtint_t_gpu ok_l[4], uint32_t ok_s[4]) {

	bwtint_t_gpu tk[4], tl[4];
	int i;
	find_occ4_gpu(bwt, ik_l - 1, ik_l - 1 + ik_s, tk, tl);
        for (i = 0; i != 4; ++i) {
                ok_l[i] = L2_gpu[i] + 1 + tk[i];
                ok_s[i] = tl[i] - tk[i];
        }
	ok_k[3] = ik_k + (ik_l <= bwt.primary && ik_l + ik_s - 1 >= bwt.primary);
	ok_k[2] = ok_k[3] + ok_s[3];
    	ok_k[1] = ok_k[2] + ok_s[2];
    	ok_k[0] = ok_k[1] + ok_s[1];

}

__device__ inline void bwt_set_intv_gpu(int c, bwtint_t_gpu *ik_k, bwtint_t_gpu *ik_l, uint32_t *ik_s) {
	*ik_k = L2_gpu[c] + 1; // k
    *ik_s = L2_gpu[c+1] - L2_gpu[c]; // s
	*ik_l = L2_gpu[3-c] + 1; // l
}

__global__ void find_seed_intervals_fow(const __restrict__ bwt_t_gpu bwt, const __restrict__ uint32_t* packed_read_batch, int read_batch_size_8, uint32_t* read_sizes, uint32_t *read_offsets, int min_seed_size, uint8_t* is_interval_flags, int max_read_size, uint4 *read_pos_num_len_intervals, int n_tasks) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// first position of DNA read
	const __restrict__ uint32_t *seq = packed_read_batch;
	// load packed read batch into shared memory for memory coalescing
	extern __shared__ int seq_s[];
	int tx = threadIdx.x; int bxdim = blockDim.x;

	int i;
	// starting index of loading into shared memory
	int s_start = i = (read_offsets[tid - tx]) >> 3;
	while ((i + tx < (read_batch_size_8) >> 3) && (i + tx - s_start) < ((bxdim * max_read_size) >> 3)) {
		seq_s[i+tx-s_start] = seq[i+tx];
		i += bxdim;
	}
	__syncthreads();

	if (tid >= n_tasks) return;

	int c;
	bwtint_t_gpu ik_k, ik_l, ok_k[4], ok_l[4];
	uint32_t ik_s, ok_s[4];
	// start of the read
	uint32_t start = read_offsets[tid];
	// end of the read
	uint32_t end = start + read_sizes[tid];

	i = start;
	
	uint32_t prev_pos;

	// offset from start of the block
	uint32_t thread_offset = read_offsets[tid] - read_offsets[tid - tx];

	for (; i < end;) {
		// unpack first base of the interval
		int base;
		do {
			//int reg_no = i >> 3;
			int reg_no = (i - start + thread_offset) >> 3;
			int reg_pos = i & 7; // i % 8 
			//int reg = seq[reg_no];
			int reg = seq_s[reg_no];
			//int reg = seq_s[(i-start+tx*max_read_size) >> 3];
			//if (reg != reg2) printf("tid = %d, tx = %d, bx = %d, reg = %d, reg2 = %d\n", tid, tx, bx, reg, reg2);
			base = (reg >> (28 - (reg_pos << 2)))&15;
			i++;
		} while (base > 3 && i < end); // skip ambiguous bases

		if (i == end && base > 3) return;

		prev_pos = i - 1 - start; // to store previous starting position for back-search

		bwt_set_intv_gpu(base, &ik_k, &ik_l, &ik_s); // the initial interval of a single base

		for (; i < end; i++) {
			// unpack base
			//int reg_no = i >> 3;
			int reg_no = (i - start + thread_offset) >> 3;
			int reg_pos = i & 7;
			//int reg = seq[reg_no];
			int reg = seq_s[reg_no];
			//int reg = seq_s[(i-start+tx*max_read_size) >> 3];
			base = (reg >> (28 - (reg_pos << 2)))&15;
			
			if (base < 4) { // an A/C/G/T base
				c = 3 - base; // complement of base
				bwt_extend_fow_gpu(bwt, ik_k, ik_l, ik_s, ok_k, ok_l, ok_s);
				if (ok_s[c] != ik_s) { // change of the interval size
					int store_idx = i-tid*(min_seed_size-1)-min_seed_size;
					// do not spawn threads that are at the beginning of the read if the length is small, this would also cause conflicts
					if (store_idx >= (int) (start-tid*(min_seed_size-1))) {
						bwtint_t_gpu u = ik_s + ik_k - 1;
						// store s in upper 31 bits and upper bit of u in lower bit and the lower 32 bits of u
						read_pos_num_len_intervals[store_idx] = make_uint4(prev_pos, ((i - start - prev_pos) << 16) | tid, (ik_s << 1) | (u >> 32), u); 
						is_interval_flags[store_idx] = 1;
					}
					if (ok_s[c] < 1) break; // no match with the reference
				}
				ik_k = ok_k[c]; ik_l = ok_l[c]; ik_s = ok_s[c];
			}
			else { // an ambiguous base
				int store_idx = i-tid*(min_seed_size-1)-min_seed_size;

				if (store_idx >= (int) start-tid*(min_seed_size-1)) {
					bwtint_t_gpu u = ik_s + ik_k - 1;
					// store s in upper 31 bits and upper bit of u in lower bit and the lower 32 bits of u
					read_pos_num_len_intervals[store_idx] = make_uint4(prev_pos, ((i - start - prev_pos) << 16) | tid, (ik_s << 1) | (u >> 32), u); 
					is_interval_flags[store_idx] = 1;
				}
				break; // always terminate extension at an ambiguous base; i<l
			}
		}	
	}
	
	int store_idx = i-tid*(min_seed_size-1)-min_seed_size;
	if (i == end && store_idx >= (int) start-tid*(min_seed_size-1)) { 
		bwtint_t_gpu u = ik_s + ik_k - 1;
		// store s in upper 31 bits and upper bit of u in lower bit and the lower 32 bits of u
		read_pos_num_len_intervals[store_idx] = make_uint4(prev_pos, ((i - start - prev_pos) << 16) | tid, (ik_s << 1) | (u >> 32), u); 
		is_interval_flags[store_idx] = 1;
	}

	return;
}


#define N_SHUFFLES 31
__global__ void find_seed_intervals_back(const __restrict__ uint32_t *packed_read_batch_fow, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals,
	
	int2 *seed_read_pos, uint8_t *is_smem_flags, uint32_t *n_smems_fow, int min_seed_size, const __restrict__ bwt_t_gpu bwt, const __restrict__ uint4 *read_pos_num_len_intervals, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	int thread_read_num = read_pos_num_len_intervals[tid].y & 0xFFFF; // read num is in lower 16 bits
	int read_len = read_sizes[thread_read_num];
	int start = read_offsets[thread_read_num];
	uint8_t is_active = 1;

	// searching back position
	int x = read_pos_num_len_intervals[tid].x;
	uint32_t fow_len = read_pos_num_len_intervals[tid].y >> 16;
	uint32_t intv_idx = start - thread_read_num*(min_seed_size-1) + x + fow_len - min_seed_size;

	uint8_t is_shfl[N_SHUFFLES];
	uint8_t neighbour_active[N_SHUFFLES];
	int m;
	unsigned mask = 0xFFFFFFFF;
	for (m = 0; m < N_SHUFFLES; m++) {
		is_shfl[m] = tid + (m+1) >= n_tasks ? 0 : (((tid&31) + m)< 31) ? 1 : 0;
		uint32_t neighbour_thread_read_num = __shfl_down_sync(mask, thread_read_num, m+1);
		if (is_shfl[m]) is_shfl[m] = (neighbour_thread_read_num == thread_read_num) ? 1 : 0;
		uint32_t neighbour_start = __shfl_down_sync(mask, x, m+1);
		if (is_shfl[m]) is_shfl[m] = (neighbour_start == x) ? 1 : 0;
		if (!is_shfl[m]) break;
		neighbour_active[m] = 1;
	}

	int i, j;
	int base;
	bwtint_t_gpu l, u;

	const __restrict__ uint32_t *seq = packed_read_batch_fow;
	uint32_t curr_intv_size = read_pos_num_len_intervals[tid].z >> 1;
	bwtint_t_gpu l_prev, u_prev;
        u_prev = (( ((bwtint_t_gpu) read_pos_num_len_intervals[tid].z) & 0x1) << 32) | ((bwtint_t_gpu) read_pos_num_len_intervals[tid].w);	
	l_prev = u_prev - curr_intv_size + 1;

	int beg_i = i = x - 1 + start;
	l = l_prev, u = u_prev;

	for (; i >= start; i--) {
		/*get the base*/
		int reg_no = i >> 3;
		int reg_pos = i & 7;
		int reg = seq[reg_no];
		int base = (reg >> (28 - (reg_pos << 2)))&15;
		/*unknown bases*/
		if (base > 3) {
			is_active = 0;
			break;
		}

		bwtint_t_gpu occ_l, occ_u;
		find_occ_gpu(bwt, l_prev - 1, u_prev, base, &occ_l, &occ_u);

		l = L2_gpu[base] + occ_l + 1; // modified to accomodate 64 bits for l
		u = L2_gpu[base] + occ_u; // modified to accomodate 64 bits for u

		// optimization: gpuErrchk for redundancy with the upper thread
		// conditions: 1) has an upper thread in the warp, 2) is in the same read and 3) performs extension to the same base
		//unsigned mask = __ballot_sync(0xFFFFFFFF, (tid & 31) < 31 && (read_pos_num_len[tid+1].y & 0xFFFF) == thread_read_num && read_pos_num_len[tid+1].x == x);
		
		if (l > u || base > 3) {
			is_active = 0;
			//break;
		}
		else {
			curr_intv_size = u - l + 1;
			l_prev = l, u_prev = u;
			beg_i = i - 1;
		}

		//curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;

		m = 0;
		while (m < N_SHUFFLES && is_shfl[m] && (!neighbour_active[m]))
			m++;

		if (m < N_SHUFFLES && is_shfl[m]) {
			uint32_t neighbour_intv_size = __shfl_down_sync(mask, curr_intv_size, m+1);
			uint8_t is_neighbour_active = __shfl_down_sync(mask, is_active, m+1);
			neighbour_active[m] = is_neighbour_active;
			if (neighbour_active[m] && neighbour_intv_size == curr_intv_size) {
				is_active = 0;
				return;
			}
		}
		if (!is_active) break;

	}
	if (fow_len + x + start - beg_i - 1 >= min_seed_size /*&& is_smem*/) {
		atomicAdd(&n_smems_fow[thread_read_num], 1);
		//seed_intervals_fow_rev[intv_idx - thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
		uint32_t low_u = u_prev; // register with lower 32-bits of u
		uint32_t s_intv_upp_u = (u_prev - l_prev + 1) << 1; // store the s_intv = u - l + 1 (occupies the leftmost 31 bits)
		s_intv_upp_u |= (u_prev >> 32); // store the 33rd bit of u in the rightmost bit
	
		seed_intervals[intv_idx] = make_uint2(s_intv_upp_u, low_u); // modified to take sa_intv and u
	
		seed_read_pos[intv_idx] = make_int2 (beg_i - start + 1, fow_len + x);
	
		is_smem_flags[intv_idx] = 1;
	}
	return;

}


__global__ void pack_4bit_fow(uint32_t *read_batch, uint32_t* packed_read_batch_fow, int n_tasks) {

	int32_t i;
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index.
	uint32_t ascii_to_dna_reg_rev = 0x24031005;
	if (tid >= n_tasks) return;
	uint32_t *packed_read_batch_addr = &(read_batch[(tid << 1)]);
	uint32_t reg1 = packed_read_batch_addr[0]; //load 4 bases of the first sequence from global memory
	uint32_t reg2 = packed_read_batch_addr[1]; //load  another 4 bases of the S1 from global memory
	uint32_t pack_reg_4bit = 0;
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> ((reg1 & 7) << 2))&15)  << 28;        // ---
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> (((reg1 >> 8) & 7) << 2))&15) << 24; //    |
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> (((reg1 >> 16) & 7) << 2))&15) << 20;//    |
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> (((reg1 >> 24) & 7) << 2))&15) << 16;//    |
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> ((reg2 & 7) << 2))&15) << 12;        //     > pack data
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> (((reg2 >> 8) & 7) << 2))&15) << 8;  //    |
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> (((reg2 >> 16) & 7) << 2))&15) << 4; //    |
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> (((reg2 >> 24) & 7) << 2))&15);      //----
	packed_read_batch_fow[tid] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory

}

__global__ void pack_4bit_rev(uint32_t *read_batch, uint32_t* packed_read_batch_rev, int n_tasks) {

	int32_t i;
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index.
	uint32_t ascii_to_dna_reg_rev = 0x14002035;
	if (tid >= n_tasks) return;
	uint32_t *packed_read_batch_addr = &(read_batch[(tid << 1)]);
	uint32_t reg1 = packed_read_batch_addr[0]; //load 4 bases of the first sequence from global memory
	uint32_t reg2 = packed_read_batch_addr[1]; //load  another 4 bases of the S1 from global memory
	uint32_t pack_reg_4bit = 0;
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> ((reg1 & 7) << 2))&15)  << 28;        // ---
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> (((reg1 >> 8) & 7) << 2))&15) << 24; //    |
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> (((reg1 >> 16) & 7) << 2))&15) << 20;//    |
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> (((reg1 >> 24) & 7) << 2))&15) << 16;//    |
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> ((reg2 & 7) << 2))&15) << 12;        //     > pack data
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> (((reg2 >> 8) & 7) << 2))&15) << 8;  //    |
	pack_reg_4bit |=  ((ascii_to_dna_reg_rev >> (((reg2 >> 16) & 7) << 2))&15) << 4; //    |
	pack_reg_4bit |= ((ascii_to_dna_reg_rev >> (((reg2 >> 24) & 7) << 2))&15);      //----
	packed_read_batch_rev[tid] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory

}


__global__ void assign_threads_to_reads(uint32_t *thread_read_num, uint32_t *thread_read_idx, int offset, int read_num, int n_tasks) {
	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if  (tid >= n_tasks) return;
	thread_read_num[offset + tid] = read_num;
	thread_read_idx[offset + tid] = tid;
}

__global__ void assign_read_num(uint32_t *thread_read_num, int offset, int read_num, int n_tasks) {
	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if  (tid >= n_tasks) return;
	thread_read_num[offset + tid] = read_num;
}

__global__ void prepare_batch(uint32_t *thread_read_num, uint32_t *thread_read_idx, uint32_t *read_offsets, uint32_t* read_sizes, int min_seed_len, int n_tasks, int pack, int max_read_length) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	if (pack) {
		int read_len = read_sizes[tid]%8 ? read_sizes[tid] + (8 - read_sizes[tid]%8) : read_sizes[tid];
		uint32_t BLOCKDIM = (read_len >> 3) > 1024  ? 1024 : (read_len >> 3);
		uint32_t N_BLOCKS = ((read_len >> 3) + BLOCKDIM - 1) / BLOCKDIM;

	}
	else {
		int read_no = tid/max_read_length;
		int offset_in_read = tid - (read_no*max_read_length);
		if (offset_in_read >= read_sizes[read_no] - min_seed_len) return;
		thread_read_num[read_offsets[read_no] - (read_no*min_seed_len) + offset_in_read] = read_no;
		thread_read_idx[read_offsets[read_no] - (read_no*min_seed_len) + offset_in_read] = offset_in_read;

	}
	return;

}

__global__ void pre_calc_seed_intervals_gpu(uint2* pre_calc_intervals, int pre_calc_seed_len, bwt_t_gpu bwt, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	uint32_t seed_patt = tid;
	uint8_t ch;
	int i;
	int donot_add = 0;
	bwtint_t_gpu l = 0, u = bwt.seq_len;
	for (i = 0 ; i < pre_calc_seed_len ; i++) {
		/*get the base*/
		ch = (seed_patt >> (i<<1)) & 3;
		bwtint_t_gpu intv_l, intv_u;
		find_occ_gpu(bwt, l - 1, u, ch, &intv_l, &intv_u);
		l = L2_gpu[ch] + intv_l + 1; // modified to accomodate 64 bits for l
		u = L2_gpu[ch] + intv_u; // modified to accomodate 64 bits for u
	
		if (l > u) {
			break;
		}

	}
	
	uint32_t sa_intv_upp_u; 

	uint32_t low_u = u; // register with lower 32-bits of u
	if (l <= u) {
		sa_intv_upp_u = (u - l + 1) << 1; // store the s_intv = l - u + 1 (occupies the leftmost 31 bits)
		sa_intv_upp_u |= (u >> 32); // store the 33rd bit of u in the rightmost bit
	}
	else
		sa_intv_upp_u = 0; // sa_intv = 0, negatives would cause a problem 

	pre_calc_intervals[tid] = make_uint2 (sa_intv_upp_u, low_u);
	return;

}

__global__ void sum_arrays(uint32_t *in1, uint32_t *in2, uint32_t *out, int num_items) {
	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= num_items) return;

	out[tid] = in1[tid] + in2[tid];

}

__global__ void print_compact_array(uint2 *seed_intervals_fow_rev_compact, uint32_t *n_smems_fow_rev, uint32_t *seed_intervals_fow_rev, uint32_t *is_smem_fow_rev_flag, int num_items) {
         //int tid = blockIdx.x * blockDim.x + threadIdx.x;      

	 for (int i=0; i < num_items; i++) {
		if (( (is_smem_fow_rev_flag[i] != 0) ) ) {
			printf("non_zero_seed_intervals_fow_rev[%d] = %u \n", i, seed_intervals_fow_rev[i]);
			printf("smem_fow_rev_flag[%d] = %u \n", i, is_smem_fow_rev_flag[i]);
		}

	 }
         for (int i=0; i < n_smems_fow_rev[0]; i++)
         	printf("seed_intervals_fow_rev_compact_gpu[%d] = %u\n", i, seed_intervals_fow_rev_compact[i].x >> 1);
 
}


int bns_pos2rid(const bntseq2_t *bns, int64_t pos_f)
{
        int left, mid, right;
        if (pos_f >= bns->l_pac) return -1;
        left = 0; mid = 0; right = bns->n_seqs;
        while (left < right) { // binary search
                mid = (left + right) >> 1;
                if (pos_f >= bns->anns[mid].offset) {
                        if (mid == bns->n_seqs - 1) break;
                        if (pos_f < bns->anns[mid+1].offset) break; // bracketed
                        left = mid + 1;
                } else right = mid;
        }
        return mid;
}

int bns_cnt_ambi(const bntseq2_t *bns, int64_t pos_f, int len, int *ref_id)
{
        int left, mid, right, nn;
        if (ref_id) *ref_id = bns_pos2rid(bns, pos_f);
        left = 0; right = bns->n_holes; nn = 0;
        while (left < right) {
                mid = (left + right) >> 1;
                if (pos_f >= bns->ambs[mid].offset + bns->ambs[mid].len) left = mid + 1;
                else if (pos_f + len <= bns->ambs[mid].offset) right = mid;
                else { // overlap
                        if (pos_f >= bns->ambs[mid].offset) {
                                nn += bns->ambs[mid].offset + bns->ambs[mid].len < pos_f + len?
                                        bns->ambs[mid].offset + bns->ambs[mid].len - pos_f : len;
                        } else {
                                nn += bns->ambs[mid].offset + bns->ambs[mid].len < pos_f + len?
                                        bns->ambs[mid].len : len - (bns->ambs[mid].offset - pos_f);
                        }
                        break;
                }
        }
        return nn;
}

static inline int64_t infer_pos_ref(int64_t l_pac, int64_t pos, int len, int *is_rev)
{
        return (*is_rev = (pos >= l_pac))? (l_pac<<1) - 1 - pos - (len - 1) : pos;
}

void bad_read(FILE *fp, const char *fname, int scanres) {
	if (EOF == scanres) {
                fprintf(stderr, "Error reading %s : %s\n", fname, "Unexpected end of file");
        }
	fprintf(stderr, __func__, "Parse error reading %s\n", fname);
	exit(EXIT_FAILURE);
}

bntseq2_t *bns_restore_gpu(const char *prefix) {
	char ann_filename[1024], amb_filename[1024];

	FILE *fp;
	strcat(strcpy(ann_filename, prefix), ".ann");
	strcat(strcpy(amb_filename, prefix), ".amb");

	bntseq2_t *bns;
	char str[8192];
	int i;
	long long xx;
        int scanres;
	bns = (bntseq2_t*)calloc(1, sizeof(bntseq2_t));

	{ // read .ann
		fp = fopen(ann_filename, "r");
		if (fp == NULL) {
			perror("Unable to open .ann file.\n");
			exit(EXIT_FAILURE);
		}

		scanres = fscanf(fp, "%lld%d%u", &xx, &bns->n_seqs, &bns->seed);
		if (scanres != 3) bad_read(fp, ann_filename, scanres);
        	bns->anns = (bntann2_t*)calloc(bns->n_seqs, sizeof(bntann2_t));
		bns->l_pac = xx;
		for (i = 0; i < bns->n_seqs; ++i) {
        		bntann2_t *p = bns->anns + i;
        		char *q = str;
        		int c;
        		// read gi and sequence name
        		scanres = fscanf(fp, "%u%s", &p->gi, str);
        		if (scanres != 2) bad_read(fp, ann_filename, scanres);
        		p->name = strdup(str);
        		// read fasta comments
        		while (q - str < sizeof(str) - 1 && (c = fgetc(fp)) != '\n' && c != EOF) *q++ = c;
        		while (c != '\n' && c != EOF) c = fgetc(fp);
        		if (c == EOF) {
        		        scanres = EOF;
				bad_read(fp, ann_filename, scanres);
        		}
        		*q = 0;
        		if (q - str > 1 && strcmp(str, " (null)") != 0) p->anno = strdup(str + 1); // skip leading space
        		else p->anno = strdup("");
        		// read the rest
        		scanres = fscanf(fp, "%lld%d%d", &xx, &p->len, &p->n_ambs);
        		if (scanres != 3) bad_read(fp, ann_filename, scanres);
        		p->offset = xx;
    		}
		int ret = fclose(fp);
        	if (ret != 0) {
			fprintf(stderr, "Error closing .ann file: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}
	}
	{ // read .amb
		int64_t l_pac;
		int32_t n_seqs;
		fp = fopen(amb_filename, "r");
		if (fp == NULL) {
			perror("Unable to open .amb file.\n");
			exit(EXIT_FAILURE);
		}

        	scanres = fscanf(fp, "%lld%d%d", &xx, &n_seqs, &bns->n_holes);
        	if (scanres != 3) bad_read(fp, amb_filename, scanres);
		l_pac = xx;
        	if (l_pac != bns->l_pac || n_seqs != bns->n_seqs) {
		       	perror("Inconsistent .ann and .amb files.\n");
			exit(EXIT_FAILURE);
		}
        	bns->ambs = bns->n_holes? (bntamb2_t*)calloc(bns->n_holes, sizeof(bntamb2_t)) : 0;
        	for (int i = 0; i < bns->n_holes; ++i) {
        	        bntamb2_t *p = bns->ambs + i;
        	        scanres = fscanf(fp, "%lld%d%s", &xx, &p->len, str);
        	        if (scanres != 3) bad_read(fp, amb_filename, scanres);
        	        p->offset = xx;
        	        p->amb = str[0];
        	}

		int ret = fclose(fp);
        	if (ret != 0) {
			fprintf(stderr, "Error closing .amb file: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}
	}

	return bns;
}

void bns_destroy(bntseq2_t *bns) {
	if (bns == 0) return;
        else {
                int i;
		free(bns->ambs);
                for (i = 0; i < bns->n_seqs; ++i) {
                        free(bns->anns[i].name);
                        free(bns->anns[i].anno);
                }
                free(bns->anns);
		free(bns);
        }
}

void bwt_restore_sa_gpu(const char *fn, bwt_t_gpu *bwt)
{
	nvtxRangePushA("bwt_restore_sa_gpu");
	char skipped[256];
	FILE *fp;
	bwtint_t_gpu primary;
	fp = fopen(fn, "rb");
	if (fp == NULL){
		fprintf(stderr, "Unable to open .sa file.\n");
		exit(1);
	}
	fread(&primary, sizeof(bwtint_t_gpu), 1, fp);
	if (primary != bwt->primary){
		fprintf(stderr, "SA-BWT inconsistency: primary is not the same.\n");
		exit(EXIT_FAILURE);
	}
	fread(skipped, sizeof(bwtint_t_gpu), 4, fp); // skip
	fread(&bwt->sa_intv, sizeof(bwtint_t_gpu), 1, fp);
	fread(&primary, sizeof(bwtint_t_gpu), 1, fp);
	//fprintf(stderr,"bwt_size=%llu\n",bwt->bwt_size);
	//printf("[GPU] Primary: %llu and seq_len: %llu\n",bwt->primary,bwt->seq_len);
	//printf("[GPU] Sa_intv: %d and bwt_size: %llu\n",bwt->sa_intv,bwt->bwt_size);
    //fprintf(stderr, "L2[0]=%llu\n", bwt->L2[0]);
    //fprintf(stderr, "L2[1]=%llu\n", bwt->L2[1]);
    //fprintf(stderr, "L2[2]=%llu\n", bwt->L2[2]);
    //fprintf(stderr, "L2[3]=%llu\n", bwt->L2[3]);
    //fprintf(stderr, "L2[4]=%llu\n", bwt->L2[4]);	

	if (primary != bwt->seq_len){
		fprintf(stderr, "SA-BWT inconsistency: seq_len is not the same.\n");
		exit(EXIT_FAILURE);
	}

	bwt->n_sa = (bwt->seq_len + bwt->sa_intv) / bwt->sa_intv;
	gpuErrchk( cudaMallocHost((uint32_t**)&(bwt->sa), (bwt->n_sa)*(sizeof(uint32_t))));
	//bwt->sa = (uint32_t*)calloc(bwt->n_sa, sizeof(uint32_t));

	bwt->sa[0] = -1;

	fread(bwt->sa + 1, sizeof(uint32_t), bwt->n_sa - 1, fp);
	fread(&bwt->pack_size, sizeof(uint8_t), 1, fp);
	
	gpuErrchk(cudaMallocHost((uint32_t**)&(bwt->sa_upper_bits), (bwt->pack_size * bwt->n_sa / 32 + 1)*(sizeof(uint32_t))));
	//bwt->sa_upper_bits = (uint32_t*)calloc(bwt->pack_size * bwt->n_sa / 32 + 1, sizeof(uint32_t));


	fread(bwt->sa_upper_bits, sizeof(uint32_t), bwt->pack_size * bwt->n_sa / 32 + 1, fp);
    	bwt->sa_upper_bits[0] |= 0x1;
	fclose(fp);
	nvtxRangePop();
}

bwt_t_gpu *bwt_restore_bwt_gpu(const char *fn)
{
	nvtxRangePushA("bwt_restore_bwt_gpu");
	bwt_t_gpu *bwt;
	FILE *fp;
	//bwt = (bwt_t_gpu*)calloc(1, sizeof(bwt_t_gpu));
	gpuErrchk( cudaMallocHost((bwt_t_gpu**)&bwt, sizeof(bwt_t_gpu)) );
	fp = fopen(fn, "rb");
	if (fp == NULL){
		fprintf(stderr, "Unable to othread_read_numpen .bwt file.\n");
		exit(1);
	}
	fseek(fp, 0, SEEK_END);
	//printf("ftell %ld\n",ftell(fp));
	bwt->bwt_size = (ftell(fp) - sizeof(bwtint_t_gpu) * 5) >> 2;
	//bwt->bwt = (uint32_t*)calloc(bwt->bwt_size, sizeof(uint32_t));
	//bwt->L2 = (bwtint_t_gpu*)calloc(5, sizeof(bwtint_t_gpu));
	gpuErrchk(cudaMallocHost((uint32_t**)&(bwt->bwt), bwt->bwt_size*sizeof(uint32_t)));
	gpuErrchk(cudaMallocHost((bwtint_t_gpu**)&(bwt->L2), 5*sizeof(bwtint_t_gpu)));
	
	fseek(fp, 0, SEEK_SET);
	fread(&bwt->primary, sizeof(bwtint_t_gpu), 1, fp);
	fread(bwt->L2+1, sizeof(bwtint_t_gpu), 4, fp);
	fread(bwt->bwt, 4, bwt->bwt_size, fp);
	bwt->seq_len = bwt->L2[4];
	fclose(fp);
	nvtxRangePop();

	return bwt;

}


void bwt_destroy_gpu(bwt_t_gpu *bwt){
	if (bwt == 0) return;
	cudaFreeHost(bwt->sa); cudaFreeHost(bwt->sa_upper_bits); cudaFreeHost(bwt->bwt);
	cudaFreeHost(bwt->L2); cudaFreeHost(bwt);
}


void  print_seq_ascii(int length, char *seq){
   int i;
   //fprintf(stderr,"seq length = %d: ", length);
   for (i = 0; i < length; ++i) {
      putc(seq[i], stdout);
   }
   printf("\n");
}

void  print_seq_dna(int length, uint8_t* seq){
   int i;
   //fprintf(stderr,"seq length = %d: ", length);
   for (i = 0; i < length; ++i) {
      putc("ACGTN"[(int)seq[i]], stdout);
   }
   fprintf(stderr,"\n");
}

void  print_seq_packed(int length, uint32_t* seq, int start, int fow){
   int i;
   //fprintf(stderr,"seq length = %d: ", length);
   if (fow) {
	   for (i = start; i < length + start; ++i) {
		   int reg_no = i >> 3;
		   int reg_pos = i & 7;
		   int reg = seq[reg_no];
		   int base = (reg >> (28 - (reg_pos << 2))) & 15;
		   //fprintf(stdout,"%x, ", reg);
		   putc("ACGTNP"[base], stdout);
		   //fflush(stdout);
	   }
   }
   else {
	   for (i = start + length - 1; i >= start; i--) {
		   int reg_no = i >> 3;
		   int reg_pos = i & 7;
		   int reg = seq[reg_no];
		   int base = (reg >> (28 - (reg_pos << 2))) & 15;
		   //fprintf(stdout,"%x, ", reg);
		   putc("ACGTNP"[base], stdout);
		   //fflush(stdout);
	   }
   }
   //fprintf(stderr,"\n");
}

bwt_t_gpu gpu_cpy_wrapper(bwt_t_gpu *bwt){

    bwt_t_gpu bwt_gpu;
	//cudaStream_t stream1, stream2;
	//cudaError_t result;
	//result = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	//result = cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	gpuErrchk(cudaMalloc(&(bwt_gpu.bwt), bwt->bwt_size*sizeof(uint32_t)));
	gpuErrchk(cudaMalloc(&(bwt_gpu.sa), bwt->n_sa*sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&(bwt_gpu.sa_upper_bits), (bwt->pack_size * bwt->n_sa / 32 + 1)*sizeof(uint32_t)));


	gpuErrchk(cudaMemcpy(bwt_gpu.bwt, bwt->bwt, bwt->bwt_size*sizeof(uint32_t),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bwt_gpu.sa, bwt->sa, bwt->n_sa*sizeof(uint32_t),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bwt_gpu.sa_upper_bits, bwt->sa_upper_bits, (bwt->pack_size * bwt->n_sa / 32 + 1)*sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	//gpuErrchk(gpuErrchk(cudaMemcpyAsync(bwt_gpu.bwt, bwt->bwt, bwt->bwt_size*sizeof(uint32_t),cudaMemcpyHostToDevice, stream1));
    //gpuErrchk(gpuErrchk(cudaMemcpyAsync(bwt_gpu.sa, bwt->sa, bwt->n_sa*sizeof(bwtint_t_gpu),cudaMemcpyHostToDevice, stream2));
    bwt_gpu.pack_size = bwt->pack_size;
	bwt_gpu.primary = bwt->primary;
    bwt_gpu.seq_len = bwt->seq_len;
    bwt_gpu.sa_intv = bwt->sa_intv;
    //fprintf(stderr, "SA intv %d\n", bwt->sa_intv);
    bwt_gpu.n_sa = bwt->n_sa;
    gpuErrchk(cudaMemcpyToSymbol(L2_gpu, bwt->L2, 5*sizeof(bwtint_t_gpu), 0, cudaMemcpyHostToDevice));		
	//
	//result = cudaStreamDestroy(stream1);
	//result = cudaStreamDestroy(stream2);
	bwt_destroy_gpu(bwt);

	return bwt_gpu;
}

void pre_calc_seed_intervals_wrapper(uint2 *pre_calc_seed_intervals, int pre_calc_seed_len, bwt_t_gpu bwt_gpu){

    gpuErrchk(cudaMalloc(&pre_calc_seed_intervals, (1 << (pre_calc_seed_len<<1))*sizeof(uint2)));
    int threads_per_block_pre_calc_seed = 128;
    int num_blocks_pre_calc_seed = ((1 << (pre_calc_seed_len<<1)) + threads_per_block_pre_calc_seed - 1)/threads_per_block_pre_calc_seed;
    pre_calc_seed_intervals_gpu<<<num_blocks_pre_calc_seed, threads_per_block_pre_calc_seed>>>(pre_calc_seed_intervals, pre_calc_seed_len, bwt_gpu, (1 << (pre_calc_seed_len<<1)));
	////cudaDeviceSynchronize();
}

void free_gpuseed_data(gpuseed_storage_vector *gpuseed_data){
	//cudaFreeHost(gpuseed_data->pre_calc_seed_intervals);
	cudaFree(gpuseed_data->pre_calc_seed_intervals);
	cudaFree(gpuseed_data->bwt_gpu.bwt);
	cudaFree(gpuseed_data->bwt_gpu.sa);
	cudaFree(gpuseed_data->bwt_gpu.sa_upper_bits);
}

void mem_print_gpuseed(mem_seed_v_gpu *data, int n_reads) {
		for (int i = 0; i < n_reads; i++)
		{
			printf("=====> Printing SMEM(s) in read '%d' Total : %d (Accum: %d)<=====\n", i+1, data->n_ref_pos_fow_rev_results[i], data->n_ref_pos_fow_rev_prefix_sums[i]);
			for (int j = data->n_ref_pos_fow_rev_prefix_sums[i]; j < (data->n_ref_pos_fow_rev_prefix_sums[i] + data->n_ref_pos_fow_rev_results[i]); j++){
				printf("[GPUSeed Host] Read[%d, %d] -> %lu (Score: %lu)\n",data->qbeg[j].x, data->qbeg[j].y, data->rbeg[j], data->score[j]);
				//printf("[GPUSeed Host] Read[%d, %d] -> %lu\n",data->qbeg[j].x, data->qbeg[j].y, data->rbeg[j]);
			}
		}
		printf("\n");
}

void mem_rev_print_gpuseed(mem_seed_v_gpu *data, int n_reads) {
		for (int i = 0; i < n_reads; i++)
		{
			printf("=====> Printing SMEM(s) in read '%d' Total : %d (Accum: %d)<=====\n", i+1, data->n_ref_pos_fow_rev_results[i], data->n_ref_pos_fow_rev_prefix_sums[i]);
			for (int j = (data->n_ref_pos_fow_rev_prefix_sums[i] + data->n_ref_pos_fow_rev_results[i]); j > ((data->n_ref_pos_fow_rev_prefix_sums[i])); j--){
				printf("[GPUSeed Host] Read[%d, %d] -> %lu (Score: %lu)\n",data->qbeg[j - 1].x, data->qbeg[j - 1].y, data->rbeg[j - 1], data->score[j - 1]);		
			}
		}
		printf("\n");
}

unsigned char seq_nt4_table[256] = {
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
};

//#define READ_BATCH_SIZE 100 //30000
#define BASE_BATCH_SIZE 1000000
#define OUTPUT_SIZE_MUL 3

#ifdef __cplusplus
extern "C" {
#endif

mem_seed_v_gpu *seed_gpu(gpuseed_storage_vector *gpuseed_data) {
	
	//fprintf(stderr, "I go to seed with n_reads: %d and Processed: %ld\n",n_reads, n_processed);

	int min_seed_size = gpuseed_data->min_seed_size;
	int is_smem = gpuseed_data->is_smem;
	int mummer_format = 0;
	int print_stats = 0;
	int c;
	
	
	double total_time = realtime_gpu();
	mem_seed_v_gpu *from_gpu_results = (mem_seed_v_gpu*)(malloc(sizeof(mem_seed_v_gpu)));
	uint32_t *n_ref_pos_fow_rev_prefix_sums_gpu;
	uint32_t *n_ref_pos_fow_rev_prefix_sums_gpu_results;
	gpuErrchk(cudaMalloc(&(n_ref_pos_fow_rev_prefix_sums_gpu), 100000000*sizeof(uint32_t)));

	void *cub_scan_temp_storage_end = NULL;
	size_t cub_scan_storage_bytes_end = 0;

    int device_no;
    gpuErrchk(cudaGetDevice(&device_no));
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, device_no);	

	cudaError_t result;

	int nStreams = 2;
	cudaStream_t stream[nStreams];

  	for (int i = 0; i < nStreams; ++i){
		result = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

	if (from_gpu_results == NULL) {
        //fprintf(stderr, "Fatal: failed to allocate %zu bytes.\n", n_reads * sizeof(mem_seed_v));
        abort();
    }

	if (print_stats)
    	fprintf(stderr, "\n-----------------------------------------------------------------------------------------------------------\n");
	FILE *read_file = fopen(gpuseed_data->read_file, "r");
	fseek(read_file, gpuseed_data->file_bytes_skip, SEEK_SET);
	int all_done = 0;
	char *read_batch;
	uint32_t *read_offsets;
	uint32_t *read_sizes;
	gpuErrchk(cudaMallocHost((char**)&read_batch, BASE_BATCH_SIZE+1e6));
	gpuErrchk(cudaMallocHost((uint32_t**)&read_offsets, 1e6));
	gpuErrchk(cudaMallocHost((uint32_t**)&read_sizes, 1e6));

	double total_gpu_time = 0.0, total_batch_load_time=0.0, total_batch_prep_time = 0.0, total_mem_time = 0.0, total_print_time=0.0;
	double total_find_seed_intervals_time =0.0, total_filter_seed_intervals_time =0.0, total_locate_seeds_time=0.0;
	double total_find_seed_intervals_fow_time = 0.0, total_find_seed_intervals_back_time = 0.0;
	int reads_processed = 0;
	int max_read_size = 0;
	//int read_count = seqs[0].id;
	int reads_loaded = 0;
	int seed_counter = 0;
	uint64_t file_bytes = 0;
	int m = 0;

	int start_read = 0;
	
	while (!all_done) {
		int total_reads = 0;
		int total_bytes = 0;
		int read_batch_size = 0;
		//int n_seed_cands = 0;
		//char *all_reads_fill_ptr = all_reads;
		char *read_batch_fill_ptr = read_batch;
		double loading_time = realtime_gpu();
		int prev_len = 0;
		while (read_batch_size < BASE_BATCH_SIZE) {
			
			size_t len;
			char *line = NULL;
			int n_bases = getline(&line, &len, read_file);
			file_bytes += n_bases;
			if (n_bases < 0){
				all_done = 1;
				break;
			}
			if (line[0] != '>') {	
				memcpy(read_batch_fill_ptr, line, n_bases - 1);
				total_bytes = total_bytes + n_bases;
				read_batch_fill_ptr  += (n_bases - 1);
				read_batch_size += (n_bases - 1);
				int read_len = n_bases - 1;

				read_offsets[total_reads] = total_reads == 0 ? 0 : read_offsets[total_reads - 1] + prev_len;
				read_sizes[total_reads] = n_bases - 1;
				prev_len = read_len;
				total_reads++;
				reads_loaded++;

				if ((n_bases - 1) > max_read_size) max_read_size = n_bases - 1;
				//if (reads_loaded == n_reads) {
					//all_done = 1;
					//break;
				//}				
			}
			free(line);
		}
		

		int n_bits_max_read_size = (int)ceil(log2((double)max_read_size));
		total_batch_load_time += (realtime_gpu() - loading_time);
		if (print_stats)
			fprintf(stderr,"A batch of %d reads loaded from file in %.3f seconds\n", total_reads, realtime_gpu() - loading_time);

		int read_batch_size_8 = read_batch_size%8 ? read_batch_size + (8 - read_batch_size%8) : read_batch_size;
		uint8_t *read_batch_gpu;
		uint32_t *packed_read_batch_fow, *packed_read_batch_rev, *read_sizes_gpu, *read_offsets_gpu, *n_smems_fow, *n_smems_rev, *n_smems_fow_rev; 
		uint32_t *n_seeds_fow_rev;

		uint32_t *thread_read_num; //, *thread_read_idx;
		int2  *smem_intv_read_pos_fow_rev;
		uint32_t *n_smems_sum_fow_rev_gpu;
		uint32_t n_smems_sum_fow_rev;
		uint32_t *n_seeds_sum_fow_rev_gpu;
		uint32_t n_seeds_sum_fow_rev;

		uint32_t* n_smems_max_gpu;

		uint32_t *n_smems_fow_rev_scan;
		uint32_t *n_seeds_fow_rev_scan;

		void *cub_scan_temp_storage = NULL;
		void *cub_scan_temp_storage2 = NULL;
		size_t cub_scan_storage_bytes = 0;
		size_t cub_scan_storage_bytes2 = 0;
		void *cub_select_temp_storage = NULL;
		size_t cub_select_storage_bytes = 0;
		void *cub_sort_temp_storage = NULL;
		size_t cub_sort_storage_bytes = 0;

		CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_fow_rev_scan, 2*(read_batch_size_8 - (total_reads*(min_seed_size-1)))));
		if (print_stats)
			fprintf(stderr, "ExclusiveSum bytes for n_smems = %d\n", cub_scan_storage_bytes);

		int max_output_size = 2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads));

		max_output_size = max_output_size > (2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)) + (read_batch_size >> 3) + 2*total_reads + read_batch_size_8 >> 2 + (read_batch_size_8 - ((min_seed_size-1)*total_reads))) ? max_output_size : (2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)) + (read_batch_size >> 3) + 2*total_reads + read_batch_size_8 >> 2 + (read_batch_size_8 - ((min_seed_size-1)*total_reads)));

		gpuErrchk(cudaMalloc(&read_batch_gpu, read_batch_size_8));
		gpuErrchk(cudaMalloc(&read_sizes_gpu, total_reads*sizeof(uint32_t)));
		gpuErrchk(cudaMalloc(&read_offsets_gpu,total_reads*sizeof(uint32_t)));
		n_smems_fow_rev = read_sizes_gpu;
		gpuErrchk(cudaMalloc(&n_smems_fow_rev_scan,(total_reads+1)*sizeof(uint32_t)));

		uint32_t *n_smems;
		gpuErrchk(cudaMalloc(&n_smems, total_reads * sizeof(uint32_t)));

		gpuErrchk(cudaMalloc(&packed_read_batch_fow,(read_batch_size_8 >> 3)*sizeof(uint32_t)));
		gpuErrchk(cudaMalloc(&n_seeds_fow_rev_scan, (((read_batch_size_8 - ((min_seed_size-1)*total_reads))) + 1)*sizeof(uint2)));

		gpuErrchk(cudaMalloc(&cub_scan_temp_storage,cub_scan_storage_bytes));
		n_smems_sum_fow_rev_gpu = &n_smems_fow_rev_scan[total_reads];
		uint32_t *n_sum_intervals_fow_gpu;
		gpuErrchk(cudaMalloc(&n_sum_intervals_fow_gpu, sizeof(uint32_t))); 
		gpuErrchk(cudaMalloc(&n_smems_max_gpu, sizeof(uint32_t)));

		uint2 *seed_intervals_fow_rev_gpu;
		int2 *seed_read_pos_fow_rev_gpu;
		uint2 *seed_intervals_fow_rev_compact_gpu;
		int2 *seed_read_pos_fow_rev_compact_gpu;

		uint8_t *is_smem_flags;
		uint32_t *n_ref_pos_fow_rev_gpu;
		uint32_t *n_ref_pos_fow_rev_scan;
		bwtint_t_gpu *seed_ref_pos_fow_rev_gpu;
		bwtint_t_gpu *seed_sa_idx_fow_rev_gpu;

		// for my kernels
		int2 *seed_read_pos_gpu, *seed_read_pos_compact_gpu;
		uint2 *seed_intervals_gpu, *seed_intervals_compact_gpu, *seed_intervals_fow_compact_gpu;


		if (is_smem) {
			CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_compact_gpu, (uint64_t*)seed_read_pos_gpu, (uint64_t*)seed_intervals_fow_rev_compact_gpu, (uint64_t*)seed_intervals_fow_rev_gpu, 2*(read_batch_size_8 - total_reads*(min_seed_size-1)),total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size));
			if (print_stats)
			fprintf(stderr, "Sort bytes = %d\n", cub_sort_storage_bytes);
		} else {
			CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_compact_gpu, (uint64_t*)seed_read_pos_gpu, seed_sa_idx_fow_rev_gpu, seed_ref_pos_fow_rev_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - total_reads*(min_seed_size-1)), total_reads, n_ref_pos_fow_rev_scan, n_ref_pos_fow_rev_scan + 1, 0, n_bits_max_read_size));
			if (print_stats)
			fprintf(stderr, "Sort bytes = %d\n", cub_sort_storage_bytes);
		}


		uint4 *seed_intervals_pos_gpu;
		uint4 *read_pos_num_len_intervals, *read_pos_num_len_intervals_compact;
		gpuErrchk(cudaMalloc(&seed_intervals_pos_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - total_reads*(min_seed_size-1))*sizeof(uint2)));
		seed_read_pos_gpu = (int2*)seed_intervals_pos_gpu;
		seed_intervals_gpu = (uint2*) &seed_intervals_pos_gpu[(read_batch_size_8 - total_reads*(min_seed_size-1))/2];
		read_pos_num_len_intervals = seed_intervals_pos_gpu;
		uint4 *seed_intervals_pos_compact_gpu;
		gpuErrchk(cudaMalloc(&seed_intervals_pos_compact_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - total_reads*(min_seed_size-1))*sizeof(uint2)));
		seed_read_pos_compact_gpu = (int2*)seed_intervals_pos_compact_gpu;
		seed_intervals_compact_gpu = (uint2*) &seed_intervals_pos_compact_gpu[(read_batch_size_8 - total_reads*(min_seed_size-1))/2];
		read_pos_num_len_intervals_compact = seed_intervals_pos_compact_gpu;

		gpuErrchk(cudaMalloc(&n_seeds_fow_rev, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - total_reads*(min_seed_size-1))*sizeof(bwtint_t_gpu))); // allocating memory for seed_sa_idx later

		uint8_t *is_interval_flags;
		is_interval_flags = (uint8_t*) n_seeds_fow_rev;


		if(!is_smem) gpuErrchk(cudaMalloc(&seed_ref_pos_fow_rev_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - total_reads*(min_seed_size-1))*sizeof(bwtint_t_gpu)));

		gpuErrchk(cudaMalloc(&cub_sort_temp_storage,cub_sort_storage_bytes));
		n_ref_pos_fow_rev_gpu = read_offsets_gpu;


		double gpu_batch_time = realtime_gpu();
		double mem_time0 = realtime_gpu();
		gpuErrchk(cudaMemcpy(read_batch_gpu, (uint8_t*)read_batch, read_batch_size, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(read_offsets_gpu, read_offsets, total_reads*sizeof(uint32_t), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(read_sizes_gpu, read_sizes, total_reads*sizeof(uint32_t), cudaMemcpyHostToDevice));
		mem_time0 = realtime_gpu() - mem_time0;
		double batch_prep_time = realtime_gpu();
		int BLOCKDIM = 128;
		double fow_pack_time_start = realtime_gpu();
		int N_BLOCKS = ((read_batch_size_8 >> 3)  + BLOCKDIM - 1)/BLOCKDIM;
		pack_4bit_fow<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)read_batch_gpu, packed_read_batch_fow, /*thread_read_num, read_offsets_gpu, read_sizes_gpu,*/ (read_batch_size_8 >> 3));
		//cudaDeviceSynchronize();
		double fow_pack_time = realtime_gpu() - fow_pack_time_start;
		if (print_stats)
		fprintf(stderr,"\tForward batch packed on GPU in %.3f seconds\n",fow_pack_time);
		if (print_stats)
		fprintf(stderr, "Processing %d reads on GPU...\n", total_reads);

		double find_seeds_time = realtime_gpu();
		
		gpuErrchk(cudaMemset(n_smems, 0, total_reads*sizeof(uint32_t)));
		gpuErrchk(cudaMemset(is_interval_flags, 0, (read_batch_size_8 - total_reads*(min_seed_size-1))*sizeof(uint8_t)));


		int n_fow_cands = total_reads;
		BLOCKDIM = 32;
		if (max_read_size < 200) {
			cudaFuncSetCacheConfig(find_seed_intervals_fow, cudaFuncCachePreferL1); // increase L1 size for very short reads
			while (((BLOCKDIM*max_read_size*sizeof(int))>>3) > 16000) BLOCKDIM /= 2; // use this with preferred L1 cache
		}
		else 
			while (((BLOCKDIM*max_read_size*sizeof(int))>>3) > dev_prop.sharedMemPerBlock) BLOCKDIM /= 2; // adjust block dimensions to fit into shared memory
		if (BLOCKDIM < 4) {
			if (print_stats)
			fprintf(stderr, "This variant of GPUseed is intended for short reads. One or more reads are very long and cannot/shouldn't be processed.\n");
			exit(1);
		}
			
		N_BLOCKS = (n_fow_cands + BLOCKDIM - 1)/BLOCKDIM;
		int shared_size = (max_read_size*BLOCKDIM >> 3) * sizeof(int);
		find_seed_intervals_fow<<<N_BLOCKS, BLOCKDIM, shared_size>>>(gpuseed_data->bwt_gpu, packed_read_batch_fow, read_batch_size_8, read_sizes_gpu, read_offsets_gpu, min_seed_size, (uint8_t*) is_interval_flags, max_read_size, read_pos_num_len_intervals, n_fow_cands); 
	
		//cudaDeviceSynchronize();
		total_find_seed_intervals_fow_time += realtime_gpu() - find_seeds_time;
		if (print_stats)
		fprintf(stderr,"\tIntervals forward time is %.3f seconds on GPU\n", realtime_gpu() - find_seeds_time);
		
		// prepare thread assignments for backward search
		
		/* init of cub kernels */

		double init_cub_time = realtime_gpu();

		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint4*)read_pos_num_len_intervals, (uint8_t*) is_interval_flags, (uint4*)read_pos_num_len_intervals_compact, n_sum_intervals_fow_gpu, (read_batch_size_8 - total_reads*(min_seed_size-1))));
		if (print_stats)
		fprintf(stderr, "Flagged bytes for seed_intervals_fow = %d\n", cub_select_storage_bytes);
		
		gpuErrchk(cudaMalloc(&cub_select_temp_storage,cub_select_storage_bytes));
		
		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tInit cub test time is %.3f seconds on GPU\n", realtime_gpu() - init_cub_time);

		/*   end of cub init  */

		double cub_kernels_time = realtime_gpu();
	
		// valid read positions, numbers and lengths for back-search
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint4*)read_pos_num_len_intervals, (uint8_t*) is_interval_flags, (uint4*)read_pos_num_len_intervals_compact, n_sum_intervals_fow_gpu, (read_batch_size_8 - total_reads*(min_seed_size-1))));

		uint32_t n_sum_intervals_fow;
		gpuErrchk(cudaMemcpy(&n_sum_intervals_fow, n_sum_intervals_fow_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tCub kernels time is %.3f seconds on GPU\n", realtime_gpu() - cub_kernels_time);
		
		// second kernel for backward search

		double find_seeds_back_time = realtime_gpu();

		int n_back_cands = n_sum_intervals_fow;
		// debugging
		if (print_stats)
		fprintf(stderr, "total_reads = %d, n_back_cands = %d\n", total_reads, n_back_cands);
		is_smem_flags = (uint8_t*) is_interval_flags;
		gpuErrchk(cudaMemset(is_smem_flags, 0, (read_batch_size_8 - total_reads*(min_seed_size-1))*sizeof(uint8_t)));

		BLOCKDIM = 64;
		N_BLOCKS = (n_back_cands + BLOCKDIM - 1)/BLOCKDIM;
		cudaFuncSetCacheConfig(find_seed_intervals_back, cudaFuncCachePreferL1);
		// second kernel from original implem.
		find_seed_intervals_back<<<N_BLOCKS, BLOCKDIM>>>(packed_read_batch_fow, read_sizes_gpu, read_offsets_gpu, seed_intervals_gpu, seed_read_pos_gpu, (uint8_t*) is_smem_flags, n_smems, min_seed_size, gpuseed_data->bwt_gpu, read_pos_num_len_intervals_compact, n_back_cands);

		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tFind seeds back time is %.3f seconds on GPU\n", realtime_gpu() - find_seeds_back_time);
		total_find_seed_intervals_time += realtime_gpu() - find_seeds_time;
		total_find_seed_intervals_back_time += realtime_gpu() - find_seeds_back_time;

		// keep only valid intervals and positions

		double flagged_find_intervals_time = realtime_gpu();

		gpuErrchk(cudaMalloc(&cub_select_temp_storage,cub_select_storage_bytes));

		//seed_intervals_compact_gpu = seed_intervals_fow_compact_gpu;
		//seed_intervals_compact_gpu = (uint2*) read_pos_num_len_intervals_compact;
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint2*)seed_intervals_gpu, (uint8_t*)is_smem_flags, (uint2*)seed_intervals_compact_gpu, n_smems_sum_fow_rev_gpu, (read_batch_size_8 - total_reads*(min_seed_size-1))));
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (int2*)seed_read_pos_gpu, (uint8_t*)is_smem_flags, (int2*)seed_read_pos_compact_gpu, n_smems_sum_fow_rev_gpu, (read_batch_size_8 - total_reads*(min_seed_size-1))));

		
		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tFlagged operations on seed_intervals and read_pos is %.3f seconds on GPU\n", realtime_gpu() - flagged_find_intervals_time);

		/*		END			*/

		double n_smems_max_time = realtime_gpu();
		CubDebugExit(cub::DeviceReduce::Max(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems, &n_smems_max_gpu[0], total_reads));
		// cub::DeviceReduce::Max(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems_fow, n_smems_max_gpu, total_reads);
		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tMax in n_smems found in %.3f seconds\n", realtime_gpu() - n_smems_max_time);


		double filter_seeds_time = realtime_gpu();

		N_BLOCKS = (total_reads + BLOCKDIM - 1)/BLOCKDIM;

		// sum_arrays<<<N_BLOCKS, BLOCKDIM>>>(n_smems_fow, n_smems_rev, n_smems_fow_rev, total_reads);
		// this has a different outcome for some reason..
		n_smems_fow_rev = n_smems;

		double n_smems_fow_rev_scan_time = realtime_gpu();
		CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems_fow_rev, n_smems_fow_rev_scan, total_reads));
		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tn_smems_fow_rev scanned in %.3f seconds on GPU\n", realtime_gpu() - n_smems_fow_rev_scan_time);

		gpuErrchk(cudaMemset(n_ref_pos_fow_rev_gpu, 0, total_reads*sizeof(uint32_t)));
		//n_seeds_fow_rev = (bwtint_t_gpu*) read_pos_num_len_compact;
		//gpuErrchk(cudaMemset(n_seeds_fow_rev, 0, (read_batch_size_8-total_reads*(min_seed_size-1))*sizeof(bwtint_t_gpu)));

		//filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_compact_gpu, seed_intervals_rev_compact_gpu, seed_read_pos_fow_compact_gpu, seed_read_pos_rev_compact_gpu, n_smems_fow, n_smems_rev,  n_smems_fow_scan, n_smems_rev_scan, n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads);
		if (is_smem) {
			filter_seed_intervals_gpu_wrapper<<<1, 1>>>(seed_intervals_compact_gpu, seed_read_pos_compact_gpu, seed_intervals_gpu, seed_read_pos_gpu, n_smems, n_smems_rev, n_smems_fow_rev,  n_seeds_fow_rev, n_smems_fow_rev_scan,  n_ref_pos_fow_rev_gpu, n_smems_max_gpu, n_smems_sum_fow_rev_gpu, cub_sort_temp_storage, cub_sort_storage_bytes, total_reads, n_bits_max_read_size, is_smem/*n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads*/);
			int2 *swap =  seed_read_pos_compact_gpu;
			seed_read_pos_compact_gpu = seed_read_pos_gpu;
			seed_read_pos_gpu = swap;
			swap = (int2*) seed_intervals_compact_gpu;
			seed_intervals_compact_gpu = seed_intervals_gpu;
			seed_intervals_gpu = (uint2*) swap;
		}
		else {
			filter_seed_intervals_gpu_wrapper_mem<<<1, 1>>>(seed_intervals_compact_gpu, seed_read_pos_compact_gpu, seed_intervals_gpu, n_smems, n_smems_rev, n_smems_fow_rev,  n_seeds_fow_rev,  n_smems_fow_rev_scan, n_ref_pos_fow_rev_gpu, n_smems_max_gpu, n_smems_sum_fow_rev_gpu, total_reads/*n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads*/);

			int2 *swap =  seed_read_pos_compact_gpu;
			seed_read_pos_compact_gpu = seed_read_pos_gpu;
			seed_read_pos_gpu = swap;
		}


		//cudaDeviceSynchronize();

		gpuErrchk(cudaMemcpy(&n_smems_sum_fow_rev, n_smems_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		if (print_stats)
		fprintf(stderr,"\tSMEM seeds filtered in %.3f seconds on GPU\n", realtime_gpu() - n_smems_max_time);
		total_filter_seed_intervals_time += realtime_gpu() - n_smems_max_time;

		double locate_seeds_time = realtime_gpu();

		n_seeds_sum_fow_rev_gpu = n_sum_intervals_fow_gpu;

		double n_seeds_fow_rev_sum_time = realtime_gpu();
		CubDebugExit(cub::DeviceReduce::Sum(cub_scan_temp_storage2, cub_scan_storage_bytes2, n_seeds_fow_rev, n_seeds_sum_fow_rev_gpu, n_smems_sum_fow_rev));
		gpuErrchk(cudaMalloc(&cub_scan_temp_storage2,cub_scan_storage_bytes2));
		CubDebugExit(cub::DeviceReduce::Sum(cub_scan_temp_storage2, cub_scan_storage_bytes2, n_seeds_fow_rev, n_seeds_sum_fow_rev_gpu, n_smems_sum_fow_rev));

		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tn_seeds_fow_rev summed in %.3f seconds\n", realtime_gpu() - n_seeds_fow_rev_sum_time);

		double n_seeds_fow_rev_scan_time = realtime_gpu();
		CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_fow_rev_scan, n_smems_sum_fow_rev));
		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tn_seeds_fow_rev scanned in %.3f seconds on GPU\n", realtime_gpu() - n_seeds_fow_rev_scan_time);


		seed_sa_idx_fow_rev_gpu = (bwtint_t_gpu*) n_seeds_fow_rev;

		int2 *final_seed_read_pos_fow_rev_gpu;


		gpuErrchk(cudaMemcpy(&n_seeds_sum_fow_rev, n_seeds_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		// if(n_seeds_sum_fow_rev > OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - (min_seed_size*total_reads))) {
		if(n_seeds_sum_fow_rev > OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))) {
			// fprintf(stderr,"n_seeds_sum_fow_rev (%llu) is more than allocated size(%d)\n", n_seeds_sum_fow_rev, OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - (min_seed_size*total_reads)));
			if (print_stats)
			fprintf(stderr,"n_seeds_sum_fow_rev (%llu) is more than allocated size(%d)\n", n_seeds_sum_fow_rev, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)));
			exit(EXIT_FAILURE);
		}
		uint32_t *final_seed_scores_gpu;
		cudaMalloc(&final_seed_scores_gpu, n_seeds_sum_fow_rev*sizeof(uint32_t));

		if (is_smem) {
			locate_seeds_gpu_wrapper<<<1, 1>>>(seed_read_pos_compact_gpu, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_gpu, seed_read_pos_gpu, n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, gpuseed_data->bwt_gpu, final_seed_scores_gpu);
			 final_seed_read_pos_fow_rev_gpu = seed_read_pos_compact_gpu;
			 seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;
		}
		else {
			locate_seeds_gpu_wrapper_mem<<<1, 1>>>(seed_read_pos_compact_gpu, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_gpu, seed_read_pos_gpu, n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, n_ref_pos_fow_rev_gpu, gpuseed_data->bwt_gpu, final_seed_scores_gpu);
			n_ref_pos_fow_rev_scan = n_smems_fow_rev_scan;
			gpuErrchk(cudaMemcpy(n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
			CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_ref_pos_fow_rev_gpu, n_ref_pos_fow_rev_scan, total_reads));
			CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_compact_gpu, (uint64_t*)seed_read_pos_gpu, seed_sa_idx_fow_rev_gpu, seed_ref_pos_fow_rev_gpu,  n_seeds_sum_fow_rev, total_reads, n_ref_pos_fow_rev_scan, n_ref_pos_fow_rev_scan + 1, 0, n_bits_max_read_size));
			final_seed_read_pos_fow_rev_gpu = seed_read_pos_gpu;
		}

		//cudaDeviceSynchronize();
		if (print_stats)
		fprintf(stderr,"\tSeeds located on ref in in %.3f seconds\n", realtime_gpu() - locate_seeds_time);
		total_locate_seeds_time += realtime_gpu() - locate_seeds_time;


		if (print_stats)
		fprintf(stderr, "n_seeds_sum_fow_rev = %d, n_smems_sum_fow = %d\n", n_seeds_sum_fow_rev, n_smems_sum_fow_rev);
		fflush(stderr);


		double mem_time1 = realtime_gpu();

		if (!reads_processed) {
			from_gpu_results->n_ref_pos_fow_rev_results = (uint32_t*)malloc(total_reads*sizeof(uint32_t));	
			from_gpu_results->qbeg = (int2*)malloc(n_seeds_sum_fow_rev * sizeof(int2));
			if (from_gpu_results->qbeg == NULL) {
				fprintf(stderr, "Realloc qbeg error\n");
				fflush(stderr);
			}
			from_gpu_results->rbeg = (bwtint_t_gpu*)malloc(n_seeds_sum_fow_rev * sizeof(bwtint_t_gpu));
			if (from_gpu_results->rbeg == NULL) fprintf(stderr, "Malloc rbeg error\n");
			from_gpu_results->score = (uint32_t*)malloc(n_seeds_sum_fow_rev * sizeof(uint32_t));
			//if (from_gpu_results->score == NULL) fprintf(stderr, "Malloc score error\n");

		}
		else {
			from_gpu_results->n_ref_pos_fow_rev_results = (uint32_t*)realloc(from_gpu_results->n_ref_pos_fow_rev_results,(total_reads + reads_processed) * sizeof(uint32_t));	
			from_gpu_results->qbeg = (int2*)realloc(from_gpu_results->qbeg, (seed_counter + n_seeds_sum_fow_rev) * sizeof(int2)); 
			if (from_gpu_results->qbeg == NULL) {
				fprintf(stderr, "Realloc qbeg error\n");
				fflush(stderr);
			}
			from_gpu_results->rbeg = (bwtint_t_gpu*)realloc(from_gpu_results->rbeg, (seed_counter + n_seeds_sum_fow_rev) * sizeof(bwtint_t_gpu));			
			if (from_gpu_results->rbeg == NULL) fprintf(stderr, "Realloc rbeg error\n");
			from_gpu_results->score = (uint32_t*)realloc(from_gpu_results->score, (seed_counter + n_seeds_sum_fow_rev) * sizeof(uint32_t));					
			//if (from_gpu_results->score == NULL) fprintf(stderr, "Realloc score error\n");
		}
		gpuErrchk(cudaMemcpy(&(from_gpu_results->qbeg[seed_counter]), final_seed_read_pos_fow_rev_gpu, n_seeds_sum_fow_rev*sizeof(int2), cudaMemcpyDeviceToHost));		
		gpuErrchk(cudaMemcpy(&(from_gpu_results->rbeg[seed_counter]), seed_ref_pos_fow_rev_gpu, n_seeds_sum_fow_rev*sizeof(bwtint_t_gpu), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&(from_gpu_results->score[seed_counter]), final_seed_scores_gpu, n_seeds_sum_fow_rev*sizeof(uint32_t), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&(from_gpu_results->n_ref_pos_fow_rev_results[reads_processed]), n_ref_pos_fow_rev_gpu, total_reads*sizeof(uint32_t), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&(n_ref_pos_fow_rev_prefix_sums_gpu[reads_processed]), n_ref_pos_fow_rev_gpu, total_reads*sizeof(uint32_t), cudaMemcpyDeviceToDevice));
		
		mem_time1 = (realtime_gpu() - mem_time1);
		
		if (print_stats)
			fprintf(stderr,"\tTime spent in cudaMemcpy is %.3f seconds\n", mem_time0 + mem_time1);
		total_mem_time += mem_time0 + mem_time1;

		if (print_stats)
			fprintf(stderr, "Total processing time of the batch on GPU is %.3f seconds\n", realtime_gpu() - gpu_batch_time);
		total_gpu_time += (realtime_gpu() - gpu_batch_time);

		double print_time = realtime_gpu();

		reads_processed += total_reads;
		seed_counter += n_seeds_sum_fow_rev;
		//if (reads_processed >= n_reads) all_done = 1;
		//fprintf(stderr,"Reads processed: %d\n",reads_processed);
		gpuErrchk(cudaFree(read_batch_gpu)); gpuErrchk(cudaFree(read_sizes_gpu)); gpuErrchk(cudaFree(read_offsets_gpu));
		gpuErrchk(cudaFree(seed_intervals_pos_gpu));

		gpuErrchk(cudaFree(seed_intervals_pos_compact_gpu));
		gpuErrchk(cudaFree(packed_read_batch_fow));gpuErrchk(cudaFree(packed_read_batch_rev));
		gpuErrchk(cudaFree(n_smems_fow)); /* gpuErrchk(cudaFree(n_smems_rev)); */ gpuErrchk(cudaFree(n_smems_fow_rev_scan));
		gpuErrchk(cudaFree(n_smems_max_gpu));
		gpuErrchk(cudaFree(cub_scan_temp_storage));
		gpuErrchk(cudaFree(cub_scan_temp_storage2));
		gpuErrchk(cudaFree(cub_select_temp_storage));
		gpuErrchk(cudaFree(cub_sort_temp_storage));
		gpuErrchk(cudaFree(n_seeds_fow_rev_scan));
		if(!is_smem) gpuErrchk(cudaFree(seed_ref_pos_fow_rev_gpu));

		gpuErrchk(cudaFree(n_seeds_fow_rev));

		gpuErrchk(cudaFree(n_sum_intervals_fow_gpu)); 		

		gpuErrchk(cudaFree(n_smems));
		gpuErrchk(cudaFree(final_seed_scores_gpu));
		//gpuErrchk(cudaFree(n_seeds_is_smem_flag_fow_rev));

	}
	from_gpu_results->file_bytes_skip = file_bytes;
	//fprintf(stderr,"[GPUSeed] File bytes %llu, ref: %llu\n",from_gpu_results->file_bytes_skip, file_bytes);
	
	from_gpu_results->n_ref_pos_fow_rev_prefix_sums = (uint32_t*)malloc(reads_processed*sizeof(uint32_t));
	gpuErrchk(cudaMalloc(&(n_ref_pos_fow_rev_prefix_sums_gpu_results), reads_processed*sizeof(uint32_t)));
			
	CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage_end, cub_scan_storage_bytes_end, n_ref_pos_fow_rev_prefix_sums_gpu, n_ref_pos_fow_rev_prefix_sums_gpu_results, reads_processed));
	
	gpuErrchk(cudaMalloc(&cub_scan_temp_storage_end,cub_scan_storage_bytes_end));

	CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage_end, cub_scan_storage_bytes_end, n_ref_pos_fow_rev_prefix_sums_gpu, n_ref_pos_fow_rev_prefix_sums_gpu_results, reads_processed));

	gpuErrchk(cudaMemcpy(from_gpu_results->n_ref_pos_fow_rev_prefix_sums, n_ref_pos_fow_rev_prefix_sums_gpu_results, reads_processed*sizeof(uint32_t), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(cub_scan_temp_storage_end));
	//mem_print_gpuseed(from_gpu_results, reads_processed);
	//mem_rev_print_gpuseed(from_gpu_results, reads_processed);
	cudaFreeHost(read_batch); cudaFreeHost(read_sizes); cudaFreeHost(read_offsets);
	double mem_time3 = realtime_gpu();
	gpuErrchk(cudaFree(n_ref_pos_fow_rev_prefix_sums_gpu));
	gpuErrchk(cudaFree(n_ref_pos_fow_rev_prefix_sums_gpu_results));
	//gpuErrchk(cudaFree(pre_calc_seed_intervals));

	for (int i = 0; i < nStreams; ++i){
    	result = cudaStreamDestroy(stream[i]);
	}	

	if (print_stats) {
		if (is_smem) {
			fprintf(stderr, "SMEM seed computing time stats\n");
		}
		if(is_smem)fprintf(stderr, "\n==================================SMEM seeds computing time stats=======================================================\n");
		else fprintf(stderr, "\n==================================MEM seeds computing time stats=======================================================\n");
		fprintf(stderr, "Total GPU time is %.3f seconds\n", total_gpu_time);
		fprintf(stderr, "\tTotal time spent in preparing the batch of reads for GPU processing is %.3f seconds\n", total_batch_prep_time);
		fprintf(stderr, "\tTotal time spent in finding seed intervals is %.3f seconds\n", total_find_seed_intervals_time);
		fprintf(stderr, "\tTotal time spent in filtering seed intervals is %.3f seconds\n", total_filter_seed_intervals_time);
		fprintf(stderr, "\tTotal time spent in locating seeds is %.3f seconds\n", total_locate_seeds_time);
		fprintf(stderr, "\tTotal time spent in cudaMemcpy is %.3f seconds \n", total_mem_time);
		fprintf(stderr, "Total time spent in loading reads from file is %.3f seconds\n", total_batch_load_time);
		fprintf(stderr, "Total time spent in printing the results is %.3f seconds\n", total_print_time);
		fprintf(stderr, "Total time: %.3f seconds\n", realtime_gpu() - total_time);
	}
	//fprintf(stderr,"I return from GPUSeed\n");
	return from_gpu_results;
}

#ifdef __cplusplus
}
#endif