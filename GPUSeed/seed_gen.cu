#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "./cub/cub/cub.cuh"
#include "seed_gen.h"



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

//#define OCC_INTERVAL 0x20


__constant__ bwtint_t_gpu L2_gpu[5];
__constant__ uint32_t ascii_to_dna_table[8];

/*typedef struct {
	bwtint_t_gpu primary; // S^{-1}(0), or the primary index of BWT
	bwtint_t_gpu *L2;
	bwtint_t_gpu seq_len; // sequence length
	bwtint_t_gpu bwt_size; // size of bwt, about seq_len/4
	uint32_t *bwt; // BWT
	int sa_intv;
	bwtint_t_gpu n_sa;
	bwtint_t_gpu *sa;
} bwt_t_gpu;*/

/* retrieve a character from the $-removed BWT string. Note that
 * bwt_t_gpu::bwt is not exactly the BWT string and therefore this macro is
 * called bwt_B0 instead of bwt_B */

#define bwt_bwt1(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4) + 4])

#define bwt_bwt(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4) + 4 + (k)%OCC_INTERVAL/16])

#define bwt_B0(b, k) (bwt_bwt(b, k)>>((~(k)&0xf)<<1)&3)

#define bwt_occ_intv(b, k) ((b).bwt + (k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4))


//#define bwt_bwt1(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>3) + 4) + 4])
//
//#define bwt_bwt(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 4) + 4 + (k)%OCC_INTERVAL/16])
//
//#define bwt_B0(b, k) (bwt_bwt(b, k)>>((~(k)&0xf)<<1)&3)
//
//#define bwt_occ_intv(b, k) ((b).bwt + (k)/OCC_INTERVAL*((OCC_INTERVAL>>3) + 4))





//__device__ inline bwtint_t_gpu bwt_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, uint8_t c)
//{
//	bwtint_t_gpu occ;
//
//	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];
//	if (k == (bwtint_t_gpu)(-1)) return 0;
//	if (k >= bwt.primary) --k; // because $ is not in bwt/__device__ inline uint2 find_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c)
//{
//	bwtint_t_gpu occ_l = 0, occ_u = 0; //occ_l_1 = 0, occ_l_2 = 0, occ_l_3 = 0, occ_u_1 = 0;
//
//	//if ((l+1) > u || c > 3)  return make_uint2(l+1, u);
//	if (l == bwt.seq_len) return make_uint2 (L2_gpu[c+1] - L2_gpu[c], bwt_occ_gpu(bwt, u, c));
//	if (u == bwt.seq_len) return make_uint2 (bwt_occ_gpu(bwt, l, c), L2_gpu[c+1] - L2_gpu[c]);
//	if (l == (bwtint_t_gpu)(-1)) return make_uint2 (0, bwt_occ_gpu(bwt, u, c));
//	if (u == (bwtint_t_gpu)(-1)) return make_uint2 (bwt_occ_gpu(bwt , l, c), 0);
//	if (l >= bwt.primary) --l;
//	if (u >= bwt.primary) --u;
//
//	bwtint_t_gpu kl = l / OCC_INTERVAL;
//	bwtint_t_gpu ku = u /OCC_INTERVAL;
//
//	uint4 bwt_str_l = ((uint4*)(&bwt_bwt1(bwt,l)))[0];
//	uint4 bwt_str_u = (kl == ku) ? bwt_str_l : ((uint4*)(&bwt_bwt1(bwt,u)))[0];
//
//	uint32_t l_words = (l&(OCC_INTERVAL-1)) >> 4;
//	uint32_t u_words = (u&(OCC_INTERVAL-1)) >> 4;
//
//	occ_l = bwt_occ_intv(bwt, l)[c];
//	if (l_words > 0) occ_l += pop_count_full( bwt_str_l.x, c );
//	if (l_words > 1) occ_l += pop_count_full( bwt_str_l.y, c );
//	if (l_words > 2) occ_l += pop_count_full( bwt_str_l.z, c );
//
//	occ_u = (kl == ku) ? occ_u + occ_l : bwt_occ_intv(bwt, u)[c];
//	uint32_t startm = (kl == ku) ? l_words : 0;
//
//	// sum up all the pop-counts of the relevant masks
//	if (u_words > 0 && startm == 0) occ_u += pop_count_full( bwt_str_u.x, c );
//	if (u_words > 1 && startm <= 1) occ_u += pop_count_full( bwt_str_u.y, c );
//	if (u_words > 2 && startm <= 2) occ_u += pop_count_full( bwt_str_u.z, c );
//
//	occ_l += pop_count_partial( l_words <= 1 ? (l_words == 0 ? bwt_str_l.x : bwt_str_l.y) : (l_words == 2 ? bwt_str_l.z : bwt_str_l.w), c,  (~l) & 15);
//	occ_u += pop_count_partial( u_words <= 1 ? (u_words == 0 ? bwt_str_u.x : bwt_str_u.y) : (u_words == 2 ? bwt_str_u.z : bwt_str_u.w), c,  (~u) & 15);
//
//
//	return make_uint2(occ_l, occ_u);
//}
//
//	// retrieve Occ at k/OCC_INTERVAL
//
//	occ = bwt_occ_intv(bwt, k/OCC_INTERVAL)[c];
//	//p += 4; // jump to the start of the first BWT cell
//
////	// calculate Occ up to the last k/32
////	j = k >> 5 << 5;
////	for (l = k/OCC_INTERVAL*OCC_INTERVAL; l < j; l += 32, p += 2)
////		n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
////
////	// calculate Occ
////	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
//	int i = k/OCC_INTERVAL;
//	for (i = i*OCC_INTERVAL; i <= k; ++i)
//			if (bwt_B0(bwt, i) == c) ++occ;
//	//if (c == 0) n -= ~k&31; // corrected for the masked bits
//
//	return occ;
//}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   bool abort = false;
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ inline uint pop_count_partial(uint32_t word, uint8_t c, uint32_t mask_bits) {

	 word =  word & ~((1<<(mask_bits<<1)) - 1);
	 uint odd  = ((c&2)? word : ~word) >> 1;
	 uint even = ((c&1)? word : ~word);
	 uint mask = odd & even & 0x55555555;
	 return (c == 0) ? __popc(mask) - mask_bits : __popc(mask);

}

__device__ inline uint pop_count_full(uint32_t word, uint8_t c) {

	 uint odd  = ((c&2)? word : ~word) >> 1;
	 uint even = ((c&1)? word : ~word);
	 uint mask = odd & even & 0x55555555;
	 return __popc(mask);
}

//__device__ inline bwtint_t_gpu bwt_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, uint8_t c)
//{
//	bwtint_t_gpu n, l;
//
//	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];
//	if (k == (bwtint_t_gpu)(-1)) return 0;
//	if (k >= bwt.primary) --k; // because $ is not in bwt
//
//	// retrieve Occ at k/OCC_INTERVAL
//	n = bwt_occ_intv(bwt, k)[c];
//	//p += 4; // jump to the start of the first BWT cell
//
////	// calculate Occ up to the last k/32
////	j = k >> 5 << 5;
////	for (l = k/OCC_INTERVAL*OCC_INTERVAL; l < j; l += 32, p += 2)
////		n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
////
////	// calculate Occ
////	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
//	uint2 bwt_str_k = ((uint2*)(&bwt_bwt1(bwt,k)))[0];
//	//uint32_t k_words = (k - (k/OCC_INTERVAL)*OCC_INTERVAL);
//	uint32_t k_words = (k - (k/OCC_INTERVAL)*OCC_INTERVAL) >> 4;
//	n += (k_words > 0) ? pop_count_full(bwt_str_k.x, c) + pop_count_partial(bwt_str_k.y, c, (~k) & 15) : pop_count_partial(bwt_str_k.x, c, (~k) & 15);
////	if (k_words > 15) {
////			n += pop_count_full(bwt_str_k.x, c) + pop_count_partial(bwt_str_k.y, c, (~k) & 15);
////	}
////	else n += pop_count_partial(bwt_str_k.x, c, (~k) & 15);
////	l = k/OCC_INTERVAL;
////	for (l = l*OCC_INTERVAL; l <= k; ++l)
////			if (bwt_B0(bwt, l) == c) ++n;
//	//if (c == 0) n -= ~k&31; // corrected for the masked bits
//
//	return n;
//}

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

	n = bwt_occ_intv(bwt, k)[c];
	if (k_words > 0) n += pop_count_full( bwt_str_k.x, c );
	if (k_words > 1) n += pop_count_full( bwt_str_k.y, c );
	if (k_words > 2) n += pop_count_full( bwt_str_k.z, c );

	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 15);

	return n;
}

//__device__ inline bwtint_t_gpu bwt_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k, uint8_t c)
//{
//	bwtint_t_gpu n;
//
//	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];
//	if (k == (bwtint_t_gpu)(-1)) return 0;
//	if (k >= bwt.primary) --k; // because $ is not in bwt
//
//	n = bwt_occ_intv(bwt, k)[c];
//	bwtint_t_gpu k_pos = k&(OCC_INTERVAL-1);
//	uint k_reg_for_pop_count;
//	uint2 bwt_str_k;
//	uint inter_count_k;
//	bwt_str_k = ((uint2*)((&bwt_bwt1(bwt,k))))[(k_pos>>4)];
//	inter_count_k = (bwt_str_k.x >> (24 - (c<<3))) & 255;
//	k_reg_for_pop_count = bwt_str_k.y;
//	n += inter_count_k + pop_count_partial(k_reg_for_pop_count, c,  (~k) & 15);
//	//printf("intercount=%u\n",(k)/OCC_INTERVAL*((OCC_INTERVAL>>3) + 4 - 1) + 5);
//	return n;
//}


//__device__ inline uint2 find_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c)
//{
//	bwtint_t_gpu occ_l = 0, occ_u = 0; //occ_l_1 = 0, occ_l_2 = 0, occ_l_3 = 0, occ_u_1 = 0;
//
//	//if ((l+1) > u || c > 3)  return make_uint2(l+1, u);
//	if (l == bwt.seq_len) return make_uint2 (L2_gpu[c+1] - L2_gpu[c], bwt_occ_gpu(bwt, u, c));
//	if (u == bwt.seq_len) return make_uint2 (bwt_occ_gpu(bwt, l, c), L2_gpu[c+1] - L2_gpu[c]);
//	if (l == (bwtint_t_gpu)(-1)) return make_uint2 (0, bwt_occ_gpu(bwt, u, c));
//	if (u == (bwtint_t_gpu)(-1)) return make_uint2 (bwt_occ_gpu(bwt , l, c), 0);
//	if (l >= bwt.primary) --l;
//	if (u >= bwt.primary) --u;
//
//	bwtint_t_gpu kl = l / OCC_INTERVAL;
//	bwtint_t_gpu ku = u /OCC_INTERVAL;
//	// retrieve Occ at k/OCC_INTERVAL
//	//occ_l_1 = bwt_occ_intv(bwt, l)[c];
//	occ_l = bwt_occ_intv(bwt, l)[c];
//	//occ_u = (kl == ku) ? occ_l : bwt_occ_intv(bwt, u)[c];
//
//	//p += 4; // jump to the start of the first BWT cell
//
////	// calculate Occ up to the last k/32
////	j = k >> 5 << 5;
////	for (l = k/OCC_INTERVAL*OCC_INTERVAL; l < j; l += 32, p += 2)
////		n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
////
////	// calculate Occ
////	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
//	uint2 bwt_str_l = ((uint2*)(&bwt_bwt1(bwt,l)))[0];
//	uint2 bwt_str_u = (kl == ku) ? bwt_str_l : ((uint2*)(&bwt_bwt1(bwt,u)))[0];
////	uint bwt_str_l.x = (&bwt_bwt1(bwt,l))[0];
////	uint bwt_str_l_y = (&bwt_bwt1(bwt,l))[1];
////	uint bwt_str_u.x = (&bwt_bwt1(bwt,u))[0];
////	uint bwt_str_u_y = (&bwt_bwt1(bwt,u))[1];
//	//uint bwt_str_l = bwt_bwt(bwt,l);
//	//uint bwt_str_u = bwt_bwt(bwt,u);
////	uint32_t l_words = (l - kl*OCC_INTERVAL);
////	uint32_t u_words = (u - ku*OCC_INTERVAL);
//	uint32_t l_words = (l - kl*OCC_INTERVAL) >> 4;
//	uint32_t u_words = (u - ku*OCC_INTERVAL) >> 4;
//	occ_l = bwt_occ_intv(bwt, l)[c];
//	if (l_words > 0) occ_l += pop_count_full( bwt_str_l.x, c );
//
//	occ_u = (kl == ku) ? occ_u + occ_l : bwt_occ_intv(bwt, u)[c];
//	uint32_t startm = (kl == ku) ? l_words : 0;
//
//	// sum up all the pop-counts of the relevant masks
//	if (u_words > 0 && startm == 0) occ_u += pop_count_full( bwt_str_u.x, c );
//
//
//	occ_l += pop_count_partial( l_words > 0 ? bwt_str_l.y : bwt_str_l.x, c,  (~l) & 15);
//	occ_u += pop_count_partial( u_words > 0 ? bwt_str_u.y : bwt_str_u.x, c,  (~u) & 15);
////	if (l_words > 15) {
////		occ_l_2 = pop_count_full(bwt_str_l.x, c);
////		occ_l = occ_l_1 + occ_l_2 + pop_count_partial(bwt_str_l.y, c, (~l) & 15);
////	}
////	else occ_l = occ_l_1 + pop_count_partial(bwt_str_l.x, c, (~l) & 15);
////	if (kl == ku) {
////		occ_u_1 = occ_l_1;
////		if (l_words > 15) occ_u = occ_l_1 + occ_l_2 + pop_count_partial(bwt_str_u.y, c, (~u) & 15);
////		else if (u_words > 15) occ_u = occ_u_1 + pop_count_full(bwt_str_u.x, c) + pop_count_partial(bwt_str_u.y, c, (~u) & 15);
////		else occ_u = occ_l_1 + pop_count_partial(bwt_str_u.x, c, (~u) & 15);
////	}
////	else {
////		occ_u = bwt_occ_intv(bwt, u)[c];
////		if (u_words > 15) occ_u += pop_count_full(bwt_str_u.x, c) + pop_count_partial(bwt_str_u.y, c, (~u) & 15);
////		else occ_u += pop_count_partial(bwt_str_u.x, c, (~u) & 15);
////	}
//
//
//
////	bwtint_t_gpu i = kl;
////	for (i = kl * OCC_INTERVAL; i <= l; ++i)
////			if (bwt_B0(bwt, i) == c) ++occ_l;
////	if(kl == ku) {
////		occ_u = occ_l;
////		for (;i <= u; ++i)
////			if (bwt_B0(bwt, i) == c) ++occ_u;
////	} else {
////		occ_u = bwt_occ_intv(bwt, u)[c];
////		i = ku;
////		for (i = ku * OCC_INTERVAL; i <= u; ++i)
////			if (bwt_B0(bwt, i) == c) ++occ_u;
////	}
//	//if (c == 0) n -= ~k&31; // corrected for the masked bits
//
//	return make_uint2(occ_l, occ_u);
//}

__device__ inline uint2 find_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c)
{
	bwtint_t_gpu occ_l = 0, occ_u = 0; //occ_l_1 = 0, occ_l_2 = 0, occ_l_3 = 0, occ_u_1 = 0;

	if (c > 3)  return make_uint2(l - L2_gpu[c], u - L2_gpu[c]);
	if (l == bwt.seq_len) return make_uint2 (L2_gpu[c+1] - L2_gpu[c], bwt_occ_gpu(bwt, u, c));
	if (u == bwt.seq_len) return make_uint2 (bwt_occ_gpu(bwt, l, c), L2_gpu[c+1] - L2_gpu[c]);
	if (l == (bwtint_t_gpu)(-1)) return make_uint2 (0, bwt_occ_gpu(bwt, u, c));
	if (u == (bwtint_t_gpu)(-1)) return make_uint2 (bwt_occ_gpu(bwt , l, c), 0);
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


	return make_uint2(occ_l, occ_u);
}
//__device__ inline uint2 find_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c)
//{
//	bwtint_t_gpu occ_l = 0, occ_u = 0; //occ_l_1 = 0, occ_l_2 = 0, occ_l_3 = 0, occ_u_1 = 0;
//
//	if (l == bwt.seq_len) return make_uint2 (L2_gpu[c+1] - L2_gpu[c], bwt_occ_gpu(bwt, u, c));
//	if (u == bwt.seq_len) return make_uint2 (bwt_occ_gpu(bwt, l, c), L2_gpu[c+1] - L2_gpu[c]);
//	if (l == (bwtint_t_gpu)(-1)) return make_uint2 (0, bwt_occ_gpu(bwt, u, c));
//	if (u == (bwtint_t_gpu)(-1)) return make_uint2 (bwt_occ_gpu(bwt , l, c), 0);
//	if (l >= bwt.primary) --l;
//	if (u >= bwt.primary) --u;
//
//	bwtint_t_gpu kl = l / OCC_INTERVAL;
//	bwtint_t_gpu ku = u /OCC_INTERVAL;
//	occ_l = bwt_occ_intv(bwt, l)[c];
//	bwtint_t_gpu l_pos = l&(OCC_INTERVAL-1);
//	bwtint_t_gpu u_pos = u&(OCC_INTERVAL-1);
//	bwtint_t_gpu l_pos_reg = (l_pos>>4);
//	bwtint_t_gpu u_pos_reg = (u_pos>>4);
//	uint inter_count_mask = 24 - (c<<3);
//	uint l_reg_for_pop_count;
//	uint u_reg_for_pop_count;
//	uint *bwt_bin_l =  &bwt_bwt1(bwt,l);
//	uint *bwt_bin_u = &bwt_bwt1(bwt,u);
//	uint2 *bwt_str_u = ((uint2*)((bwt_bin_u)));
//	uint inter_count_l;
//	uint inter_count_u;
//	uint2 bwt_str_l = ((uint2*)((bwt_bin_l)))[l_pos_reg];
//	inter_count_l= (bwt_str_l.x >> inter_count_mask) & 255;
//	l_reg_for_pop_count = bwt_str_l.y;
//
//	/*if (l_pos < 16) {hg19.fasta.sa
//		l_reg_for_pop_count = bwt_bwt1(bwt,l);
//		inter_count_l = 0;
//	}
//	else {
//		bwt_str_l = ((uint2*)((&bwt_bwt1(bwt,l))+1))[(l_pos>>4) - 1];
//		inter_count_l= (bwt_str_l.x >> (24 - (c<<3))) & 255;
//		l_reg_for_pop_count = bwt_str_l.y;
//	}*/
//	//occ_u = (kl==ku) ? occ_l : bwt_occ_intv(bwt, u)[c];
//	if (kl == ku) {
//		occ_u = occ_l;
//		if (l_pos_reg == u_pos_reg) {
//			inter_count_u = inter_count_l;
//			u_reg_for_pop_count = l_reg_for_pop_count;
//		} /*else if ((u_pos>>4) - (l_pos>>4) == 1) {
//			inter_count_u = inter_count_l + pop_count_full(l_reg_for_pop_count, c);
//			u_reg_for_pop_count = ((bwt_bin_u))[(u_pos_reg<<1)];
//		}*/
//		else {
//			//bwt_str_u = ((uint2*)((&bwt_bwt1(bwt,u))))[(u_pos>>4)];
//			inter_count_u= (bwt_str_u[u_pos_reg].x >> inter_count_mask) & 255;
//			u_reg_for_pop_count = bwt_str_u[u_pos_reg].y;
//		}
//	}
//	else  {
//		occ_u = bwt_occ_intv(bwt, u)[c];
//		//bwt_str_u = ((uint2*)((&bwt_bwt1(bwt,u))))[(u_pos>>4)];
//		inter_count_u= (bwt_str_u[u_pos_reg].x >> inter_count_mask) & 255;
//		u_reg_for_pop_count = bwt_str_u[u_pos_reg].y;
//		//		if (u_pos < 16) {
//		//			u_reg_for_pop_count = bwt_bwt1(bwt,u);
//		//			inter_count_u = 0;
//		//		}
//		//	else {
//		//		bwt_str_u = ((uint2*)((&bwt_bwt1(bwt,u))))[(u_pos>>4)];
//		//		inter_count_u= (bwt_str_u.x >> (24 - (c<<3))) & 255;
//		//		u_reg_for_pop_count = bwt_str_u.y;
//	}
//
//
//	//}
//	occ_l += inter_count_l + pop_count_partial(l_reg_for_pop_count, c,  (~l) & 15);
//	occ_u += inter_count_u + pop_count_partial(u_reg_for_pop_count, c,  (~u) & 15);
//
//
//	return make_uint2(occ_l, occ_u);
//}

//__device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k) {
//	bwtint_t_gpu x = k - (k > bwt.primary);
//	x = bwt_B0(bwt, x);
//	x = L2_gpu[x] + bwt_occ_gpu(bwt, k, x);
//	return k == bwt.primary ? 0 : x;
//}

__device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k) {
	if (k > bwt.primary) --k; // because $ is not in bwt
	if (k == bwt.primary) return 0;



	uint4 bwt_str_k = ((uint4*)(&bwt_bwt1(bwt,k)))[0];
	uint32_t k_words = (k&(OCC_INTERVAL-1)) >> 4;

	uint8_t c;
	if (k_words == 0) c = (bwt_str_k.x>>((~(k)&0xf)<<1))&3;
	if (k_words == 1) c = (bwt_str_k.y>>((~(k)&0xf)<<1))&3;
	if (k_words == 2) c = (bwt_str_k.z>>((~(k)&0xf)<<1))&3;
	if (k_words == 3) c = (bwt_str_k.w>>((~(k)&0xf)<<1))&3;

	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];



	bwtint_t_gpu n = bwt_occ_intv(bwt, k)[c];

	n = bwt_occ_intv(bwt, k)[c];
	if (k_words > 0) n += pop_count_full( bwt_str_k.x, c );
	if (k_words > 1) n += pop_count_full( bwt_str_k.y, c );
	if (k_words > 2) n += pop_count_full( bwt_str_k.z, c );

	n += pop_count_partial( k_words <= 1 ? (k_words == 0 ? bwt_str_k.x : bwt_str_k.y) : (k_words == 2 ? bwt_str_k.z : bwt_str_k.w), c,  (~k) & 15);

	return L2_gpu[c] + n;
}

//#define MAX_READ_LENGTH 35500

//#define MAX_SEEDS_PER_READ MAX_READ_LENGTH

#define THREADS_PER_SMEM 1


//__device__ int seed_intervals_gpu(uint8_t *read, int read_len, bwtint_t_gpu *seed_interval_l_gpu, bwtint_t_gpu *seed_interval_u_gpu,
//		int *seed_read_begin_gpu, int *seed_read_end_gpu, int min_seed_size, bwt_t_gpu bwt) {
//
//	uint8_t ch;
//	/*For each position of the sequence, we search the maximal exact match*/
//	int curr_end;
//	int n_seeds = 0;
//	for (curr_end = read_len - 1; curr_end >= min_seed_size - 1; curr_end--) {
//		//for each starting position in the query
//		int i;
//		bwtint_t_gpu l, u;
//		bwtint_t_gpu prev_l = 0, prev_u = bwt.seq_len;
//		int curr_begin = curr_end + 1;
//		for (i = curr_end ; i >= 0 ; i--) {
//			/*get the base*/
//			ch = read[i];
//
//			/*unknown bases*/
//			if (ch == 4) {
//				break;
//			}
//
//			//calculate the range
//			l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//			u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//			if (l > u) {
//				break;
//			}
//			prev_l = l;
//			prev_u = u;
//			curr_begin--;
//		}
//		if ((curr_end - curr_begin + 1) >= min_seed_size) {
//			seed_interval_l_gpu[n_seeds] = prev_l;
//			seed_interval_u_gpu[n_seeds] = prev_u;
//			seed_read_begin_gpu[n_seeds] = curr_begin;bwt_bwt1
//			seed_read_end_gpu[n_seeds] = curr_end;
//			n_seeds++;
//		}
//		/*If an exact match is found at the end of the query, no need to check any more*/
//	}
//	return n_seeds;
//
//}
//
//__global__ void calc_smems(uint8_t *read_batch_gpu, int *read_sizes_gpu, int *read_offsets_gpu, bwtint_t_gpu *seed_interval_l_gpu, bwtint_t_gpu *seed_interval_u_gpu,
//		int *seed_read_begin_gpu, int *seed_read_end_gpu, int *num_seeds, int min_seed_size, bwt_t_gpu bwt, int actual_read_batch_size, int (MAX_READ_LENGTH - min_seed_size)) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= actual_read_batch_size) return;$

//	int intv_idx = tid * (MAX_READ_LENGTH - min_seed_size);
//	int n_seeds = 0;
//	int read_len = read_sizes_gpu[tid];
//	int read_off = read_offsets_gpu[tid];
//	uint8_t *read = &(read_batch_gpu[read_off]);
//	n_seeds = seed_intervals_gpu(read, read_len, &(seed_interval_l_gpu[intv_idx]), &(seed_interval_u_gpu[intv_idx]), &(seed_read_begin_gpu[intv_idx]), &(seed_read_end_gpu[intv_idx]), min_seed_size, bwt);
//	uint8_t *rev_comp_read = &(read_batch_gpu[read_off + read_len]);
//	n_seeds += seed_intervals_gpu(rev_comp_read, read_len, &(seed_interval_l_gpu[intv_idx]), &(seed_interval_u_gpu[intv_idx]), &(seed_read_begin_gpu[intv_idx]), &(seed_read_end_gpu[intv_idx]), min_seed_size, bwt);
//	num_seeds[tid] = n_seeds;
//}

__global__ void seeds_to_threads(int2 *final_seed_read_pos_fow_rev, uint32_t *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t n_smems_sum_fow_rev) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= (n_smems_sum_fow_rev)*THREADS_PER_SMEM) return;

        int n_seeds = n_seeds_fow_rev_scan[tid+1] - n_seeds_fow_rev_scan[tid];//n_seeds_fow[tid];
        int2 seed_read_pos = seed_read_pos_fow_rev[tid];
        uint32_t intv_l = seed_intervals_fow_rev[tid].x;
        uint32_t offset = n_seeds_fow_rev_scan[tid];
        int idx = tid%THREADS_PER_SMEM;
        int i;
        for(i = 0; (i + idx) < n_seeds; i+=THREADS_PER_SMEM) {
        	seed_sa_idx_fow_rev_gpu[offset + i + idx] = intv_l + i + idx;
        	final_seed_read_pos_fow_rev[offset + i + idx] = seed_read_pos;
        }

        return;

}

//__global__ void seeds_to_threads_mem(int2 *final_seed_read_pos_fow_rev, uint32_t *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t n_smems_sum_fow_rev) {
//
//        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//        if (tid >= (n_smems_sum_fow_rev)*THREADS_PER_SMEM) return;
//
//        int n_seeds = n_seeds_fow_rev_scan[tid+1] - n_seeds_fow_rev_scan[tid];//n_seeds_fow[tid];
//        int2 seed_read_pos = seed_read_pos_fow_rev[tid];
//        uint32_t is_rev_strand = (seed_read_pos.y) >> 31;
//        uint32_t intv_l = seed_intervals_fow_rev[tid].x;
//        uint32_t offset = n_seeds_fow_rev_scan[tid];
//        uint2 next_seed_interval = make_uint2(1,0);
//        if (is_rev_strand){
//        	int p = 1;
//        	while (seed_read_pos.y == seed_read_pos_fow_rev[tid-p].y && tid-p >= 0) {
//        		next_seed_interval = seed_intervals_fow_rev[tid-p];
//        		if (next_seed_interval.y - next_seed_interval.x + 1 > 0) break;
//        		p++;
//        	}
//        }
//        else {
//        	int p = 1;
//        	while (seed_read_pos.x == seed_read_pos_fow_rev[tid+p].x && tid + p < n_smems_sum_fow_rev){
//        		next_seed_interval = seed_intervals_fow_rev[tid+p];
//        		if (next_seed_interval.y - next_seed_interval.x + 1 > 0) break;
//        		p++;
//
//        	}
//        	//next_seed_interval = seed_intervals_fow_rev[tid+1];
//        }
//
//        int i = 0;
//        //int seed_count = 0;
//        for(i = 0; i < n_seeds; i++) {
//        //for(i = 0; seed_count < n_seeds ; i++) {
//        	if (((intv_l + i) >= next_seed_interval.x) && ((intv_l + i) <= next_seed_interval.y)){
//        		seed_sa_idx_fow_rev_gpu[offset + i] = UINT_MAX;
//        		final_seed_read_pos_fow_rev[offset + i] = make_int2(UINT_MAX, UINT_MAX);
//        		//seed_count++;
//        	}
//        	else {
//        		seed_sa_idx_fow_rev_gpu[offset + i] = intv_l + i;
//        		final_seed_read_pos_fow_rev[offset + i] = seed_read_pos;
//        	}
//        }
//
//        return;
//
//}

__global__ void seeds_to_threads_mem(int2 *final_seed_read_pos_fow_rev, uint32_t *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t n_smems_sum_fow_rev) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= (n_smems_sum_fow_rev)*THREADS_PER_SMEM) return;

        int n_seeds = n_seeds_fow_rev_scan[tid+1] - n_seeds_fow_rev_scan[tid];//n_seeds_fow[tid];
        int2 seed_read_pos = seed_read_pos_fow_rev[tid];
        uint32_t is_rev_strand = (seed_read_pos.y) >> 31;
        uint32_t intv_l = seed_intervals_fow_rev[tid].x;
        uint32_t offset = n_seeds_fow_rev_scan[tid];
        uint2 next_seed_interval = make_uint2(1,0);
        if (is_rev_strand){
        	int p = 1;
        	while (seed_read_pos.y == seed_read_pos_fow_rev[tid-p].y && tid-p >= 0) {
        		next_seed_interval = seed_intervals_fow_rev[tid-p];
        		if (next_seed_interval.y - next_seed_interval.x + 1 > 0) break;
        		p++;
        	}
        }
        else {
        	int p = 1;
        	while (seed_read_pos.x == seed_read_pos_fow_rev[tid+p].x && tid + p < n_smems_sum_fow_rev){
        		next_seed_interval = seed_intervals_fow_rev[tid+p];
        		if (next_seed_interval.y - next_seed_interval.x + 1 > 0) break;
        		p++;

        	}
        	//next_seed_interval = seed_intervals_fow_rev[tid+1];
        }

        int i = 0;
        int seed_count = 0;
        for(i = 0; seed_count < n_seeds; i++) {
        //for(i = 0; seed_count < n_seeds ; i++) {
        	if (((intv_l + i) < next_seed_interval.x) || ((intv_l + i) > next_seed_interval.y)){
        		seed_sa_idx_fow_rev_gpu[offset + seed_count] = intv_l + i;
        		final_seed_read_pos_fow_rev[offset + seed_count] = seed_read_pos;
        		seed_count++;
        	}
        }

        return;

}
__global__ void locate_seeds_gpu(uint32_t *seed_ref_pos_fow_rev_gpu, bwt_t_gpu bwt, uint32_t n_seeds_sum_fow_rev) {

        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
        if (tid >= n_seeds_sum_fow_rev) return;

        	uint32_t sa_idx = seed_ref_pos_fow_rev_gpu[tid];
			//printf("SA: %lu\n",sa_idx);
        	if (sa_idx == UINT_MAX) return;
        	int itr = 0;
        	while(sa_idx % bwt.sa_intv){
        		itr++;
        		sa_idx = bwt_inv_psi_gpu(bwt, sa_idx);
        	}
			//printf("SA after: %lu\n",sa_idx);
        	seed_ref_pos_fow_rev_gpu[tid] = bwt.sa[sa_idx/bwt.sa_intv] + itr;
			//printf("Seed ref pos: %lu\n",seed_ref_pos_fow_rev_gpu[tid]);
        return;

}


__global__ void locate_seeds_gpu_wrapper(int2 *final_seed_read_pos_fow_rev, uint32_t *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev,uint32_t *n_smems_sum_fow_rev, uint32_t *n_seeds_sum_fow_rev, bwt_t_gpu bwt) {


	int BLOCKDIM =128;
	int N_BLOCKS = (n_smems_sum_fow_rev[0]*THREADS_PER_SMEM  + BLOCKDIM - 1)/BLOCKDIM;

	n_seeds_fow_rev_scan[n_smems_sum_fow_rev[0]] = n_seeds_sum_fow_rev[0];

	seeds_to_threads<<<N_BLOCKS, BLOCKDIM>>>(final_seed_read_pos_fow_rev, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev, seed_read_pos_fow_rev, n_smems_sum_fow_rev[0]);

	uint32_t *seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;


	N_BLOCKS = (n_seeds_sum_fow_rev[0]  + BLOCKDIM - 1)/BLOCKDIM;

	//printf("seed_ref_pos_fow_rev_gpu:%lu\n", seed_ref_pos_fow_rev_gpu[1]);

	locate_seeds_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_ref_pos_fow_rev_gpu, bwt, n_seeds_sum_fow_rev[0]);
}


__global__ void locate_seeds_gpu_wrapper_mem(int2 *final_seed_read_pos_fow_rev, uint32_t *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev,uint32_t *n_smems_sum_fow_rev, uint32_t *n_seeds_sum_fow_rev, uint32_t *n_ref_pos_fow_rev, bwt_t_gpu bwt) {


	int BLOCKDIM =128;
	int N_BLOCKS = (n_smems_sum_fow_rev[0]*THREADS_PER_SMEM  + BLOCKDIM - 1)/BLOCKDIM;

	n_seeds_fow_rev_scan[n_smems_sum_fow_rev[0]] = n_seeds_sum_fow_rev[0];

	seeds_to_threads_mem<<<N_BLOCKS, BLOCKDIM>>>(final_seed_read_pos_fow_rev, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev, seed_read_pos_fow_rev, n_smems_sum_fow_rev[0]);

	uint32_t *seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;


	N_BLOCKS = (n_seeds_sum_fow_rev[0]  + BLOCKDIM - 1)/BLOCKDIM;

	locate_seeds_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_ref_pos_fow_rev_gpu, bwt, n_seeds_sum_fow_rev[0]);


}

//__global__ void locate_seeds_gpu(uint2 *seed_read_pos_gpu, uint2 *seed_read_pos_rev_gpu, uint2 *final_seed_read_pos_gpu, uint2 *final_seed_read_pos_rev_gpu, uint32_t *seed_ref_pos_fow_gpu, uint32_t *seed_ref_pos_rev_gpu, uint32_t *n_seeds_fow_scan, uint32_t *n_seeds_rev_scan, uint32_t *smem_intv_l_fow, uint32_t *smem_intv_l_rev, bwt_t_gpu bwt, uint32_t n_smems_sum_fow, uint32_t n_smems_sum_rev, uint32_t n_ref_pos) {
//
//        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//        if (tid >= (n_smems_sum_fow + n_smems_sum_rev)*THREADS_PER_SMEM) return;
//
//        if (tid < n_smems_sum_fow*THREADS_PER_SMEM) {
//        	int thread_idx = tid;
//        }
//        int i;
//        int intv_idx = tid * (MAX_READ_LENGTH - min_seed_size);
//        int pos_idx = tid * MAX_SEEDS_PER_READ;
//        int n_seeds = 0;
//        for (i = 0; i < num_seeds[tid]; i++) {
//        	int j;
//        	int n_locs = seed_interval_u_gpu[intv_idx + i] -  seed_interval_l_gpu[intv_idx + i] + 1;
//        	for (j = 0 ; j < n_locs; j++) {
//        		int k, itr = 0;
//        		bwtint_t_gpu u = seed_interval_u_gpu[intv_idx + i] + j;
//        		//for(k = 0; k < 100; k++) {
//        		while(u % bwt.sa_intv){
//        			itr++;
//        			u = bwt_inv_psi_gpu(bwt, u);
//        			//if(u % bwt.sa_intv == 0) break;
//        		}
//        		seed_ref_begin_gpu[pos_idx + n_seeds] = bwt.sa[u/bwt.sa_intv] + itr;
//        		n_seeds++;
//        	}
//
//        }
//        num_seeds[tid] = n_seeds;
//        return;
//
//}

//__global__ void finalize_seeds_gpu(int *read_sizes_gpu, bwtint_t_gpu *seed_interval_l_gpu, bwtint_t_gpu *seed_interval_u_gpu,
//                int *seed_read_begin_gpu, int *seed_read_end_gpu, int *num_seeds, int actual_read_batch_size, int min_seed_size) {
//
//        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//        if (tid >= actual_read_batch_size) return;
//        int n_seeds = 0;
//        int intv_idx = tid * (MAX_READ_LENGTH - min_seed_size);
//        int n_intervals = read_sizes_gpu[tid] - min_seed_size;
//        int idx_comp_read;
//        if(tid%2 == 0) {
//        	idx_comp_read = tid + 1;
//        }
//        else {
//        	idx_comp_read = tid - 1;
//        }
//        int n_intervals_comp_read =  read_sizes_gpu[idx_comp_read] - min_seed_size;
//        __syncthreads();
//        //if (tid%2 == 0) {
//        	int j;
//        	for (j = 0; j < n_intervals; j++) {
//        		int k;
//        		int pass = 1;
//        		if(seed_read_end_gpu[intv_idx + j] == -1/*MAX_READ_LENGTH*/) pass = 0;
//        		else {
//        			for (k=0; k < n_intervals_comp_read; k++) {
//        				int seed_comp_read_begin = read_sizes_gpu[idx_comp_read] - seed_read_end_gpu[(idx_comp_read*(MAX_READ_LENGTH - min_seed_size)) + k];
//        				int seed_comp_read_end = read_sizes_gpu[idx_comp_read] - seed_read_begin_gpu[(idx_comp_read*(MAX_READ_LENGTH - min_seed_size)) + k];
//        				if ((seed_read_begin_gpu[intv_idx + j] > seed_comp_read_begin && seed_read_end_gpu[intv_idx+ j] < seed_comp_read_end)
//        						|| (seed_read_begin_gpu[intv_idx + j] == seed_comp_read_begin && seed_read_end_gpu[intv_idx + j] < seed_comp_read_end)
//        						|| (seed_read_begin_gpu[intv_idx + j] > seed_comp_read_begin && seed_read_end_gpu[intv_idx + j] == seed_comp_read_end)){
//        					pass = 0;
//
//        				}
//        			}
//        		}
//        		if (pass) {
//        			seed_interval_l_gpu[intv_idx + n_seeds] = seed_interval_l_gpu[intv_idx + j];
//        			seed_interval_u_gpu[intv_idx + n_seeds] = seed_interval_u_gpu[intv_idx + j];
//        			seed_read_begin_gpu[intv_idx + n_seeds] = seed_read_begin_gpu[intv_idx + j];
//        			seed_read_end_gpu[intv_idx + n_seeds] = seed_read_end_gpu[intv_idx + j];
//        			n_seeds++;
//        		}
//
//        	}
//        	num_seeds[tid] = n_seeds;
////        } else {
////        	int j;
////        	for (j = 0; j < num_seeds[tid]; j++) {
////        		int k;
////        		int pass = 1;
////        		for (k=0; k < num_seeds_comp_read; k++) {
////        			int seed_read_begin_r_strand = read_sizes_gpu[tid+1] - seed_read_end_gpu[((tid+1)*(MAX_READ_LENGTH - min_seed_size)) + k];
////        			int seed_read_end_r_strand = read_sizes_gpu[tid+1] - seed_read_begin[((tid+1)*(MAX_READ_LENGTH - min_seed_size)) + k];
////        			if ((seed_read_begin_gpu[intv_idx + j] > seed_read_begin_r_strand && seed_read_end[intv_idx+ j] < seed_read_end_r_strand)
////        					|| (seed_read_begin[intv_idx + j] == seed_read_begin_r_strand && seed_read_end[intv_idx + j] < seed_read_end_r_strand)
////        					|| (seed_read_begin[intv_idx + j] > seed_read_begin_r_strand && seed_read_end[intv_idx + j] == seed_read_end_r_strand)){
////        				pass = 0;
////
////        			}
////        		}
////        		if (pass) {
////        			seed_interval_l_gpu[intv_idx + n_seeds] = seed_interval_l_gpu[intv_idx + j];
////        			seed_interval_u_gpu[intv_idx + n_seeds] = seed_interval_y_gpu[intv_idx + j];
////        			seed_read_begin_gpu[intv_idx + n_seeds] = seed_read_begin_gpu[intv_idx + j];
////        			seed_read_end_gpu[intv_idx + n_seeds] = seed_read_end_gpu[intv_idx + j];
////        			n_seeds++;
////
////        		}
////
////        	}
////        }
//
//        return;
//
//}

//__global__ void filter_seed_intervals_gpu(int *read_sizes_gpu, bwtint_t_gpu *seed_interval_l_gpu, bwtint_t_gpu *seed_interval_u_gpu,
//		int *seed_read_begin_gpu, int *seed_read_end_gpu, int min_seed_size, int batch_size) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= batch_size) return;
//	int n_seeds = 0;
//	int intv_idx = tid * (MAX_READ_LENGTH - min_seed_size);
//	int n_intervals = read_sizes_gpu[tid] - min_seed_size;
//
//
//	int i, j;
//	if ((seed_read_end_gpu[intv_idx] - seed_read_begin_gpu[intv_idx]) < min_seed_size)  seed_read_end_gpu[intv_idx] = -1;//MAX_READ_LENGTH;
//	for(i = 1; i < n_intervals; i++){
//		if ((seed_read_end_gpu[intv_idx + i] - seed_read_begin_gpu[intv_idx + i]) < min_seed_size)  seed_read_end_gpu[intv_idx + i] = -1;//MAX_READ_LENGTH;
//		else {
//			//for(j = i - 1; j >=0; j--) {
//				if (seed_read_begin_gpu[intv_idx + i] ==  seed_read_begin_gpu[intv_idx + (i-1)]) {
//					seed_read_end_gpu[intv_idx + i] = -1;//MAX_READ_LENGTH;
//					//break;
//				}
//			//}
//		}
//	}
//
//
//	return;
//
//}

__global__ void count_seed_intervals_gpu(uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan, uint32_t* n_ref_pos_fow_rev,  uint32_t n_smems_max, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int thread_read_num = tid/n_smems_max;
	int offset_in_read = tid - (thread_read_num*n_smems_max);
	if(offset_in_read >= n_smems_fow_rev[thread_read_num]) return;
	int intv_idx = n_smems_fow_rev_scan[thread_read_num];
	int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
	n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals;
	if (n_intervals > 0)  atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals);

	return;

}


//__global__ void count_seed_intervals_gpu_mem(uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow,  uint32_t *n_smems_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,uint32_t *n_ref_pos_fow_rev, uint32_t n_smems_max, int n_tasks) {
//
//	 int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	 if (tid >= 2*n_tasks) return;
//
//	 if (tid < n_tasks) {
//		 int thread_read_num = tid/n_smems_max;
//		 int offset_in_read = tid - (thread_read_num*n_smems_max);
//		 if(offset_in_read >= n_smems_fow[thread_read_num]) return;
//		 int intv_idx = n_smems_fow_rev_scan[thread_read_num];
//		 int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
////		 if (n_intervals > 0 && (offset_in_read < n_smems_fow[thread_read_num] - 1) && seed_read_pos_fow_rev[intv_idx + offset_in_read].x == seed_read_pos_fow_rev[intv_idx + offset_in_read + 1].x) {
////			int next_n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read + 1].y - seed_intervals_fow_rev[intv_idx + offset_in_read + 1].x + 1;
////			n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals - next_n_intervals;
////			atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals - next_n_intervals);
////		 }
////		 else  {
//			 n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals;
//			 if (n_intervals > 0) atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals);
//		 //}
//
//
//	 } else {
//		 tid = tid - n_tasks;
//		 int thread_read_num = tid/n_smems_max;
//		 int offset_in_read = tid - (thread_read_num*n_smems_max);
//		 if(offset_in_read >= n_smems_rev[thread_read_num]) return;
//		 int intv_idx = n_smems_fow_rev_scan[thread_read_num] + n_smems_fow[thread_read_num];
//		 int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
////		 if (n_intervals > 0 && offset_in_read > 0 && seed_read_pos_fow_rev[intv_idx + offset_in_read].y == seed_read_pos_fow_rev[intv_idx + offset_in_read - 1].y) {
////			 int next_n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read - 1].y - seed_intervals_fow_rev[intv_idx + offset_in_read - 1].x + 1;
////			 n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals - next_n_intervals;
////			 atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals - next_n_intervals);
////		 }
////		 else {
//			 n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals;
//			 if (n_intervals > 0) atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals);
////		 }
//	 }
//
//	return;
//
//}

__global__ void count_seed_intervals_gpu_mem(uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow,  uint32_t *n_smems_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,uint32_t *n_ref_pos_fow_rev, uint32_t n_smems_max, int n_tasks) {

	 int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	 if (tid >= 2*n_tasks) return;

	 if (tid < n_tasks) {
		 int thread_read_num = tid/n_smems_max;
		 int offset_in_read = tid - (thread_read_num*n_smems_max);
		 if(offset_in_read >= n_smems_fow[thread_read_num]) return;
		 int intv_idx = n_smems_fow_rev_scan[thread_read_num];
		 int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
		 int n_intervals_to_add = n_intervals;
		 int next_n_intervals = 0;
		 if (n_intervals > 0) {
			 int seed_read_pos_x = seed_read_pos_fow_rev[intv_idx + offset_in_read].x;
			 int p = 1;
			 while (seed_read_pos_x == seed_read_pos_fow_rev[intv_idx + offset_in_read + p].x && offset_in_read + p < n_smems_fow[thread_read_num]) {
				 next_n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read + p].y - seed_intervals_fow_rev[intv_idx + offset_in_read + p].x + 1;
				 if (next_n_intervals > 0) {
					 n_intervals_to_add = n_intervals - next_n_intervals;
					 break;
				 }

				 p++;
			 }
			 atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals_to_add);
		 }

		 n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals_to_add;



	 } else {
		 tid = tid - n_tasks;
		 int thread_read_num = tid/n_smems_max;
		 int offset_in_read = tid - (thread_read_num*n_smems_max);
		 if(offset_in_read >= n_smems_rev[thread_read_num]) return;
		 int intv_idx = n_smems_fow_rev_scan[thread_read_num] + n_smems_fow[thread_read_num];
		 int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
		 int n_intervals_to_add = n_intervals;
		 int next_n_intervals = 0;
		 if (n_intervals > 0) {
			 int seed_read_pos_y = seed_read_pos_fow_rev[intv_idx + offset_in_read].y;
			 int p = 1;
			 while (seed_read_pos_y == seed_read_pos_fow_rev[intv_idx + offset_in_read - p].y && offset_in_read - p >= 0) {
				 next_n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read - p].y - seed_intervals_fow_rev[intv_idx + offset_in_read - p].x + 1;
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
	if (tid >= 2*n_tasks) return;
	if (tid < n_tasks) {
		int thread_read_num = tid/n_smems_max;
		int offset_in_read = tid - (thread_read_num*n_smems_max);
		if(offset_in_read >= n_smems_fow[thread_read_num] || offset_in_read == 0) return;
		int intv_idx = n_smems_fow_rev_scan[thread_read_num];
		int seed_begin_pos = seed_read_pos_fow_rev[intv_idx + offset_in_read].x;
		int comp_seed_begin_pos = seed_read_pos_fow_rev[intv_idx + offset_in_read - 1].x;
		if(seed_begin_pos == comp_seed_begin_pos ) {
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1]= make_uint2 (1, 0);
			//seed_read_pos_fow[intv_idx + offset_in_read].y =  -1;
		}
	} else {
		tid = tid - n_tasks;
		int thread_read_num = tid/n_smems_max;
		int offset_in_read = tid - (thread_read_num*n_smems_max);
		if(offset_in_read >= n_smems_rev[thread_read_num] || offset_in_read == 0) return;
		int intv_idx = n_smems_fow_rev_scan[thread_read_num] + n_smems_fow[thread_read_num];
		int seed_begin_pos = seed_read_pos_fow_rev[intv_idx + offset_in_read].y;
		int comp_seed_begin_pos = seed_read_pos_fow_rev[intv_idx + offset_in_read - 1].y;
		if(seed_begin_pos == comp_seed_begin_pos) {
			seed_intervals_fow_rev[intv_idx + offset_in_read] =  make_uint2 (1, 0);

		}
	}

//	int intv_idx = read_offsets[thread_read_num] - (thread_read_num*min_seed_size);//thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//	if (tid < n_tasks) {
//		int smems_num = n_smems_fow[thread_read_num];
//		if (thread_read_idx >= smems_num) return;
//		int i;
//		int2 seed_pos_on_read = seed_read_pos_fow[intv_idx + thread_read_idx];
//		for (i = 0; i < smems_num; i++) {
//			int2 comp_seed_pos_on_read = seed_read_pos_fow[intv_idx + i];
//			/*if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y))*/
//			if ((seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)){
//				pass = 0;intv_idx
//				//break;
//
//			}
//		}
//		if (pass) {
//			int smems_num_rev = n_smems_rev[thread_read_num];
//			int i;
//			for (i = 0; i < smems_num_rev; i++) {
//				int2 comp_seed_pos_on_read = seed_read_pos_rev[intv_idx + i];
//				if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//								|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//								|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y)){
//					pass = 0;
//					//break;
//
//				}
//			}
//		}
//		if (pass == 0) {
//			seed_intervals_fow[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
//			seed_read_pos_fow[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
//		}
//
//	} else {
//		int smems_num = n_smems_rev[thread_read_num];
//		if (thread_read_idx >= smems_num) return;
//		int i;
//		int2 seed_pos_on_read = seed_read_pos_rev[intv_idx + thread_read_idx];
//		for (i = 0; i < smems_num; i++) {
//			int2 comp_seed_pos_on_read = seed_read_pos_rev[intv_idx + i];
//			/*if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && s$
//					eed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y))*/
//			if (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y){
//				pass = 0;
//				//break;
//
//			}
//		}
//		if (pass) {
//			int smems_num_fow = n_smems_fow[thread_read_num];
//			int i;
//			for (i = 0; i < smems_num_fow; i++) {
//				int2 comp_seed_pos_on_read = seed_read_pos_fow[intv_idx + i];
//				if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//						|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//						|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y)){
//					pass = 0;
//					//break;
//_compact_gpu
//				}
//			}
//		}
//		if (pass == 0) {
//			seed_intervals_rev[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
//			seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
//		}
//
//	}





	return;

}

__global__ void filter_seed_intervals_gpu_mem(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev_scan, uint32_t n_smems_max, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= 2*n_tasks) return;
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
		if(seed_begin_pos == comp_seed_begin_pos && ((seed_intervals_fow_rev_compact[intv_idx + offset_in_read].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read].x + 1) == (seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].x + 1))) {
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1] =  make_uint2 (1, 0);			//seed_read_pos_fow[intv_idx + offset_in_read].y =  -1;
		} else {
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1] = seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1];
		}
	} else {
		tid = tid - n_tasks;
		int thread_read_num = tid/n_smems_max;
		int offset_in_read = tid - (thread_read_num*n_smems_max);
		if(offset_in_read >= n_smems_rev[thread_read_num] /*|| offset_in_read == 0*/) return;
		int intv_idx = n_smems_fow_rev_scan[thread_read_num] + n_smems_fow[thread_read_num];
		if(offset_in_read == 0){
			seed_intervals_fow_rev[intv_idx + offset_in_read] = seed_intervals_fow_rev_compact[intv_idx + offset_in_read];
			return;
		}
		int seed_begin_pos = seed_read_pos_fow_rev_compact[intv_idx + offset_in_read].y;
		int comp_seed_begin_pos = seed_read_pos_fow_rev_compact[intv_idx + offset_in_read - 1].y;
		if(seed_begin_pos == comp_seed_begin_pos && ((seed_intervals_fow_rev_compact[intv_idx + offset_in_read].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read].x + 1) == (seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].x + 1))) {
			seed_intervals_fow_rev[intv_idx + offset_in_read] =  make_uint2 (1, 0);			//seed_read_pos_fow[intv_idx + offset_in_read].y =  -1;
		} else {
			seed_intervals_fow_rev[intv_idx + offset_in_read] = seed_intervals_fow_rev_compact[intv_idx + offset_in_read];
		}
	}

//	int intv_idx = read_offsets[thread_read_num] - (thread_read_num*min_seed_size);//thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//	if (tid < n_tasks) {
//		int smems_num = n_smems_fow[thread_read_num];
//		if (thread_read_idx >= smems_num) return;
//		int i;
//		int2 seed_pos_on_read = seed_read_pos_fow[intv_idx + thread_read_idx];
//		for (i = 0; i < smems_num; i++) {
//			int2 comp_seed_pos_on_read = seed_read_pos_fow[intv_idx + i];
//			/*if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y))*/
//			if ((seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)){
//				pass = 0;intv_idx
//				//break;
//
//			}
//		}
//		if (pass) {
//			int smems_num_rev = n_smems_rev[thread_read_num];
//			int i;
//			for (i = 0; i < smems_num_rev; i++) {
//				int2 comp_seed_pos_on_read = seed_read_pos_rev[intv_idx + i];
//				if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//								|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//								|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y)){
//					pass = 0;
//					//break;
//
//				}
//			}
//		}
//		if (pass == 0) {
//			seed_intervals_fow[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
//			seed_read_pos_fow[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
//		}
//
//	} else {
//		int smems_num = n_smems_rev[thread_read_num];
//		if (thread_read_idx >= smems_num) return;
//		int i;
//		int2 seed_pos_on_read = seed_read_pos_rev[intv_idx + thread_read_idx];
//		for (i = 0; i < smems_num; i++) {
//			int2 comp_seed_pos_on_read = seed_read_pos_rev[intv_idx + i];
//			/*if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && s$
//					eed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y))*/
//			if (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y){
//				pass = 0;
//				//break;
//
//			}
//		}
//		if (pass) {
//			int smems_num_fow = n_smems_fow[thread_read_num];
//			int i;
//			for (i = 0; i < smems_num_fow; i++) {
//				int2 comp_seed_pos_on_read = seed_read_pos_fow[intv_idx + i];
//				if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//						|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//						|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y)){
//					pass = 0;
//					//break;
//_compact_gpu
//				}
//			}
//		}
//		if (pass == 0) {
//			seed_intervals_rev[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
//			seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
//		}
//
//	}





	return;

}

__global__ void filter_seed_intervals_gpu_wrapper(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev,  int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev, uint32_t* n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,  uint32_t* n_ref_pos_fow_rev, uint32_t *n_smems_max, uint32_t *n_smems_sum_fow_rev, void *cub_sort_temp_storage, size_t cub_sort_storage_bytes, int total_reads, int n_bits_max_read_size, int is_smem) {

	uint32_t n_smems_max_val = n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1];
	int n_tasks = n_smems_max_val*total_reads;
	n_smems_sum_fow_rev[0] = n_smems_sum_fow_rev[0]/2;
	int BLOCKDIM = 128;
	int N_BLOCKS = (2*n_tasks + BLOCKDIM - 1)/BLOCKDIM;

	filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);
	cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact, (uint64_t*)seed_read_pos_fow_rev, (uint64_t*)seed_intervals_fow_rev_compact, (uint64_t*)seed_intervals_fow_rev,  n_smems_sum_fow_rev[0], total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size);
	count_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev, n_smems_fow_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val << 1, 2*n_tasks);

}

__global__ void filter_seed_intervals_gpu_wrapper_mem(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev, uint32_t *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan, uint32_t *n_ref_pos_fow_rev, uint32_t *n_smems_max, uint32_t *n_smems_sum_fow_rev, int total_reads) {

	uint32_t n_smems_max_val = n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1];
	int n_tasks = n_smems_max_val*total_reads;
	n_smems_sum_fow_rev[0] = n_smems_sum_fow_rev[0]/2;
	int BLOCKDIM = 128;
	int N_BLOCKS = (2*n_tasks + BLOCKDIM - 1)/BLOCKDIM;

	//filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);

	filter_seed_intervals_gpu_mem<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, seed_intervals_fow_rev, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);
	count_seed_intervals_gpu_mem<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val, n_tasks);

}


//__global__ void filter_seed_intervals_gpu(uint2 *seed_intervals_fow, uint2 *seed_intervals_rev, int2 *seed_read_pos_fow, int2 *seed_read_pos_rev, uint32_t *read_num, uint32_t *read_idx, uint32_t *n_smems_fow,  uint32_t *n_smems_rev, int min_seed_size, int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	int threads_per_read;
//	//int pass = 1;
//	if (tid >= 2*n_tasks) return;
//	int thread_read_num = read_num[tid%n_tasks];
//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//	if (tid < n_tasks) {
//		int smems_num = n_smems_fow[thread_read_num];
//		if (thread_read_idx >= smems_num) return;
//		int i;
//		int2 seed_pos_on_read = seed_read_pos_fow[intv_idx + thread_read_idx];
//		for (i = thread_read_idx + 1; i < smems_num; i++) {
//			int2 comp_seed_pos_on_read = seed_read_pos_fow[intv_idx + i];
//			/*if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y))*/
//			if ((seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)){
//				seed_intervals_fow[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
//				seed_read_pos_fow[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
//				//break;
//
//			} else if ((seed_pos_on_read.x == comp_seed_pos_on_read.x && comp_seed_pos_on_read.y < seed_pos_on_read.y)) {
//				seed_intervals_fow[intv_idx + i] =  make_uint2 (1, 0) ;
//				seed_read_pos_fow[intv_idx + i] =  make_int2 (-1, -1) ;
//
//			}
//		}
////		if (pass) {
////			int smems_num_rev = n_smems_rev[thread_read_num];
////			int i;
////			for (i = 0; i < smems_num_rev; i++) {
////				int2 comp_seed_pos_on_read = seed_read_pos_rev[intv_idx + i];
////				if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
////								|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
////								|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y)){
////					pass = 0;
////					//break;
////
////				}
////			}
////		}
////		if (pass == 0) {
////			seed_intervals_fow[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
////			seed_read_pos_fow[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
////		}
//
//	} else {
//		int smems_num = n_smems_rev[thread_read_num];
//		if (thread_read_idx >= smems_num) return;
//		int i;
//		int2 seed_pos_on_read = seed_read_pos_rev[intv_idx + thread_read_idx];
//		for (i = thread_read_idx + 1; i < smems_num; i++) {
//			int2 comp_seed_pos_on_read = seed_read_pos_rev[intv_idx + i];
//			/*if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && s$
//					eed_pos_on_read.y < comp_seed_pos_on_read.y)
//					|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y))*/
//			if (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y){
//				seed_intervals_rev[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
//				seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
//				//break;
//
//			} else if (comp_seed_pos_on_read.x > seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y) {
//				seed_intervals_rev[intv_idx + i] =  make_uint2 (1, 0) ;
//				seed_read_pos_rev[intv_idx + i] =  make_int2 (-1, -1) ;
//
//			}
//		}
////		if (pass) {
////			int smems_num_fow = n_smems_fow[thread_read_num];
////			int i;
////			for (i = 0; i < smems_num_fow; i++) {
////				int2 comp_seed_pos_on_read = seed_read_pos_fow[intv_idx + i];
////				if ((seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
////						|| (seed_pos_on_read.x == comp_seed_pos_on_read.x && seed_pos_on_read.y < comp_seed_pos_on_read.y)
////						|| (seed_pos_on_read.x > comp_seed_pos_on_read.x && seed_pos_on_read.y == comp_seed_pos_on_read.y)){
////					pass = 0;
////					//break;
////
////				}
////			}
////		}
////		if (pass == 0) {
////			seed_intervals_rev[intv_idx + thread_read_idx] =  make_uint2 (1, 0) ;
////			seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (-1, -1) ;
////		}
//
//	}
//
//
//
//
//
//	return;
//
//}

//__global__ void finalize_seeds_gpu(uint2 *seed_intervals_fow, uint2 *seed_intervals_rev, int2 *seed_read_pos_fow, int2 *seed_read_pos_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, int n_smems_max_fow, int n_smems_max_rev, int min_seed_size, int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= n_tasks) return;
//	int thread_read_num = tid/(n_smems_max_fow*n_smems_max_rev);
//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	int no_of_smems_fow = n_smems_fow[thread_read_num];
//	int no_of_smems_rev = n_smems_rev[thread_read_num];
//	int thread_read_offset = tid - thread_read_num*(n_smems_max_fow*n_smems_max_rev);
//	int smem1_idx = thread_read_offset/n_smems_max_fow;
//	int smem2_idx = thread_read_offset%n_smems_max_rev;
//	if (smem1_idx < no_of_smems_fow && smem2_idx < no_of_smems_rev) {
//		int2 smem1_read_pos =  seed_read_pos_fow[intv_idx + smem1_idx];
//		int2 smem2_read_pos =  seed_read_pos_rev[intv_idx + smem2_idx];
//		if ((smem1_read_pos.x > smem2_read_pos.x && smem1_read_pos.y < smem2_read_pos.y)
//				|| (smem1_read_pos.x == smem2_read_pos.x && smem1_read_pos.y < smem2_read_pos.y)
//				|| (smem1_read_pos.x > smem2_read_pos.x && smem1_read_pos.y == smem2_read_pos.y)){
//			seed_read_pos_fow[intv_idx + smem1_idx] = make_int2 (-1, -1);
//			seed_intervals_fow[intv_idx + smem1_idx] = make_uint2 (1, 0);
//		}
//		else if ((smem2_read_pos.x > smem1_read_pos.x && smem2_read_pos.y < smem1_read_pos.y)
//				|| (smem2_read_pos.x == smem1_read_pos.x && smem2_read_pos.y < smem1_read_pos.y)
//				|| (smem2_read_pos.x > smem1_read_pos.x && smem2_read_pos.y == smem1_read_pos.y)){
//			seed_read_pos_rev[intv_idx + smem1_idx] = make_int2 (-1, -1);
//			seed_intervals_rev[intv_idx + smem1_idx] = make_uint2 (1, 0);
//		}
//
//	}
//
//
//
//	return;
//
//}

//__global__ void filter_seeds_gpu(uint2 *seed_intervals, int2 *seed_read_pos, uint32_t *n_smems, int n_smems_max, int is_fow, int min_seed_size, int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= n_tasks) return;
//	int thread_read_num = tid/(n_smems_max*n_smems_max);
//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	int no_of_smems = n_smems[thread_read_num];
//	int thread_read_offset = tid - thread_read_num*(n_smems_max*n_smems_max);
//	int smem1_idx = thread_read_offset/n_smems_max;
//	int smem2_idx = thread_read_offset%n_smems_max;
//	if (smem1_idx < no_of_smems && smem2_idx < no_of_smems && smem1_idx != smem2_idx) {
//		int2 smem1_read_pos =  seed_read_pos[intv_idx + smem1_idx];
//		int2 smem2_read_pos =  seed_read_pos[intv_idx + smem2_idx];
//		if (is_fow) {
//			if (smem1_read_pos.x == smem2_read_pos.x && smem1_read_pos.y < smem2_read_pos.y){
//				seed_read_pos[intv_idx + smem1_idx] = make_int2 (-1, -1);
//				seed_intervals[intv_idx + smem1_idx] = make_uint2 (1, 0);
//			}
//		} else {
//			if (smem1_read_pos.x > smem2_read_pos.x && smem1_read_pos.y == smem2_read_pos.y){
//				seed_read_pos[intv_idx + smem1_idx] = make_int2 (-1, -1);
//				seed_intervals[intv_idx + smem1_idx] = make_uint2 (1, 0);
//			}
//		}
//
//	}
//
//
//
//	return;
//
//}
//
//
//


//__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow, uint2 *seed_intervals_rev,
//		int2 *seed_read_pos_fow, int2 *seed_read_pos_rev, uint32_t *read_num, uint32_t *read_idx, /*uint2* pre_calc_intervals,*/ int min_seed_size, bwt_t_gpu bwt, /*int pre_calc_seed_len,*/ int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= 2*n_tasks) return;
//	int thread_read_num = read_num[tid%n_tasks];
//	int read_len = read_sizes[thread_read_num];
//	int read_off = read_offsets[thread_read_num];
//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//
//
//	int i;
//	bwtint_t_gpu prev_l = 0, prev_u = bwt.seq_len;
//	//printf("prev_l=%u, prev_u=%u\n", prev_l, prev_u);
////	uint32_t pre_calc_seed = 0;
////	for (i = 0; i < pre_calc_seed_len; i++, thread_read_idx++) {
////		int reg_no = thread_read_idx >> 3;
////		int reg_pos = thread_read_idx & 7;
////		int reg = read_batch_gpu[read_off + reg_no];
////		int base = reg >> (28 - (reg_pos << 2));
////		if (base > 3) {
////			thread_read_idx = MAX_READ_LENGTH;
////			break;
////		}
////		pre_calc_seed |= (base << (i << 1));
////	}
////	uint2 start_seed_intervals = make_uint2(0,0);
////	if (thread_read_idx < MAX_READ_LENGTH) {
////		uint2 start_seed_intervals = pre_calc_intervals[pre_calc_seed];if (tid < n_tasks) {
////	}
////	prev_l = start_seed_intervals.x;
////	prev_u = start_seed_intervals.y;
//	int base;
//	bwtint_t_gpu l, u;
//	int donot_add = 0;
//	if (tid < n_tasks) {
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_fow[read_off >> 3]);
//		for (i = start + read_len - thread_read_idx - 1 ; i >= start; i--) {
//			/*get the base*/
//			int reg_no = i >> 3;
//			int reg_pos = i & 7;
//			int reg = seq[reg_no];
//			int base = (reg >> (28 - (reg_pos << 2)))&15;
//			/*unknown bases*/
//			if (base > 3) {
//				break;
//			}
//			//printf("prev_l=%u, prev_u=%u, intv=%d\n", prev_l, prev_u, prev_u - prev_l + 1);
//			uint2 intv = find_occ_gpu(bwt, prev_l - 1, prev_u, base);
//			//calculate the range
////			l = L2_gpu[base] + bwt_occ_gpu(bwt, prev_l - 1, base) + 1;
////			u = L2_gpu[base] + bwt_occ_gpu(bwt, prev_u, base);
//			l = L2_gpu[base] + intv.x + 1;
//			u = L2_gpu[base] + intv.y;
//			if (l > u) {
//				break;
//			}
//
//			prev_l = l;
//			prev_u = u;
//		}
//		seed_intervals_fow[intv_idx + thread_read_idx] = make_uint2(prev_l, prev_u);
//		seed_read_pos_fow[intv_idx + thread_read_idx] = make_int2 (i - start + 1, read_len - thread_read_idx) ;
//	}
//	else {
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_rev[read_off >> 3]);
//		for (i = start + thread_read_idx ; i < read_len + start; i++) {
//			/*get the base*/
//			int reg_no = i >> 3;
//			int reg_pos = i & 7;
//			int reg = seq[reg_no];
//			int base = (reg >> (28 - (reg_pos << 2)))&15;
//			/*unknown bases*/
//			if (base > 3) {
//				break;
//			}
//
//			uint2 intv = find_occ_gpu(bwt, prev_l - 1, prev_u, base);
//			//calculate the rangei
//			//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//			//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//			l = L2_gpu[base] + intv.x + 1;
//			u = L2_gpu[base] + intv.y;
//			if (l > u) {
//				break;
//			}
//
//			prev_l = l;
//			prev_u = u;
//		}
//		seed_intervals_rev[intv_idx + thread_read_idx] = make_uint2(prev_l, prev_u);
//		seed_read_pos_rev[intv_idx + thread_read_idx] = make_int2 (read_len + start - i, read_len - thread_read_idx) ;
//	}
//
//	return;
//
//}

//__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow, uint2 *seed_intervals_rev,
//		int2 *seed_read_pos_fow, int2 *seed_read_pos_rev, uint32_t *read_num, uint32_t *read_idx, /*uint2* pre_calc_intervals,*/ int min_seed_size, bwt_t_gpu bwt, /*int pre_calc_seed_len,*/ int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= 2*n_tasks) return;
//	int thread_read_num = read_num[tid%n_tasks];
//	int read_len = read_sizes[thread_read_num];
//	int read_off = read_offsets[thread_read_num];
//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//	int is_active = 1;
//	int is_shfl = tid%n_tasks ? 1 : 0;
//	if (is_shfl) is_shfl = (thread_read_num == read_num[(tid%n_tasks) - 1]) ? 1 : 0;
//	if (is_shfl) is_shfl = (tid%32) ? 1 : 0;
//
//	int i;
//	bwtint_t_gpu prev_l = 0, prev_u = bwt.seq_len;
////	uint32_t pre_calc_seed = 0;
////	for (i = 0; i < pre_calc_seed_len; i++, thread_read_idx++) {
////		int reg_no = thread_read_idx >> 3;
////		int reg_pos = thread_read_idx & 7;
////		int reg = read_batch_gpu[read_off + reg_no];
////		int base = reg >> (28 - (reg_pos << 2));
////		if (base > 3) {
////			thread_read_idx = MAX_READ_LENGTH;
////			break;
////		}
////		pre_calc_seed |= (base << (i << 1));
////	}
////	uint2 start_seed_intervals = make_uint2(0,0);
////	if (thread_read_idx < MAX_READ_LENGTH) {
////		uint2 start_seed_intervals = pre_calc_intervals[pre_calc_seed];
////	}
////	prev_l = start_seed_intervals.x;
////	prev_u = start_seed_intervals.y;
//	int base;
//	bwtint_t_gpu l = 0, u = bwt.seq_len;
//	int donot_add = 0;
//	if (tid < n_tasks) {
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_fow[read_off >> 3]);
//		uint32_t curr_intv_size = bwt.seq_len - 0 + 1;
//		uint32_t prev_intv_size = 0;
//		int beg_i;
//		int itr = 0;
//		for (i = start + read_len - thread_read_idx - 1 ; i >= start; i--, itr++) {
//			/*get the base*/
//			if (is_active) {
//				prev_u = u;
//				prev_l = l;
//				int reg_no = i >> 3;
//				int reg_pos = i & 7;
//				int reg = seq[reg_no];
//				int base = (reg >> (28 - (reg_pos << 2)))&15;
//				/*unknown bases*/
//				if (base > 3) {
//					//break;
//					is_active = 0;
//
//				}
//
//				uint2 intv = find_occ_gpu(bwt, prev_l - 1, prev_u, base);
//				//calculate the range
//				//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//				//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//				l = L2_gpu[base] + intv.x + 1;
//				u = L2_gpu[base] + intv.y;
//
//				if (l > u) {
//					//break;
//					is_active = 0;
//				}
//				beg_i = i;
//			}
//				//if (tid == 26 ||tid == 27 || tid == 28) printf("%d-->%d,%d, ",tid, u-l+1, itr);
//				uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 1);
//				uint32_t curr_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, 1);
//				if (is_shfl && curr_neighbour_active && prev_intv_size == neighbour_intv_size) {
//					if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//
//				prev_intv_size =  curr_intv_size;
//				curr_intv_size = u - l + 1;
//
//
//
//
//		}
//		seed_intervals_fow[intv_idx + thread_read_idx] = make_uint2(prev_l, prev_u);
//		seed_read_pos_fow[intv_idx + thread_read_idx] = make_int2 (beg_i - start + 1, read_len - thread_read_idx) ;
//	}
//	else {
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_rev[read_off >> 3]);
//		for (i = start + thread_read_idx ; i < read_len + start; i++) {
//			/*get the base*/
//			int reg_no = i >> 3;
//			int reg_pos = i & 7;
//			int reg = seq[reg_no];
//			int base = (reg >> (28 - (reg_pos << 2)))&15; make_int2 (read_len + start - i, read_len - thread_read_idx) ;
//			/*unknown bases*/
//			if (base > 3) {
//				break;
//			}
//
//			uint2 intv = find_occ_gpu(bwt, prev_l - 1, prev_u, base);
//			//calculate the rangei
//			//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//			//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//			l = L2_gpu[base] + intv.x + 1;
//			u = L2_gpu[base] + intv.y;
//			if (l > u) {
//				break;
//			}
//
//			prev_l = l;
//			prev_u = u;
//		}
//		seed_intervals_rev[intv_idx + thread_read_idx] = make_uint2(prev_l, prev_u);
//		seed_read_pos_rev[intv_idx + thread_read_idx] = make_int2 (read_len + start - i , read_len - thread_read_idx) ;
//	}
//
//	return;
//
//}
//__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow, uint2 *seed_intervals_rev,
//		int2 *seed_read_pos_fow, int2 *seed_read_pos_rev, uint32_t *read_num, uint32_t *read_idx, uint2* pre_calc_intervals, int min_seed_size, bwt_t_gpu bwt, int pre_calc_seed_len, int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= 2*n_tasks) return;
//	int thread_read_num = read_num[tid%n_tasks];
//	int read_len = read_sizes[thread_read_num];
//	int read_off = read_offsets[thread_read_num];
//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//
//
//	int i, j;
//	int base;
//	bwtint_t_gpu l, u;
//	if (tid < n_tasks) {
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_fow[read_off >> 3]);
//		uint32_t pre_calc_seed = 0;
//		for (i = start + read_len - thread_read_idx - 1, j = 0; j < pre_calc_seed_len; i--, j++) {
//			int reg_no = i >> 3;
//			int reg_pos = i & 7;
//			int reg = seq[reg_no];
//			uint32_t base = (reg >> (28 - (reg_pos << 2)))&15;
//			/*unknown bases*/
//			if (base > 3) {
//				break;
//			}
//			pre_calc_seed |= (base << (j<<1));
//		}
//		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
//		if(prev_seed_interval.x <= prev_seed_interval.y) {
//
//			for (; i >= start; i--) {
//				/*get the base*/
//				int reg_no = i >> 3;
//				int reg_pos = i & 7;
//				int reg = seq[reg_no];
//				int base = (reg >> (28 - (reg_pos << 2)))&15;
//				/*unknown bases*/
//				if (base > 3) {
//					break;
//				}
//
//				uint2 intv = find_occ_gpu(bwt, prev_seed_interval.x - 1, prev_seed_interval.y, base);
//				//calculate the range
//				//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//				//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//				l = L2_gpu[base] + intv.x + 1;
//				u = L2_gpu[base] + intv.y;
//				if (l > u) {
//					break;
//				}
//
//				prev_seed_interval = make_uint2(l,u);
//			}
//		}
//		seed_intervals_fow[intv_idx + thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
//		seed_read_pos_fow[intv_idx + thread_read_idx] = make_int2 (i - start + 1, read_len - thread_read_idx);
//	}
//	else {
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_rev[read_off >> 3]);
//		uint32_t pre_calc_seed = 0;
//		for (i = start + thread_read_idx, j = 0; j < pre_calc_seed_len; i++, j++) {
//			int reg_no = i >> 3;
//			int reg_pos = i & 7;
//			int reg = seq[reg_no];
//			uint32_t base = (reg >> (28 - (reg_pos << 2)))&15;
//			/*unknown bases*/
//			if (base > 3) {
//				break;

//			}
//			pre_calc_seed |= (base << (j<<1));
//		}
//		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
//		if(prev_seed_interval.x <= prev_seed_interval.y) {
//			for (; i < read_len + start; i++) {
//				/*get the base*/
//				int reg_no = i >> 3;
//				int reg_pos = i & 7;
//				int reg = seq[reg_no];
//				int base = (reg >> (28 - (reg_pos << 2)))&15;
//				/*unknown bases*/
//				if (base > 3) {
//					break;
//				}
//
//				uint2 intv = find_occ_gpu(bwt, prev_seed_interval.x - 1, prev_seed_interval.y, base);
//				//calculate the range
//				//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//				//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//				l = L2_gpu[base] + intv.x + 1;
//				u = L2_gpu[base] + intv.y;
//				if (l > u) {
//					break;
//				}
//
//				prev_seed_interval = make_uint2(l,u);
//			}

//		}
//		seed_intervals_rev[intv_idx + thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
//		seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (read_len + start - i, read_len - thread_read_idx) ;
//	}
//
//	return;
//
//}

//#define N_SHUFFLES 20
//__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow_rev,
//		int2 *seed_read_pos_fow_rev, uint32_t *read_num, uint32_t *read_idx, uint32_t *is_smem_fow_rev_flag, uint2* pre_calc_intervals, uint32_t *n_smems_fow,  uint32_t *n_smems_rev,int min_seed_size, bwt_t_gpu bwt, int pre_calc_seed_len, int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= 2*n_tasks) return;
//	int thread_read_num = read_num[tid%n_tasks];
//	int read_len = read_sizes[thread_read_num];
//	int read_off = read_offsets[thread_read_num];
//	//thread_read_num * (MAX_READ_LENGTH - min_seed_size);
//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
//	int is_active = 0;
//	int is_smem = 1;
////	int is_shfl[N_SHUFFLES];
////	int only_next_time = 0;
////	uint32_t neighbour_active[N_SHUFFLES];
////	uint32_t prev_intv_size[N_SHUFFLES];
////
////	int m;
////	for (m = 0; m < N_SHUFFLES; m++) {
////		is_shfl[m] = tid%n_tasks ? 1 : 0;
////		if (is_shfl[m]) is_shfl[m] = (thread_read_num == read_num[(tid%n_tasks) - (m+1)]) ? 1 : 0;
////		if (is_shfl[m]) is_shfl[m] = ((tid%32) - m > 0) ? 1 : 0;
////		prev_intv_size[m] = 0;
////		neighbour_active[m] = 1;
////	}
//
////	int is_shfl = tid%n_tasks ? 1 : 0;
////	if (is_shfl) is_shfl = (thread_read_num == read_num[(tid%n_tasks) - 1]) ? 1 : 0;
////	if (is_shfl) is_shfl = (tid%32) ? 1 : 0;
////	int is_shfl_2 = (thread_read_num == read_num[(tid%n_tasks) - 2]) ? 1 : 0;
////	if (is_shfl_2) is_shfl_2 = (tid%33) ? 1 : 0;
////	int is_shfl_3 = (thread_read_num == read_num[(tid%n_tasks) - 3]) ? 1 : 0;
////	if (is_shfl_3) is_shfl_3 = (tid%34) ? 1 : 0;
////	int is_shfl_4 = (thread_read_num == read_num[(tid%n_tasks) - 4]) ? 1 : 0;
////	if (is_shfl_4) is_shfl_4 = (tid%35) ? 1 : 0;
////	int is_shfl_5 = (thread_read_num == read_num[(tid%n_tasks) - 5]) ? 1 : 0;
////	if (is_shfl_5) is_shfl_5 = (tid%36) ? 1 : 0;
//	int i, j;
//	int base;
//	bwtint_t_gpu l, u;
//	if (tid < n_tasks) {
//		int intv_idx = (2*(read_offsets[thread_read_num] - (thread_read_num*min_seed_size))) + read_len - min_seed_size - 1;
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_fow[read_off >> 3]);
////		uint32_t pre_calc_seed = 0;
////		for (i = start + read_len - thread_read_idx - 1, j = 0; j < pre_calc_seed_len; i--, j++) {
////			int reg_no = i >> 3;
////			int reg_pos = i & 7;
////			int reg = seq[reg_no];
////			uint32_t base = (reg >> (28 - (r//__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow_rev,
//		int2 *seed_read_pos_fow_rev, uint32_t *read_num, uint32_t *read_idx, uint32_t *is_smem_fow_rev_flag, uint2* pre_calc_intervals, uint32_t *n_smems_fow,  uint32_t *n_smems_rev,int min_seed_size, bwt_t_gpu bwt, int pre_calc_seed_len, int n_tasks) {eg_pos << 2)))&15;
////			/*unknown bases*/
////			if (base > 3) {
////				break;
////			}
////			pre_calc_seed |= (base << (j<<1));
////		}
////		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
//		uint2 prev_seed_interval; //= make_uint2(0, bwt.seq_len);
//		//int beg_i = i;
//		int beg_i = i = start + read_len - thread_read_idx - 1;
//		//if(prev_seed_interval.x <= prev_seed_interval.y) {
//			is_active = 1;
//			//uint32_t curr_intv_size = prev_seed_interval.y - prev_seed_interval.x + 1;
////			uint32_t prev_intv_size = 0;
////			uint32_t prev_2_intv_size = 0;
////			uint32_t prev_3_intv_size = 0;
////			uint32_t prev_4_intv_size = 0;
////			uint32_t prev_5_intv_size = 0;
//			l = 0, u = bwt.seq_len;//prev_seed_interval.x, u = prev_seed_interval.y;
//			for (; i >= start; i--) {
//				/*get the base*/
//				if (is_active) {
//					prev_seed_interval = make_uint2(l,u);
//					int reg_no = i >> 3;
//					int reg_pos = i & 7;
//					int reg = seq[reg_no];
//					int base = (reg >> (28 - (reg_pos << 2)))&15;
//					/*unknown bases*/
//					if (base > 3) {
//						//is_active = 0;
//						break;
//					}
//
//					uint2 intv = find_occ_gpu(bwt, prev_seed_interval.x - 1, prev_seed_interval.y, base);
//					//calculate the range
//					//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//					//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//					l = L2_gpu[base] + intv.x + 1;
//					u = L2_gpu[base] + intv.y;
//
//
//					//beg_i = (i == start) ? i - 1 : i;
//				}
//				//if (tid == 26 ||tid == 27 || tid == 28) printf("%d-->%d,%d, ",tid, u-l+1, itr);
////				uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 1);
////				uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, 1);
////				uint32_t neighbour_2_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 2);
////				uint32_t is_neighbour_2_active = __shfl_up_sync(0xFFFFFFFF, is_active, 2);
////				uint32_t neighbour_3_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 3);
////				uint32_t is_neighbour_3_active = __shfl_up_sync(0xFFFFFFFF, is_active, 3);
////				uint32_t neighbour_4_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 4);
////				uint32_t is_neighbour_4_active = __shfl_up_sync(0xFFFFFFFF, is_active, 4);
////				uint32_t neighbour_5_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 5);
////				uint32_t is_neighbour_5_active = __shfl_up_sync(0xFFFFFFFF, is_active, 5);
//
////				for (m = 0; m <N_SHUFFLES; m++){
////					uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, m+1);
////					uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, m+1);
////					if(neighbour_active[m]) neighbour_active[m] = is_neighbour_active;
////					if (is_shfl[m] && neighbour_active[m] && prev_intv_size[m] == neighbour_intv_size) {
////						//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////						is_active = 0;
////						is_smem = 0;
////						break;
////						//prev_seed_interval = make_uint2(m,m);
////					}
////					//if(is_shfl[m] == 0) break;
////				}
////				only_next_time = is_active ? only_next_time : only_next_time + 1;
////				if(only_next_time == 2) break;
////
////
////				for (m = N_SHUFFLES - 1; m >= 1; m--){
////					prev_intv_size[m] = prev_intv_size[m-1];
////				}
////				prev_intv_size[0] = curr_intv_size;
////				if (is_shfl && is_neighbour_active && prev_intv_size == neighbour_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////				if (is_shfl_2 && is_neighbour_2_active && prev_2_intv_size == neighbour_2_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////
////				if (is_shfl_3 && is_neighbour_3_active && prev_3_intv_size == neighbour_3_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////
////				if (is_shfl_4 && is_neighbour_4_active && prev_4_intv_size == neighbour_4_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////				if (is_shfl_5 && is_neighbour_5_active && prev_5_intv_size == neighbour_5_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////
////				prev_5_intv_size = prev_4_intv_size;
////				prev_4_intv_size = prev_3_intv_size;
////				prev_3_intv_size = prev_2_intv_size;
////				prev_2_intv_size = prev_intv_size;
////				prev_intv_size =  curr_intv_size;
////				if (l > u || base > 3) {
////					is_active = 0;
////				}
////
////				curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;
//
//			}
//		//}
//			beg_i = (i == start) ? i - 1 : i;
//		if (read_len - thread_read_idx - beg_i + start - 1 >= min_seed_size && is_smem) {
//			atomicAdd(&n_smems_fow[thread_read_num], 1);
//			seed_intervals_fow_rev[intv_idx - thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
//			seed_read_pos_fow_rev[intv_idx - thread_read_idx] = make_int2 (beg_i - start + 1, read_len - thread_read_idx);
//			is_smem_fow_rev_flag[intv_idx - thread_read_idx] = 0x00010001;
//		}
//
//	}
//	else {
//		int intv_idx = 2*(read_offsets[thread_read_num] - (thread_read_num*min_seed_size)) + read_len - min_seed_size;
//		int start = read_off&7;
//		uint32_t *seq = &(packed_read_batch_rev[read_off >> 3]);
////		uint32_t pre_calc_seed = 0;
////		for (i = start + thread_read_idx, j = 0; j < pre_calc_seed_len; i++, j++) {
////			int reg_no = i >> 3;
////			int reg_pos = i & 7;
////			int reg = seq[reg_no];
////			uint32_t base = (reg >> (28 - (reg_pos << 2)))&15;
////			/*unknown bases*/
////			if (base > 3) {
////
////				break;
////			}
////			pre_calc_seed |= (base << (j<<1));
////		}
//		//uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
//		uint2 prev_seed_interval ;//= make_uint2(0, bwt.seq_len);
//		//int beg_i = i;
//		int beg_i = i = start + thread_read_idx;
//		//if(prev_seed_interval.x <= prev_seed_interval.y) {
//			is_active = 1;
//			//uint32_t curr_intv_size = prev_seed_interval.y - prev_seed_interval.x + 1;
////			uint32_t prev_intv_size = 0;
////			uint32_t prev_2_intv_size = 0;
////			uint32_t prev_3_intv_size = 0;
////			uint32_t prev_4_intv_size = 0;
////			uint32_t prev_5_intv_size = 0;
//			l = 0, u = bwt.seq_len;//prev_seed_interval.x, u = prev_seed_interval.y;
//			for (; i < read_len + start; i++) {
//				/*get the base*/
//				if (is_active) {
//					prev_seed_interval = make_uint2(l,u);
//					int reg_no = i >> 3;
//					int reg_pos = i & 7;
//					int reg = seq[reg_no];
//					int base = (reg >> (28 - (reg_pos << 2)))&15;
//					/*unknown bases*/
//					if (base > 3) {
//						break;
//						//is_active = 0;
//					}
//
//					uint2 intv = find_occ_gpu(bwt, prev_seed_interval.x - 1, prev_seed_interval.y, base);
//					//calculate the range
//					//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
//					//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
//					l = L2_gpu[base] + intv.x + 1;
//					u = L2_gpu[base] + intv.y;
////					if (l > u) {
////						//break;
////						is_active = 0;
////					}
//					//beg_i = i == (read_len + start - 1) ? read_len + start : i;
//				}
////				uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 1);
////				uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, 1);
////				uint32_t neighbour_2_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 2);
////				uint32_t is_neighbour_2_active = __shfl_up_sync(0xFFFFFFFF, is_active, 2);
////				uint32_t neighbour_3_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 3);
////				uint32_t is_neighbour_3_active = __shfl_up_sync(0xFFFFFFFF, is_active, 3);
////				uint32_t neighbour_4_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 4);
////				uint32_t is_neighbour_4_active = __shfl_up_sync(0xFFFFFFFF, is_active, 4);
////				uint32_t neighbour_5_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 5);
////				uint32_t is_neighbour_5_active = __shfl_up_sync(0xFFFFFFFF, is_active, 5);
////				for (m = 0; m < N_SHUFFLES; m++){
////					uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, m+1);
////					uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, m+1);
////					if(neighbour_active[m]) neighbour_active[m] = is_neighbour_active;
////					if (is_shfl[m] && neighbour_active[m] && prev_intv_size[m] == neighbour_intv_size) {
////						//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////						is_active = 0;
////						is_smem = 0;
////						break;
////					}
////					//if (is_shfl[m] == 0) break;
////				}
////				only_next_time = is_active ? only_next_time : only_next_time + 1;
////				if(only_next_time == 2) break;
////
////				for (m = N_SHUFFLES - 1 ; m >= 1; m--){
////					prev_intv_size[m] = prev_intv_size[m-1];
////				}
////				prev_intv_size[0] = curr_intv_size;
////
////				if (is_shfl && is_neighbour_active && prev_intv_size == neighbour_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////				if (is_shfl_2 && is_neighbour_2_active && prev_2_intv_size == neighbour_2_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
//
////
////				if (is_shfl_3 && is_neighbour_3_active && prev_3_intv_size == neighbour_3_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////
////				if (is_shfl_4 && is_neighbour_4_active && prev_4_intv_size == neighbour_4_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////				if (is_shfl_5 && is_neighbour_5_active && prev_5_intv_size == neighbour_5_intv_size) {
////					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
////					is_active = 0;
////					//break;
////				}
////
////				prev_5_intv_size = prev_4_intv_size;
////				prev_4_intv_size = prev_3_intv_size;
////				prev_3_intv_size = prev_2_intv_size;
////				prev_2_intv_size = prev_intv_size;
////				prev_intv_size =  curr_intv_size;
////				if (l > u || base > 3) {
////					is_active = 0;
////				}
////
////				curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;
//			}
//			beg_i = i == (read_len + start - 1) ? read_len + start : i;
//			if (beg_i - start - thread_read_idx >= min_seed_size && is_smem) {
//				atomicAdd(&n_smems_rev[thread_read_num], 1);
//				seed_intervals_fow_rev[intv_idx + thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
//				//seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (read_len + start - beg_i, read_len - thread_read_idx) ;
//				seed_read_pos_fow_rev[intv_idx + thread_read_idx] =  make_int2 (thread_read_idx|0x80000000, (beg_i - start)|0x80000000) ;
//				is_smem_fow_rev_flag[intv_idx + thread_read_idx]=0x00010001;
//			}
//		//}
////		seed_intervals_rev[intv_idx + thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
////		//seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (read_len + start - beg_i, read_len - thread_read_idx) ;
////		seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (thread_read_idx, beg_i - start) ;
//	}
//
//	return;
//
//}

#define N_SHUFFLES 30
__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow_rev,
		int2 *seed_read_pos_fow_rev, uint32_t *read_num, uint32_t *read_idx, uint32_t *is_smem_fow_rev_flag, uint2* pre_calc_intervals, uint32_t *n_smems_fow,  uint32_t *n_smems_rev,int min_seed_size, bwt_t_gpu bwt, int pre_calc_seed_len, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= 2*n_tasks) return;
	int thread_read_num = read_num[tid%n_tasks];
	int read_len = read_sizes[thread_read_num];
	int read_off = read_offsets[thread_read_num];
	//thread_read_num * (MAX_READ_LENGTH - min_seed_size);
	uint32_t thread_read_idx = read_idx[tid%n_tasks];
	int is_active = 0;
	int is_smem = 1;
	int is_shfl[N_SHUFFLES];
	int only_next_time = 0;
	uint32_t neighbour_active[N_SHUFFLES];
	uint32_t prev_intv_size[N_SHUFFLES];

	int m;
	for (m = 0; m < N_SHUFFLES; m++) {
		if (is_shfl[m]) is_shfl[m] = ((tid%32) - m > 0) ? 1 : 0;
		if (is_shfl[m]) is_shfl[m] = (tid%n_tasks) - (m+1) < 0 ? 0 : (thread_read_num == read_num[(tid%n_tasks) - (m+1)]) ? 1 : 0;
		prev_intv_size[m] = 0;
		neighbour_active[m] = 1;
	}

//	int is_shfl = tid%n_tasks ? 1 : 0;
//	if (is_shfl) is_shfl = (thread_read_num == read_num[(tid%n_tasks) - 1]) ? 1 : 0;
//	if (is_shfl) is_shfl = (tid%32) ? 1 : 0;
//	int is_shfl_2 = (thread_read_num == read_num[(tid%n_tasks) - 2]) ? 1 : 0;
//	if (is_shfl_2) is_shfl_2 = (tid%33) ? 1 : 0;
//	int is_shfl_3 = (thread_read_num == read_num[(tid%n_tasks) - 3]) ? 1 : 0;
//	if (is_shfl_3) is_shfl_3 = (tid%34) ? 1 : 0;
//	int is_shfl_4 = (thread_read_num == read_num[(tid%n_tasks) - 4]) ? 1 : 0;
//	if (is_shfl_4) is_shfl_4 = (tid%35) ? 1 : 0;
//	int is_shfl_5 = (thread_read_num == read_num[(tid%n_tasks) - 5]) ? 1 : 0;
//	if (is_shfl_5) is_shfl_5 = (tid%36) ? 1 : 0;
	int i, j;
	int base;
	bwtint_t_gpu l, u;
	if (tid < n_tasks) {
		int intv_idx = (2*(read_offsets[thread_read_num] - (thread_read_num*(min_seed_size-1)))) + read_len - min_seed_size;
		int start = read_off&7;
		uint32_t *seq = &(packed_read_batch_fow[read_off >> 3]);
		uint32_t pre_calc_seed = 0;
		for (i = start + read_len - thread_read_idx - 1, j = 0; j < pre_calc_seed_len; i--, j++) {
			int reg_no = i >> 3;
			int reg_pos = i & 7;
			int reg = seq[reg_no];
			uint32_t base = (reg >> (28 - (reg_pos << 2)))&15;
			/*unknown bases*/
			if (base > 3) {
				break;
			}
			pre_calc_seed |= (base << (j<<1));
		}
		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
		int beg_i = i;
		if(prev_seed_interval.x <= prev_seed_interval.y) {
			is_active = 1;
			uint32_t curr_intv_size = prev_seed_interval.y - prev_seed_interval.x + 1;
//			uint32_t prev_intv_size = 0;
//			uint32_t prev_2_intv_size = 0;
//			uint32_t prev_3_intv_size = 0;
//			uint32_t prev_4_intv_size = 0;
//			uint32_t prev_5_intv_size = 0;
			l = prev_seed_interval.x, u = prev_seed_interval.y;
			for (; i >= start; i--) {
				/*get the base*/
				if (is_active) {
					// moved at the bottom of the loop
					// prev_seed_interval = make_uint2(l,u);
					int reg_no = i >> 3;
					int reg_pos = i & 7;
					int reg = seq[reg_no];
					int base = (reg >> (28 - (reg_pos << 2)))&15;
					/*unknown bases*/
//					if (base > 3) {
//						is_active = 0;
//						break;
//					}

					uint2 intv = find_occ_gpu(bwt, prev_seed_interval.x - 1, prev_seed_interval.y, base);
					//calculate the range
					//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
					//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
					l = L2_gpu[base] + intv.x + 1;
					u = L2_gpu[base] + intv.y;


					// moved at the bottom of the loop
					//beg_i = (i == start) ? i - 1 : i;
				}
				//if (tid == 26 ||tid == 27 || tid == 28) printf("%d-->%d,%d, ",tid, u-l+1, itr);
//				uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 1);
//				uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, 1);
//				uint32_t neighbour_2_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 2);
//				uint32_t is_neighbour_2_active = __shfl_up_sync(0xFFFFFFFF, is_active, 2);
//				uint32_t neighbour_3_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 3);
//				uint32_t is_neighbour_3_active = __shfl_up_sync(0xFFFFFFFF, is_active, 3);
//				uint32_t neighbour_4_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 4);
//				uint32_t is_neighbour_4_active = __shfl_up_sync(0xFFFFFFFF, is_active, 4);
//				uint32_t neighbour_5_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 5);
//				uint32_t is_neighbour_5_active = __shfl_up_sync(0xFFFFFFFF, is_active, 5);

				for (m = 0; m <N_SHUFFLES; m++){
					uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, m+1);
					uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, m+1);
					if(neighbour_active[m]) neighbour_active[m] = is_neighbour_active;
					if (is_shfl[m] && neighbour_active[m] && prev_intv_size[m] == neighbour_intv_size) {
						//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
						is_active = 0;
						is_smem = 0;
						break;
						//prev_seed_interval = make_uint2(m,m);
					}
					//if(is_shfl[m] == 0) break;
				}
				only_next_time = is_active ? only_next_time : only_next_time + 1;
				if(only_next_time == 2) break;


				for (m = N_SHUFFLES - 1; m >= 1; m--){
					prev_intv_size[m] = prev_intv_size[m-1];
				}
				prev_intv_size[0] = curr_intv_size;
//				if (is_shfl && is_neighbour_active && prev_intv_size == neighbour_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//				if (is_shfl_2 && is_neighbour_2_active && prev_2_intv_size == neighbour_2_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//
//				if (is_shfl_3 && is_neighbour_3_active && prev_3_intv_size == neighbour_3_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//
//				if (is_shfl_4 && is_neighbour_4_active && prev_4_intv_size == neighbour_4_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//				if (is_shfl_5 && is_neighbour_5_active && prev_5_intv_size == neighbour_5_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//
//				prev_5_intv_size = prev_4_intv_size;
//				prev_4_intv_size = prev_3_intv_size;
//				prev_3_intv_size = prev_2_intv_size;
//				prev_2_intv_size = prev_intv_size;
//				prev_intv_size =  curr_intv_size;
				if (l > u || base > 3) {
					is_active = 0;
				}

				curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;

				if (is_active) {
					prev_seed_interval = make_uint2(l,u);
					beg_i = i - 1;	
				}

			}
		}
		if (read_len - thread_read_idx - beg_i + start - 1 >= min_seed_size && is_smem) {
			atomicAdd(&n_smems_fow[thread_read_num], 1);
			seed_intervals_fow_rev[intv_idx - thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
			seed_read_pos_fow_rev[intv_idx - thread_read_idx] = make_int2 (beg_i - start + 1, read_len - thread_read_idx);
			is_smem_fow_rev_flag[intv_idx - thread_read_idx] = 0x00010001;
		}

	}
	else {
		int intv_idx = 2*(read_offsets[thread_read_num] - (thread_read_num*(min_seed_size-1))) + read_len - min_seed_size + 1;
		int start = read_off&7;
		uint32_t *seq = &(packed_read_batch_rev[read_off >> 3]);
		uint32_t pre_calc_seed = 0;
		for (i = start + thread_read_idx, j = 0; j < pre_calc_seed_len; i++, j++) {
			int reg_no = i >> 3;
			int reg_pos = i & 7;
			int reg = seq[reg_no];
			uint32_t base = (reg >> (28 - (reg_pos << 2)))&15;
			/*unknown bases*/
			if (base > 3) {

				break;
			}
			pre_calc_seed |= (base << (j<<1));
		}
		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
		int beg_i = i;
		if(prev_seed_interval.x <= prev_seed_interval.y) {
			is_active = 1;
			uint32_t curr_intv_size = prev_seed_interval.y - prev_seed_interval.x + 1;
//			uint32_t prev_intv_size = 0;
//			uint32_t prev_2_intv_size = 0;
//			uint32_t prev_3_intv_size = 0;
//			uint32_t prev_4_intv_size = 0;
//			uint32_t prev_5_intv_size = 0;
			l = prev_seed_interval.x, u = prev_seed_interval.y;
			for (; i < read_len + start; i++) {
				/*get the base*/
				if (is_active) {
					// moved at the bottom of the loop
					//prev_seed_interval = make_uint2(l,u);
					int reg_no = i >> 3;
					int reg_pos = i & 7;
					int reg = seq[reg_no];
					int base = (reg >> (28 - (reg_pos << 2)))&15;
					/*unknown bases*/
//					if (base > 3) {
//						//break;
//						is_active = 0;
//					}

					uint2 intv = find_occ_gpu(bwt, prev_seed_interval.x - 1, prev_seed_interval.y, base);
					//calculate the range
					//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
					//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
					l = L2_gpu[base] + intv.x + 1;
					u = L2_gpu[base] + intv.y;
//					if (l > u) {
//						//break;
//						is_active = 0;
//					}
					// moved at the bottom of the loop
					//beg_i = i == (read_len + start - 1) ? read_len + start : i;
				}
//				uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 1);
//				uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, 1);
//				uint32_t neighbour_2_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 2);
//				uint32_t is_neighbour_2_active = __shfl_up_sync(0xFFFFFFFF, is_active, 2);
//				uint32_t neighbour_3_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 3);
//				uint32_t is_neighbour_3_active = __shfl_up_sync(0xFFFFFFFF, is_active, 3);
//				uint32_t neighbour_4_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 4);
//				uint32_t is_neighbour_4_active = __shfl_up_sync(0xFFFFFFFF, is_active, 4);
//				uint32_t neighbour_5_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, 5);
//				uint32_t is_neighbour_5_active = __shfl_up_sync(0xFFFFFFFF, is_active, 5);
				for (m = 0; m < N_SHUFFLES; m++){
					uint32_t neighbour_intv_size = __shfl_up_sync(0xFFFFFFFF, curr_intv_size, m+1);
					uint32_t is_neighbour_active = __shfl_up_sync(0xFFFFFFFF, is_active, m+1);
					if(neighbour_active[m]) neighbour_active[m] = is_neighbour_active;
					if (is_shfl[m] && neighbour_active[m] && prev_intv_size[m] == neighbour_intv_size) {
						//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
						is_active = 0;
						is_smem = 0;
						break;
					}
					//if (is_shfl[m] == 0) break;
				}
				only_next_time = is_active ? only_next_time : only_next_time + 1;
				if(only_next_time == 2) break;

				for (m = N_SHUFFLES - 1 ; m >= 1; m--){
					prev_intv_size[m] = prev_intv_size[m-1];
				}
				prev_intv_size[0] = curr_intv_size;
//
//				if (is_shfl && is_neighbour_active && prev_intv_size == neighbour_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//				if (is_shfl_2 && is_neighbour_2_active && prev_2_intv_size == neighbour_2_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}

//
//				if (is_shfl_3 && is_neighbour_3_active && prev_3_intv_size == neighbour_3_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//
//				if (is_shfl_4 && is_neighbour_4_active && prev_4_intv_size == neighbour_4_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//				if (is_shfl_5 && is_neighbour_5_active && prev_5_intv_size == neighbour_5_intv_size) {
//					//if (tid == 26 ||tid == 27 || tid == 28) printf("I am out thread_read_idx = %d, %d\n", thread_read_idx, i-start+1);
//					is_active = 0;
//					//break;
//				}
//
//				prev_5_intv_size = prev_4_intv_size;
//				prev_4_intv_size = prev_3_intv_size;
//				prev_3_intv_size = prev_2_intv_size;
//				prev_2_intv_size = prev_intv_size;
//				prev_intv_size =  curr_intv_size;
				if (l > u || base > 3) {
					is_active = 0;
				}

				curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;

				if (is_active) {
					prev_seed_interval = make_uint2(l,u);
					beg_i = i + 1;
				}
			}
			if (beg_i - start - thread_read_idx >= min_seed_size && is_smem) {
				atomicAdd(&n_smems_rev[thread_read_num], 1);
				seed_intervals_fow_rev[intv_idx + thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
				//seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (read_len + start - beg_i, read_len - thread_read_idx) ;
				seed_read_pos_fow_rev[intv_idx + thread_read_idx] =  make_int2 (thread_read_idx|0x80000000, (beg_i - start)|0x80000000) ;
				is_smem_fow_rev_flag[intv_idx + thread_read_idx]=0x00010001;
			}
		}
//		seed_intervals_rev[intv_idx + thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
//		//seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (read_len + start - beg_i, read_len - thread_read_idx) ;
//		seed_read_pos_rev[intv_idx + thread_read_idx] =  make_int2 (thread_read_idx, beg_i - start) ;
	}

	return;

}


//__global__ void pack_2bit(uint32_t* read_batch, uint32_t* packed_read_batch, int packed_read_offset, uint32_t* packed_read_4bit, uint32_t* packed_ref_4bit,
//						int read_tasks_per_thread, uint32_t total_packed_read_regs) {
//
//	int32_t i;
//	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index.
//	uint32_t n_threads = gridDim.x * blockDim.x;
//	for (i = 0; i < read_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1) < total_packed_read_regs); ++i) {
//		uint2 reg = ((uint2*)(read_batch))[(i*n_threads)<<1];
//		uint32_t pack_reg_4bit = 0;
//		pack_reg_4bit |= (reg.x & 15) << 28;        // ---
//		pack_reg_4bit |= ((reg.x >> 8) & 15) << 24; //    |
//		pack_reg_4bit |= ((reg.x >> 16) & 15) << 20;//    |
//		pack_reg_4bit |= ((reg.x >> 24) & 15) << 16;//    |


//		pack_reg_4bit |= (reg.y & 15) << 12;        //     > pack data
//		pack_reg_4bit |= ((reg.y >> 8) & 15) << 8;  //    |
//		pack_reg_4bit |= ((reg.y >> 16) & 15) << 4; //    |
//		pack_reg_4bit |= ((reg.y >> 24) & 15);      //----
//		uint32_t *packed_read_4bit_batch_addr = &(packed_read_4bit[i*n_threads]);
//		packed_read_4bit_batch_addr[tid] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory
//	}
//
//}
//
//__global__ void prepare_batch(uint32_t *read_batch, uint32_t *packed_read_batch, uint32_t *read_offsets, int min_seed_len, int n_tasks) {
//
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if (tid >= n_tasks) return;
//	int read_len = read_offsets[tid+1] - read_offsets[tid];
//	int read_len_16 = read_len%16 ? read_len + (16 - read_len%16) : read_len;
//	uint32_t BLOCKDIM = 128;
//	uint32_t N_BLOCKS = (n_tasks + BLOCKDIM - 1) / BLOCKDIM;
//	int read_tasks_per_thread = (int)ceil((double)read_len/(16*BLOCKDIM*N_BLOCKS));
//
//
//
//	return;
//
//}

//__global__ void pack_4bit_fow(uint32_t *read_batch, uint32_t* packed_read_batch_fow, uint32_t* thread_read_num, uint32_t *read_offsets, uint32_t* read_sizes, int n_tasks) {
//
//	int32_t i;
//	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index.
//	if (tid >= n_tasks) return;
//	uint32_t ascii_to_dna_reg_fow = 0x24031005;
//	uint32_t offset = (read_offsets[thread_read_num[tid]]%8 ?  read_offsets[thread_read_num[tid]] + (8 - read_offsets[thread_read_num[tid]]%8) : read_offsets[thread_read_num[tid]]) >> 3;
//	uint32_t len = read_sizes[thread_read_num[tid]]%8 ? read_sizes[thread_read_num[tid]] + (8 - read_sizes[thread_read_num[tid]]%8) : read_sizes[thread_read_num[tid]];
//	len = len >> 3;
//	uint32_t diff = tid - offset;
//	uint32_t *packed_read_batch_addr = &(read_batch[(tid<<1)]);
//	uint32_t reg1 = packed_read_batch_addr[0]; //load 4 bases of the first sequence from global memory
//	uint32_t reg2 = packed_read_batch_addr[1]; //load  another 4 bases of the S1 from global memory
//	uint32_t pack_reg_4bit = 0;
//	pack_reg_4bit |= ((ascii_to_dna_reg_fow >> (((reg2 >> 24) & 7) << 2))&15) << 28;        // ---
//	pack_reg_4bit |=  ((ascii_to_dna_reg_fow >> (((reg2 >> 16) & 7) << 2))&15) << 24; //    |
//	pack_reg_4bit |= ((ascii_to_dna_reg_fow >> (((reg2 >> 8) & 7) << 2))&15) << 20;//    |
//	pack_reg_4bit |= ((ascii_to_dna_reg_fow >> ((reg2 & 7) << 2))&15) << 16;//    |
//	pack_reg_4bit |=  ((ascii_to_dna_reg_fow >> (((reg1 >> 24) & 7) << 2))&15) << 12;        //     > pack data
//	pack_reg_4bit |=  ((ascii_to_dna_reg_fow >> (((reg1 >> 16) & 7) << 2))&15) << 8;  //    |
//	pack_reg_4bit |=  ((ascii_to_dna_reg_fow >> (((reg1 >> 8) & 7) << 2))&15) << 4; //    |
//	pack_reg_4bit |= ((ascii_to_dna_reg_fow >> ((reg1 & 7) << 2))&15);      //----
//	//uint32_t *packed_read_4bit_batch_addr = &(packed_read_batch_fow[offset]);
//	packed_read_batch_fow[offset + len - diff - 1] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory
//
//}


//__global__ void pack_4bit_rev(uint32_t read_batch, uint32_t* packed_read_batch_rev, int read_tasks_per_thread, uint32_t total_packed_read_regs) {
//
//	int32_t i;
//	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index.
//	uint32_t n_threads = gridDim.x * blockDim.x;
//	uint32_t ascii_to_dna_reg_rev = 0x14002035;
//	for (i = 0; i < read_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1) < total_packed_read_regs); ++i) {
//		uint32_t *packed_read_batch_addr = &(read_batch[(i*n_threads)<<1]);
//		uint32_t reg1 = packed_read_batch_addr[(tid << 2)]; //load 4 bases of the first sequence from global memory
//		uint32_t reg2 = packed_read_batch_addr[(tid << 2) + 1]; //load  another 4 bases of the S1 from global memory
//		uint32_t pack_reg_4bit = 0;
//		pack_reg_4bit |= (ascii_to_dna_reg_rev >> ((reg1 & 7) << 2))  << 28;        // ---
//		pack_reg_4bit |=  (ascii_to_dna_reg_rev >> ((reg1 >> 8) & 7) << 2) << 24; //    |
//		pack_reg_4bit |= (ascii_to_dna_reg_rev >> ((reg1 >> 16) & 7) << 2) << 20;//    |
//		pack_reg_4bit |= (ascii_to_dna_reg_rev >> ((reg1 >> 24) & 7) << 2) << 16;//    |
//		pack_reg_4bit |=  (ascii_to_dna_reg_rev >> (reg2 & 7) << 2) << 12 << 12;        //     > pack data
//		pack_reg_4bit |=  (ascii_to_dna_reg_rev >> ((reg2 >> 8) & 7) << 2) << 8;  //    |
//		pack_reg_4bit |=  (ascii_to_dna_reg_rev >> ((reg2 >> 16) & 7) << 2) << 4; //    |
//		pack_reg_4bit |= (ascii_to_dna_reg_rev >> ((reg2 >> 24) & 7) << 2);      //----
//		uint32_t *packed_read_4bit_batch_addr = &(packed_read_batch_rev[i*n_threads]);
//		packed_read_4bit_batch_addr[tid] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory
//	}
//
//}
//
//__global__ void pack_4bit_fow(uint32_t *read_batch, uint32_t* packed_read_batch_fow, int read_tasks_per_thread, uint32_t total_packed_read_regs, int offset, int len) {
//
//	int32_t i;
//	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index.
//	uint32_t n_threads = gridDim.x * blockDim.x;
//	uint32_t ascii_to_dna_reg_fow = 0x24031005;
//	for (i = 0; i < read_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1) < total_packed_read_regs); ++i) {
//		uint32_t *packed_read_batch_addr = &(read_batch[offset + (i*n_threads)<<1]);
//		uint32_t reg1 = packed_read_batch_addr[(tid << 1)]; //load 4 bases of the first sequence from global memory
//		uint32_t reg2 = packed_read_batch_addr[(tid << 1 ) + 1]; //load  another 4 bases of the S1 from global memory
//		uint32_t pack_reg_4bit = 0;
//		pack_reg_4bit |= (ascii_to_dna_reg_fow >> ((reg2 & 7) << 2));        // ---
//		pack_reg_4bit |=  (ascii_to_dna_reg_fow >> ((reg2 >> 8) & 7) << 2) << 4; //    |
//		pack_reg_4bit |= (ascii_to_dna_reg_fow >> ((reg2 >> 16) & 7) << 2) << 8;//    |
//		pack_reg_4bit |= (ascii_to_dna_reg_fow >> ((reg2 >> 24) & 7) << 2) << 12;//    |
//		pack_reg_4bit |=  (ascii_to_dna_reg_fow >> (reg1 & 7) << 2) << 12 << 16;        //     > pack data
//		pack_reg_4bit |=  (ascii_to_dna_reg_fow >> ((reg1 >> 8) & 7) << 2) << 20;  //    |
//		pack_reg_4bit |=  (ascii_to_dna_reg_fow >> ((reg1 >> 16) & 7) << 2) << 24; //    |
//		pack_reg_4bit |= (ascii_to_dna_reg_fow >> ((reg1 >> 24) & 7) << 2) << 28;      //----
//		uint32_t *packed_read_4bit_batch_addr = &(packed_read_batch_fow[(offset >> 2) + i*n_threads]);
//		packed_read_4bit_batch_addr[(len >> 2) - tid] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory
//	}
//
//}find_seed_intervals_gpu


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

//__global__ void ascii_to_dna(uint32_t *read_batch, int n_tasks){
//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
//	if  (tid >= n_tasks) return;
//	uint32_t ascii_reg = read_batch[tid];
//	uint32_t dna_reg = 0;
//	dna_reg |= ascii_to_dna_table[(ascii_reg & 7)] << 24;
//	dna_reg |= ascii_to_dna_table[((ascii_reg >> 8) & 7)] << 16;
//	dna_reg |= ascii_to_dna_table[((ascii_reg >> 16) & 7)] << 8;
//	dna_reg |= ascii_to_dna_table[((ascii_reg >> 24) & 7)];
//	read_batch[tid] = dna_reg;
//	return;
//}


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
//		cudaStream_t s;
//		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
//		int i;
//#pragma unroll
//		for (i = 0; i < read_len >> 3; i++){
//			thread_read_num[(read_offsets[tid] >> 3) + i] = tid;
//		}
//		assign_read_num<<<N_BLOCKS, BLOCKDIM, 0, s>>>(thread_read_num, read_offsets[tid] >> 3, tid, read_len >> 3);
//		cudaDeviceSynchronize();
	}
	else {
		int read_no = tid/max_read_length;
		int offset_in_read = tid - (read_no*max_read_length);
		//if (offset_in_read >= read_sizes[read_no] - min_seed_len) return;
		if (offset_in_read > read_sizes[read_no] - min_seed_len) return;
		thread_read_num[read_offsets[read_no] - (read_no*(min_seed_len-1)) + offset_in_read] = read_no;
		thread_read_idx[read_offsets[read_no] - (read_no*(min_seed_len-1)) + offset_in_read] = offset_in_read;
//		int i;
//		#pragma unroll
//		for (i = 0; i < read_sizes[tid] - min_seed_len; i++){
//			thread_read_num[read_offsets[tid] - (tid*min_seed_len) + i] = tid;
//			thread_read_idx[read_offsets[tid] - (tid*min_seed_len) + i] = i;
//		}
//		uint32_t BLOCKDIM = (read_len - min_seed_len) > 1024  ? 1024 : (read_len - min_seed_len);
//		uint32_t N_BLOCKS = ((read_len - min_seed_len) + BLOCKDIM - 1) / BLOCKDIM;
//		assign_threads_to_reads<<<N_BLOCKS, BLOCKDIM>>>(thread_read_num, thread_read_idx, read_offsets[tid] - (tid*min_seed_len), tid, (read_len - min_seed_len));
		//cudaDeviceSynchronize();
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
		uint2 intv = find_occ_gpu(bwt, l - 1, u, ch);
		//calculate the range
		//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
		//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
		l = L2_gpu[ch] + intv.x + 1;
		u = L2_gpu[ch] + intv.y;
		if (l > u) {
			break;
		}

	}

	pre_calc_intervals[tid] = make_uint2 (l, u);
	return;

}

__global__ void sum_arrays(uint32_t *in1, uint32_t *in2, uint32_t *out, int num_items) {
	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= num_items) return;

	out[tid] = in1[tid] + in2[tid];

}

void bwt_restore_sa_gpu(const char *fn, bwt_t_gpu *bwt)
{
	char skipped[256];
	FILE *fp;
	bwtint_t_gpu primary;

	fp = fopen(fn, "rb");
	if (fp == NULL){
		fprintf(stderr, "Unable to open .sa file.");
		exit(1);
	}
	//printf("file SA: %s\n",fn);
	fread(&primary, sizeof(bwtint_t_gpu), 1, fp);
	if (primary != bwt->primary){
		fprintf(stderr, "SA-BWT inconsistency: primary is not the same.\n");
		exit(EXIT_FAILURE);
	}
	fread(skipped, sizeof(bwtint_t_gpu), 4, fp); // skip
	fread(&bwt->sa_intv, sizeof(bwtint_t_gpu), 1, fp);
	fread(&primary, sizeof(bwtint_t_gpu), 1, fp);
	//printf("Primary: %d and seq_len: %d\n",primary,bwt->seq_len);
	if (primary != bwt->seq_len){
		fprintf(stderr, "SA-BWT inconsistency: seq_len is not the same.\n");
		exit(EXIT_FAILURE);
	}

	bwt->n_sa = (bwt->seq_len + bwt->sa_intv) / bwt->sa_intv;
	bwt->sa = (bwtint_t_gpu*)calloc(bwt->n_sa, sizeof(bwtint_t_gpu));
	bwt->sa[0] = -1;

	fread(bwt->sa + 1, sizeof(bwtint_t_gpu), bwt->n_sa - 1, fp);
	fclose(fp);
}

bwt_t_gpu *bwt_restore_bwt_gpu(const char *fn)
{
	bwt_t_gpu *bwt;
	FILE *fp;
	bwt = (bwt_t_gpu*)calloc(1, sizeof(bwt_t_gpu));
	fp = fopen(fn, "rb");
	//printf("file: %s\n",fn);
	if (fp == NULL){
		fprintf(stderr, "Unable to othread_read_numpen .bwt file.");
		exit(1);
	}
	fseek(fp, 0, SEEK_END);
	bwt->bwt_size = (ftell(fp) - sizeof(bwtint_t_gpu) * 5) >> 2;
	bwt->bwt = (uint32_t*)calloc(bwt->bwt_size, 4);
	//bwt->L2 = (bwtint_t_gpu*)calloc(5, 4);
	fseek(fp, 0, SEEK_SET);
	fread(&bwt->primary, sizeof(bwtint_t_gpu), 1, fp);
	fread(bwt->L2+1, sizeof(bwtint_t_gpu), 4, fp);
	fread(bwt->bwt, 4, bwt->bwt_size, fp);
	bwt->seq_len = bwt->L2[4];
	fclose(fp);

	return bwt;

}


void bwt_destroy_gpu(bwt_t_gpu *bwt)
{

	if (bwt == 0) return;
	free(bwt->sa); free(bwt->bwt);
	free(bwt);
}


void  print_seq_ascii(int length, char *seq){
   int i;
   //fprintf(stderr,"seq length = %d: ", length);
   for (i = 0; i < length; ++i) {
      putc(seq[i], stdout);
   }
   //fprintf(stderr,"\n");
}

void  print_seq_dna(int length, uint8_t* seq){
   int i;
   //fprintf(stderr,"seq length = %d: ", length);
   for (i = 0; i < length; ++i) {
      putc("ACGTN"[(int)seq[i]], stdout);
   }
   //fprintf(stderr,"\n");
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

int seed(int argc, char **argv, int u) {

	//printf("I go to seed %d\n",u);
	//printf("*** [GPU Wrapper] Seq: "); for (int j = 0; j < len; ++j) putchar("ACGTN"[(int)seq[j]]); putchar('\n');

    double total_time = realtime_gpu();
	bwt_t_gpu *bwt;
	int min_seed_size = 20;
	int pre_calc_seed_len = 13;
	int is_smem = 1;
	int print_out = 1;
	int print_stats = 0;
	int c;
	/*while ((c = getopt(argc, argv, "k:so")) >= 0) {
			switch (c) {
			case 'k':
				min_seed_size = atoi(optarg);
				break;
			case 's':
				is_smem = 1;
				break;
			case 'o':
				print_out = 1;
				break;

			}
	}*/

	// load index
	//printf("Argv1 %s Argv2 %s\n",argv[1],argv[2]);
	char *prefix = argv[1];
	fprintf(stderr, "Loading index from file and copying it to GPU...\n");
	char *str = (char*)calloc(strlen(prefix) + 10, 1);
	strcpy(str, prefix); strcat(str, ".gpu.bwt");  bwt = bwt_restore_bwt_gpu(str);
	free(str);
	str = (char*)calloc(strlen(prefix) + 10, 1);
	strcpy(str, prefix); strcat(str, ".gpu.sa");  bwt_restore_sa_gpu(str, bwt);
	free(str);
	uint32_t i;
	int count_0 = 0;
	//fprintf(stderr,"bwt_size=%llu\n",bwt->bwt_size);

    bwt_t_gpu bwt_gpu;

    double index_copy_time = realtime_gpu();
    gpuErrchk(cudaMalloc(&(bwt_gpu.bwt), bwt->bwt_size*sizeof(uint32_t)));
	gpuErrchk(cudaMemcpy(bwt_gpu.bwt, bwt->bwt, bwt->bwt_size*sizeof(uint32_t),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&(bwt_gpu.sa), bwt->n_sa*sizeof(bwtint_t_gpu)));
    gpuErrchk(cudaMemcpy(bwt_gpu.sa, bwt->sa, bwt->n_sa*sizeof(bwtint_t_gpu),cudaMemcpyHostToDevice));
    bwt_gpu.primary = bwt->primary;
    bwt_gpu.seq_len = bwt->seq_len;
    bwt_gpu.sa_intv = bwt->sa_intv;
    //fprintf(stderr, "SA intv %d\n", bwt->sa_intv);
    bwt_gpu.n_sa = bwt->n_sa;
    cudaMemcpyToSymbol(L2_gpu, bwt->L2, 5*sizeof(bwtint_t_gpu), 0, cudaMemcpyHostToDevice);

    /*fprintf(stderr, "primary=%llu\n", bwt->primary);
    fprintf(stderr, "seq_len=%llu\n", bwt->seq_len);
    fprintf(stderr, "L2[0]=%llu\n", bwt->L2[0]);
    fprintf(stderr, "L2[1]=%llu\n", bwt->L2[1]);
    fprintf(stderr, "L2[2]=%llu\n", bwt->L2[2]);
    fprintf(stderr, "L2[3]=%llu\n", bwt->L2[3]);
    fprintf(stderr, "L2[4]=%llu\n", bwt->L2[4]);*/

    if (print_stats)
		fprintf(stderr,"Index copied to GPU memory in %.3f seconds\n", realtime_gpu() - index_copy_time);
    
	if (print_stats)
		fprintf(stderr, "Pre-calculate intervals GPU...\n");
    
	double precalc_time = realtime_gpu();
    uint2 *pre_calc_seed_intervals;
    cudaMalloc(&pre_calc_seed_intervals, (1 << (pre_calc_seed_len<<1))*sizeof(uint2));
    int threads_per_block_pre_calc_seed = 128;
    int num_blocks_pre_calc_seed = ((1 << (pre_calc_seed_len<<1)) + threads_per_block_pre_calc_seed - 1)/threads_per_block_pre_calc_seed;
    pre_calc_seed_intervals_gpu<<<num_blocks_pre_calc_seed, threads_per_block_pre_calc_seed>>>(pre_calc_seed_intervals, pre_calc_seed_len, bwt_gpu, (1 << (pre_calc_seed_len<<1)));
    cudaDeviceSynchronize();
    
	if (print_stats)
		fprintf(stderr,"Pre-calculate intervals GPU in %.3f seconds\n", realtime_gpu() - precalc_time);

    fprintf(stderr, "\n-----------------------------------------------------------------------------------------------------------\n");
	FILE *read_file = fopen(argv[2], "r");

	// open reads file (fasta format) and copying reads into read buffer in a concatenated fashion.
	int all_done = 0;
	//char *all_reads = (char*)calloc(MAX_READ_LENGTH*READ_BATCH_SIZE,1);
	char *read_batch = (char*)malloc(BASE_BATCH_SIZE+1e6);
	uint32_t *read_offsets = (uint32_t*)calloc((1e6), sizeof(uint32_t));
	uint32_t *read_sizes = (uint32_t*)calloc((1e6), sizeof(uint32_t));
	double total_gpu_time = 0.0, total_batch_load_time=0.0, total_batch_prep_time = 0.0, total_mem_time = 0.0, total_print_time=0.0;
	double total_find_seed_intervals_time =0.0, total_filter_seed_intervals_time =0.0, total_locate_seeds_time=0.0;
	int reads_processed = 0;
	int max_read_size = 0;
	while (!all_done) {
		int total_reads = 0;
		int total_bytes = 0;
		int read_batch_size = 0;
		//int n_seed_cands = 0;
		//char *all_reads_fill_ptr = all_reads;
		char *read_batch_fill_ptr = read_batch;
		double loading_time = realtime_gpu();
		int prev_len = 0;
//		while (total_reads < READ_BATCH_SIZE) {
//			size_t len;
//			char *line = NULL;
//			int n_bases = getline(&line, &len, read_file);
//			if (n_bases < 0){
//				all_done = 1;
//				break;
//			}
//			if (n_bases > MAX_READ_LENGTH + 1) {
//				fprintf(stderr, "Read length exceeds MAX_READ_LENGTH");
//				exit(1);
//			}
//
//			if (line[0] != '>') {
//				//memcpy(all_reads_fill_ptr, line, n_bases);
//				memcpy(read_batch_fill_ptr, line, n_bases-1);
//				total_bytes = total_bytes + n_bases;
//				//all_reads_fill_ptr = all_reads_fill_ptr + n_bases;
//				read_batch_fill_ptr  += (n_bases - 1);
//				read_batch_size += (n_bases - 1);
//				//n_seed_cands += ((n_bases -1) - min_seed_size);
//				int read_len = n_bases - 1;
////				while (read_len%8) {
////					read_batch_fill_ptr[0] = 'P';
////					read_batch_fill_ptr++;
////					read_batch_size++;
////					read_len++;
////				}
//				read_offsets[total_reads] = total_reads == 0 ? 0 : read_offsets[total_reads - 1] + prev_len;
//				read_sizes[total_reads] = n_bases - 1;
//				prev_len = read_len;
//				total_reads++;
//				if ((n_bases - 1) > max_read_size) max_read_size = n_bases - 1;
//			}
//			free(line);
//		}
		while (read_batch_size < BASE_BATCH_SIZE) {
			size_t len;
			char *line = NULL;
			int n_bases = getline(&line, &len, read_file);
			if (n_bases < 0){
				all_done = 1;
				break;
			}
//			if (n_bases > MAX_READ_LENGTH + 1) {
//				fprintf(stderr, "Read length exceeds MAX_READ_LENGTH");
//				exit(1);
//			}

			if (line[0] != '>') {
				//memcpy(all_reads_fill_ptr, line, n_bases);
				memcpy(read_batch_fill_ptr, line, n_bases-1);
				total_bytes = total_bytes + n_bases;
				//all_reads_fill_ptr = all_reads_fill_ptr + n_bases;
				read_batch_fill_ptr  += (n_bases - 1);
				read_batch_size += (n_bases - 1);
				//n_seed_cands += ((n_bases -1) - min_seed_size);
				int read_len = n_bases - 1;
				//				while (read_len%8) {
				//					read_batch_fill_ptr[0] = 'P';
				//					read_batch_fill_ptr++;
				//					read_batch_size++;
				//					read_len++;
				//				}
				read_offsets[total_reads] = total_reads == 0 ? 0 : read_offsets[total_reads - 1] + prev_len;
				read_sizes[total_reads] = n_bases - 1;
				prev_len = read_len;
				total_reads++;
				if ((n_bases - 1) > max_read_size) max_read_size = n_bases - 1;
			}
			free(line);
		}
		int n_bits_max_read_size = (int)ceil(log2((double)max_read_size));
		total_batch_load_time += (realtime_gpu() - loading_time);
		
		if (print_stats)
			fprintf(stderr,"A batch of %d reads loaded from file in %.3f seconds\n", total_reads, realtime_gpu() - loading_time);

		int read_batch_size_8 = read_batch_size%8 ? read_batch_size + (8 - read_batch_size%8) : read_batch_size;
//		int all_reads_ptr = 0;
		uint8_t *read_batch_gpu;
		uint32_t *packed_read_batch_fow, *packed_read_batch_rev, *read_sizes_gpu, *read_offsets_gpu, *n_smems_fow, *n_smems_rev, *n_smems_fow_rev, *n_seeds_fow_rev;
		uint32_t *thread_read_num, *thread_read_idx;
		int2  *smem_intv_read_pos_fow_rev;
		uint32_t *n_smems_sum_fow_rev_gpu;
		uint32_t n_smems_sum_fow_rev;
		uint32_t *n_seeds_sum_fow_rev_gpu;
		uint32_t n_seeds_sum_fow_rev;

		uint32_t *n_smems_max_gpu;

		//uint32_t *n_seeds_sum_gpu;
		uint32_t *n_smems_fow_rev_scan;
		uint32_t *n_seeds_fow_rev_scan;

		void *cub_scan_temp_storage = NULL;
		size_t cub_scan_storage_bytes = 0;
		void *cub_select_temp_storage = NULL;
		size_t cub_select_storage_bytes = 0;
		void *cub_sort_temp_storage = NULL;
		size_t cub_sort_storage_bytes = 0;

		cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_fow_rev_scan, 2*(read_batch_size_8 - (total_reads*(min_seed_size-1))));
		
		if (print_stats)
			fprintf(stderr, "ExclusiveSum bytes for n_smems = %d\n", cub_scan_storage_bytes);

//		cub::DeviceReduce::Sum(cub_sum_temp_storage, cub_sum_storage_bytes, n_smems_fow, &n_smems_fow_sum, total_reads);
//		fprintf(stderr, "ExclusiveSum bytes for n_smems = %d\n", cub_sum_storage_bytes);

		int max_output_size = 2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads));

		max_output_size = max_output_size > (2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)) + (read_batch_size >> 3) + 2*total_reads + read_batch_size_8 >> 2 + (read_batch_size_8 - ((min_seed_size-1)*total_reads))) ? max_output_size : (2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)) + (read_batch_size >> 3) + 2*total_reads + read_batch_size_8 >> 2 + (read_batch_size_8 - ((min_seed_size-1)*total_reads)));

		cudaMalloc(&read_batch_gpu, read_batch_size_8);
		cudaMalloc(&read_sizes_gpu, total_reads*sizeof(uint32_t));
		cudaMalloc(&read_offsets_gpu,total_reads*sizeof(uint32_t));
		cudaMalloc(&n_smems_fow,total_reads*sizeof(uint32_t));
		cudaMalloc(&n_smems_rev,total_reads*sizeof(uint32_t));
		n_smems_fow_rev = read_sizes_gpu;
		cudaMalloc(&n_smems_fow_rev_scan,(total_reads+1)*sizeof(uint32_t));



		cudaMalloc(&packed_read_batch_fow,(read_batch_size_8 >> 3)*sizeof(uint32_t));
		cudaMalloc(&packed_read_batch_rev,(read_batch_size_8 >> 3)*sizeof(uint32_t));
		cudaMalloc(&n_seeds_fow_rev_scan, ((2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))) + 1)*sizeof(uint32_t));
		thread_read_num = n_seeds_fow_rev_scan;
		thread_read_idx = &n_seeds_fow_rev_scan[read_batch_size_8 - ((min_seed_size-1)*total_reads)];
		//cudaMalloc(&smem_intv_l_fow, (read_batch_size_8 - (min_seed_size*total_reads))*sizeof(uint32_t));
		//cudaMalloc(&smem_intv_l_rev, (read_batch_size_8 - (min_seed_size*total_reads))*sizeof(uint32_t));
		//cudaMalloc(&smem_intv_read_pos_fow, (read_batch_size_8 - (min_seed_size*total_reads))*sizeof(int2));
		//cudaMalloc(&smem_intv_read_pos_rev, (read_batch_size_8 - (min_seed_size*total_reads))*sizeof(int2));
		cudaMalloc(&cub_scan_temp_storage,cub_scan_storage_bytes);
		n_smems_sum_fow_rev_gpu = &n_smems_fow_rev_scan[total_reads];
		cudaMalloc(&n_seeds_sum_fow_rev_gpu, sizeof(uint32_t));
		cudaMalloc(&n_smems_max_gpu, 2*sizeof(uint32_t));

		uint2 *seed_intervals_fow_rev_gpu;
		int2 *seed_read_pos_fow_rev_gpu;
		uint2 *seed_intervals_fow_rev_compact_gpu;
		int2 *seed_read_pos_fow_rev_compact_gpu;
		uint32_t *is_smem_fow_rev_flag;
		uint32_t *n_ref_pos_fow_rev_gpu;
		uint32_t *n_ref_pos_fow_rev_scan;
		uint32_t *seed_ref_pos_fow_rev_gpu;
		uint32_t *seed_sa_idx_fow_rev_gpu;


		cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint32_t*)seed_intervals_fow_rev_gpu, (uint16_t*)is_smem_fow_rev_flag, (uint32_t*)seed_intervals_fow_rev_compact_gpu, n_smems_sum_fow_rev_gpu, 2*2*(read_batch_size_8 - (total_reads*(min_seed_size-1))));
		
		if (print_stats)
			fprintf(stderr, "Flagged bytes = %d\n", cub_select_storage_bytes);

		if (is_smem) {
			cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, (uint64_t*)seed_intervals_fow_rev_compact_gpu, (uint64_t*)seed_intervals_fow_rev_gpu,   2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)), total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size);
			if (print_stats)
				fprintf(stderr, "Sort bytes = %d\n", cub_sort_storage_bytes);
		} else {
			cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, seed_sa_idx_fow_rev_gpu, seed_ref_pos_fow_rev_gpu,  OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)), total_reads, n_ref_pos_fow_rev_scan, n_ref_pos_fow_rev_scan + 1, 0, n_bits_max_read_size);
			if (print_stats)
				fprintf(stderr, "Sort bytes = %d\n", cub_sort_storage_bytes);
		}


		uint2 *seed_intervals_pos_fow_rev_gpu;
		cudaMalloc(&seed_intervals_pos_fow_rev_gpu, OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint2));
		seed_read_pos_fow_rev_gpu = (int2*)seed_intervals_pos_fow_rev_gpu;
		seed_intervals_fow_rev_gpu = &seed_intervals_pos_fow_rev_gpu[2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))];
		uint2 *seed_intervals_pos_fow_rev_compact_gpu;
		cudaMalloc(&seed_intervals_pos_fow_rev_compact_gpu, OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint2));
		seed_read_pos_fow_rev_compact_gpu = (int2*)seed_intervals_pos_fow_rev_compact_gpu;
		seed_intervals_fow_rev_compact_gpu = &seed_intervals_pos_fow_rev_compact_gpu[2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))];
		//cudaMalloc(&seed_intervals_fow_rev_gpu, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint2));
		//cudaMalloc(&seed_read_pos_fow_rev_gpu, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(int2));
//		cudaMalloc(&seed_intervals_fow_rev_compact_gpu, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint2));
//		cudaMalloc(&seed_read_pos_fow_rev_compact_gpu, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(int2));

		uint32_t *n_seeds_is_smem_flag_fow_rev;
		cudaMalloc(&n_seeds_is_smem_flag_fow_rev, OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint32_t));
		n_seeds_fow_rev = n_seeds_is_smem_flag_fow_rev;
		is_smem_fow_rev_flag = &n_seeds_is_smem_flag_fow_rev[2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))];

		if(!is_smem) cudaMalloc(&seed_ref_pos_fow_rev_gpu, OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint32_t));


//		cudaMalloc(&n_seeds_fow_rev, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint32_t));
//
//		cudaMalloc(&is_smem_fow_rev_flag, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint32_t));
		cudaMalloc(&cub_select_temp_storage,cub_select_storage_bytes);
		cudaMalloc(&cub_sort_temp_storage,cub_sort_storage_bytes);
		n_ref_pos_fow_rev_gpu = read_offsets_gpu;


		//cudaMalloc(&n_seeds_sum_gpu, 2*sizeof(uint32_t));
		double gpu_batch_time = realtime_gpu();
		double mem_time0 = realtime_gpu();
		cudaMemcpy(read_batch_gpu, (uint8_t*)read_batch, read_batch_size, cudaMemcpyHostToDevice);
		cudaMemcpy(read_offsets_gpu, read_offsets, total_reads*sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(read_sizes_gpu, read_sizes, total_reads*sizeof(uint32_t), cudaMemcpyHostToDevice);
		mem_time0 = realtime_gpu() - mem_time0;
		double batch_prep_time = realtime_gpu();
		int BLOCKDIM = 128;
		double rev_pack_time_start = realtime_gpu();
		int N_BLOCKS = ((read_batch_size_8 >> 3)  + BLOCKDIM - 1)/BLOCKDIM;
		pack_4bit_rev<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)read_batch_gpu, packed_read_batch_rev, (read_batch_size_8 >> 3));
		cudaDeviceSynchronize();
		double rev_pack_time = realtime_gpu() - rev_pack_time_start;
		double assign_threads_for_fow_pack_time_start = realtime_gpu();
		//N_BLOCKS = (total_reads + BLOCKDIM - 1)/BLOCKDIM;
		//prepare_batch<<<BLOCKDIM, N_BLOCKS>>>(thread_read_num, thread_read_idx, read_offsets_gpu, read_sizes_gpu, min_seed_size, total_reads, 1);
		cudaDeviceSynchronize();
		double assign_threads_for_fow_pack_time = realtime_gpu() - assign_threads_for_fow_pack_time_start;
//		uint32_t *thread_read_num_cpu = (uint32_t*)calloc((read_batch_size_8 >> 3), sizeof(uint32_t));
//		cudaMemcpy(thread_read_num_cpu, thread_read_num, (read_batch_size_8 >> 3)*sizeof(uint32_tOUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))), cudaMemcpyDeviceToHost);
//		int k;
//		for (k = 0; k < (read_batch_size_8 >> 3); k++) {
//			fprintf(stderr, "%d, ",thread_read_num_cpu[k]);
//		}
//		fprintf(stderr, "\n");
		double fow_pack_time_start = realtime_gpu();
		N_BLOCKS = ((read_batch_size_8 >> 3)  + BLOCKDIM - 1)/BLOCKDIM;
		pack_4bit_fow<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)read_batch_gpu, packed_read_batch_fow, /*thread_read_num, read_offsets_gpu, read_sizes_gpu,*/ (read_batch_size_8 >> 3));
		cudaDeviceSynchronize();
		double fow_pack_time = realtime_gpu() - fow_pack_time_start;
		double assign_threads_time_start = realtime_gpu();
		N_BLOCKS = ((total_reads*max_read_size) + BLOCKDIM - 1)/BLOCKDIM;
		prepare_batch<<<N_BLOCKS, BLOCKDIM>>>(thread_read_num, thread_read_idx, read_offsets_gpu, read_sizes_gpu, min_seed_size, total_reads*max_read_size, 0, max_read_size);
		cudaDeviceSynchronize();
		double assign_threads_time = realtime_gpu() - assign_threads_time_start;
//		uint32_t *thread_read_num_cpu = (uint32_t*)calloc((MAX_READ_LENGTH - min_seed_size)*total_reads, sizeof(uint32_t));
//		cudaMemcpy(thread_read_num_cpu, thread_read_num, (MAX_READ_LENGTH - min_seed_size)*total_reads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//		int k;
//		for (k = 0; k < (MAX_READ_LENGTH - min_seed_size)*total_reads; k++) {
//			fprintf(stderr, "%d, ",thread_read_num_cpu[k]);
//		}
//		fprintf(stderr, "\n");
//		uint32_t *thread_read_idx_cpu = (uint32_t*)calloc((MAX_READ_LENGTH - min_seed_size)*total_reads, sizeof(uint32_t));
//		cudaMemcpy(thread_read_idx_cpu, thread_read_idx, (MAX_READ_LENGTH - min_seed_size)*total_reads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//		for (k = 0; k < (MAX_READ_LENGTH - min_seed_size)*total_reads; k++) {
//			fprintf(stderr, "%d, ",thread_read_idx_cpu[k]);smem_intv_l_fow
//		}
//		fprintf(stderr, "\n");
//
//		int *read_offsets = (int*)calloc((2*actual_read_batch_size), sizeof(int));
//		ushort *read_num = (ushort*)calloc((MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size), sizeof(ushort));
//		ushort *read_idx = (ushort*)calloc((MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size), sizeof(ushort));
//		uint8_t *read_batch = (uint8_t*)malloc(MAX_READ_LENGTH*(2*actual_read_batch_size));
//
//		int read_batch_ptr = 0;
//		int i;
//		int read_batch_size = 0;
//		int size_idx = 0;
//		int offset_idx = 0;
//		ushort read_no = 0;
//		int read_num_idx = 0, n_tasks = 0;
//		for (i = 0; i < actual_read_batch_size; i++) {
//			int curr_read_size = 0;
//			while (all_reads[all_reads_ptr] != '\n') {
//				char base = all_reads[all_reads_ptr];
//				read_batch[read_batch_ptr] = seq_nt4_table [(int)base];
//				curr_read_size++;
//				read_batch_ptr++;
//				all_reads_ptr++;
//				read_batch_size++;
//			}
//			int k;
//			for(k = 0; k < curr_read_size - min_seed_size; k++ ){
//				read_num[read_num_idx] = read_no;
//				read_idx[read_num_idx] = k;
//				read_num_idx++;
//				n_tasks++;
//			}
//			read_no++;
//			int l;
//			for (l = 0; l < curr_read_size; l++) {
//				if (read_batch[read_batch_ptr - l - 1] < 4) read_batch[read_batch_ptr + l] = 3 - read_batch[read_batch_ptr - l - 1];
//				else read_batch[read_batch_ptr + l] = 4;
//				read_batch_size++;
//				//read_batch_ptr++;
//			}
//			for(k = 0; k < curr_read_size - min_seed_size; k++ ){
//				read_num[read_num_idx] = read_no;
//				read_idx[read_num_idx] = k;
//				read_num_idx++;
//				n_tasks++;
//			}
//			read_no++;
//			read_sizes[size_idx] = curr_read_size;
//			size_idx++;
//			read_sizes[size_idx] = curr_read_size;
//			size_idx++;
//			if (offset_idx + 1 < (2*actual_read_batch_size)) {
//				read_offsets[offset_idx + 1] = read_offsets[offset_idx] + curr_read_size;
//				offset_idx++;
//			}
//			if (offset_idx + 1 < (2*actual_read_batch_size)) {
//				read_offsets[offset_idx + 1] = read_offsets[offset_idx] + curr_read_size;
//				offset_idx++;
//			}
//			all_reads_ptr++;
//			read_batch_ptr += curr_read_size;
//		}
		total_batch_prep_time += (realtime_gpu() - batch_prep_time);

		if (print_stats) {
			fprintf(stderr,"Batch prepared for processing on GPU in %.3f seconds\n", realtime_gpu() - batch_prep_time);
			fprintf(stderr,"\tReverse batch packed on GPU in %.3f seconds\n", rev_pack_time);
			fprintf(stderr,"\tThread assignment for forward batch packing on GPU in %.3f seconds\n", assign_threads_for_fow_pack_time);
			fprintf(stderr,"\tForward batch packed on GPU in %.3f seconds\n",fow_pack_time);
			fprintf(stderr,"\tThread assignment for computing seed intervals on GPU in %.3f seconds\n",assign_threads_time);
			fprintf(stderr, "Processing %d reads on GPU...\n", total_reads);
		}

//		uint2 *seed_intervals_fow = (uint2*)calloc((MAX_READ_LENGTH - min_seed_size)*total_reads, sizeof(uint2));
//		uint2 *seed_intervals_rev = (uint2*)calloc((MAX_READ_LENGTH - min_seed_size)*total_reads, sizeof(uint2));
//		int2 *seed_read_pos_fow = (int2*)calloc((MAX_READ_LENGTH - min_seed_size)*total_reads, sizeof(int2));
//		int2 *seed_read_pos_rev = (int2*)calloc((MAX_READ_LENGTH - min_seed_size)*total_reads, sizeof(int2));

		//cudaMalloc(&n_seeds_fow_scan, (MAX_READ_LENGTH - min_seed_size)*(total_reads)*sizeof(uint32_t));
		//cudaMalloc(&n_seeds_rev_scan, (MAX_READ_LENGTH - min_seed_size)*(total_reads)*sizeof(uint32_t));
//		cudaMalloc(&seed_intervals_u_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(bwtint_t_gpu));
//		cudaMalloc(&seed_read_begin_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(int));
//		cudaMalloc(&seed_read_end_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(int));
//		bwtint_t_gpu *seed_intervals_l = (bwtint_t_gpu*)calloc((MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size), sizeof(bwtint_t_gpu));
//		int *seed_read_begin = (int*)malloc((MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(int));
//		int *seed_read_end = (int*)malloc((MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(int));
//		int *num_seeds = (int*)malloc((2*actual_read_batch_size)*sizeof(int));
//		int *seed_ref_begin = (int*)malloc(MAX_SEEDS_PER_READ*count_seed_intervals_gpu(2*actual_read_batch_size)*sizeof(bwtint_t_gpu));
		//fprintf(stderr, "Processing %d reads on GPU\n", actual_read_batch_siz//__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow, uint2 *seed_intervals_rev,
		//		int2 *seed_read_pos_fow, int2 *seed_read_pos_rev, uint32_t *read_num, uint32_t *read_idx, /*uint2* pre_calc_intervals,*/ int min_seed_size, bwt_t_gpu bwt, /*int pre_calc_seed_len,*/ int n_tasks) {
		//
		//	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
		//	if (tid >= 2*n_tasks) return;
		//	int thread_read_num = read_num[tid%n_tasks];
		//	int read_len = read_sizes[thread_read_num];
		//	int read_off = read_offsets[thread_read_num];
		//	int intv_idx = thread_read_num * (MAX_READ_LENGTH - min_seed_size);
		//	uint32_t thread_read_idx = read_idx[tid%n_tasks];
		//
		//
		//	int i;
		//	bwtint_t_gpu prev_l = 0, prev_u = bwt.seq_len;
		////	uint32_t pre_calc_seed = 0;
		////	for (i = 0; i < pre_calc_seed_len; i++, thread_read_idx++) {
		////		int reg_no = thread_read_idx >> 3;
		////		int reg_pos = thread_read_idx & 7;
		////		int reg = read_batch_gpu[read_off + reg_no];
		////		int base = reg >> (28 - (reg_pos << 2));
		////		if (base > 3) {
		////			thread_read_idx = MAX_READ_LENGTH;
		////			break;
		////		}
		////		pre_calc_seed |= (base << (i << 1));gzrewind
		////	}
		////	uint2 start_seed_intervals = make_uint2(0,0);
		////	if (thread_read_idx < MAX_READ_LENGTH) {
		////		uint2 start_seed_intervals = pre_calc_intervals[pre_calc_seed];
		////	}
		////	prev_l = start_seed_intervals.x;
		////	prev_u = start_seed_intervals.y;
		//	int base;
		//	bwtint_t_gpu l, u;
		//	int donot_add = 0;
		//	if (tid < n_tasks) {
		//		int start = read_off&7;
		//		uint32_t *seq = &(packed_read_batch_fow[read_off >> 3]);
		//		for (i = start + read_len - thread_read_idx - 1 ; i >= start; i--) {
		//			/*get the base*/
		//			int reg_no = i >> 3;
		//			int reg_pos = i & 7;
		//			int reg = seq[reg_no];
		//			int base = (reg >> (28 - (reg_pos << 2)))&15;
		//			/*unknown bases*/
		//			if (base > 3) {
		//				break;
		//			}
		//
		//			uint2 intv = find_occ_gpu(bwt, prev_l - 1, prev_u, base);
		//			//calculate the range
		//			//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
		//			//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
		//			l = L2_gpu[base] + intv.x + 1;
		//			u = L2_gpu[base] + intv.y;
		//			if (l > u) {
		//				break;
		//			}
		//
		//			prev_l = l;
		//			prev_u = u;
		//		}
		//		seed_intervals_fow[intv_idx + thread_read_idx] = make_uint2(prev_l, prev_u);
		//		seed_read_pos_fow[intv_idx + thread_read_idx] = make_int2 (thread_read_idx, read_len + start - i) ;
		//	}
		//	else {
		//		int start = read_off&7;
		//		uint32_t *seq = &(packed_read_batch_rev[read_off >> 3]);
		//		for (i = start + thread_read_idx ; i < read_len + start; i++) {
		//			/*get the base*/
		//			int reg_no = i >> 3;
		//			int reg_pos = i & 7;
		//			int reg = seq[reg_no];
		//			int base = (reg >> (28 - (reg_pos << 2)))&15;
		//			/*unknown bases*/
		//			if (base > 3) {
		//				break;
		//			}
		//
		//			uint2 intv = find_occ_gpu(bwt, prev_l - 1, prev_u, base);
		//			//calculate the rangei
		//			//l = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_l - 1, ch) + 1;
		//			//u = L2_gpu[ch] + bwt_occ_gpu(bwt, prev_u, ch);
		//			l = L2_gpu[base] + intv.x + 1;
		//			u = L2_gpu[base] + intv.y;
		//			if (l > u) {
		//				break;
		//			}
		//
		//			prev_l = l;
		//			prev_u = u;
		//		}
		//		seed_intervals_rev[intv_idx + thread_read_idx] = make_uint2(prev_l, prev_u);
		//		seed_read_pos_rev[intv_idx + thread_read_idx] = make_int2 (thread_read_idx, i - start) ;
		//	}
		//
		//	return;
		//
		//}e);
		//fflush(stderr);

//
//		bwtint_t_gpu *seed_intervals_l_gpu,  *seed_intervals_u_gpu, *seed_ref_begin_gpu;
//		int *seed_read_begin_gpu, *seed_read_end_gpu, *num_seeds_gpu;
//
//		cudaMalloc(&seed_intervals_l_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(bwtint_t_gpu));
//		cudaMalloc(&seed_intervals_u_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(bwtint_t_gpu));
//		cudaMalloc(&seed_read_begin_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(int));
//		cudaMalloc(&seed_read_end_gpu, (MAX_READ_LENGTH - min_seed_size)*(2*actual_read_batch_size)*sizeof(int));
//		cudaMalloc(&num_seeds_gpu, (2*actual_read_batch_size)*sizeof(int));
//		cudaMalloc(&seed_ref_begin_gpu, MAX_SEEDS_PER_READ*(2*actual_read_batch_size)*sizeof(bwtint_t_gpu));

		//cudaMemcpy(read_num_gpu, read_num, n_tasks*sizeof(ushort), cudaMemcpyHostToDevice);
		//cudaMemcpy(read_idx_gpu, read_idxtotal_n_ref_pos_rev, n_tasks*sizeof(ushort), cudaMemcpyHostToDevice);


		double find_seeds_time = realtime_gpu();
		int n_seed_cands = read_batch_size - (total_reads*(min_seed_size-1));
		cudaMemset(is_smem_fow_rev_flag, 0, 2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint32_t));
		cudaMemset(n_smems_fow, 0, total_reads*sizeof(uint32_t));
		cudaMemset(n_smems_rev, 0, total_reads*sizeof(uint32_t));

		N_BLOCKS = ((2*n_seed_cands) + BLOCKDIM - 1)/BLOCKDIM;
		find_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(packed_read_batch_fow, packed_read_batch_rev,  read_sizes_gpu, read_offsets_gpu, seed_intervals_fow_rev_gpu,
				seed_read_pos_fow_rev_gpu, thread_read_num, thread_read_idx, is_smem_fow_rev_flag, pre_calc_seed_intervals,   n_smems_fow, n_smems_rev, min_seed_size,bwt_gpu, pre_calc_seed_len,  n_seed_cands);
		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tIntervals of SMEM seeds computed in %.3f seconds on GPU\n", realtime_gpu() - find_seeds_time);
		
		total_find_seed_intervals_time += realtime_gpu() - find_seeds_time;

		double n_smems_fow_max_time = realtime_gpu();
		cub::DeviceReduce::Max(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems_fow, &n_smems_max_gpu[0], total_reads);
		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tMax in n_smems_fow found in %.3f seconds\n", realtime_gpu() - n_smems_fow_max_time);
		
		double n_smems_rev_max_time = realtime_gpu();
		cub::DeviceReduce::Max(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems_rev, &n_smems_max_gpu[1], total_reads);
		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tMax in n_smems_rev found in %.3f seconds\n", realtime_gpu() - n_smems_rev_max_time);



//




		double filter_seeds_time = realtime_gpu();
		cudaError_t err = cudaSuccess;
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint32_t*)seed_intervals_fow_rev_gpu, (uint16_t*)is_smem_fow_rev_flag, (uint32_t*)seed_intervals_fow_rev_compact_gpu, n_smems_sum_fow_rev_gpu, 2*2*(read_batch_size_8 - (total_reads*(min_seed_size-1)))));
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (int32_t*)seed_read_pos_fow_rev_gpu, (uint16_t*)is_smem_fow_rev_flag, (int32_t*)seed_read_pos_fow_rev_compact_gpu, n_smems_sum_fow_rev_gpu, 2*2*(read_batch_size_8 - (total_reads*(min_seed_size-1)))));



		N_BLOCKS = (total_reads + BLOCKDIM - 1)/BLOCKDIM;

		sum_arrays<<<N_BLOCKS, BLOCKDIM>>>(n_smems_fow, n_smems_rev, n_smems_fow_rev, total_reads);

		double n_smems_fow_rev_scan_time = realtime_gpu();
		cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems_fow_rev, n_smems_fow_rev_scan, total_reads);
		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tn_smems_fow_rev scanned in %.3f seconds on GPU\n", realtime_gpu() - n_smems_fow_rev_scan_time);



		cudaMemset(n_ref_pos_fow_rev_gpu, 0, total_reads*sizeof(uint32_t));

		//filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_compact_gpu, seed_intervals_rev_compact_gpu, seed_read_pos_fow_compact_gpu, seed_read_pos_rev_compact_gpu, n_smems_fow, n_smems_rev,  n_smems_fow_scan, n_smems_rev_scan, n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads);
		if (is_smem) filter_seed_intervals_gpu_wrapper<<<1, 1>>>(seed_intervals_fow_rev_compact_gpu, seed_read_pos_fow_rev_compact_gpu, seed_intervals_fow_rev_gpu, seed_read_pos_fow_rev_gpu,   n_smems_fow, n_smems_rev, n_smems_fow_rev,  n_seeds_fow_rev, n_smems_fow_rev_scan,  n_ref_pos_fow_rev_gpu, n_smems_max_gpu, n_smems_sum_fow_rev_gpu, cub_sort_temp_storage, cub_sort_storage_bytes, total_reads, n_bits_max_read_size, is_smem/*n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads*/);
		else {
			filter_seed_intervals_gpu_wrapper_mem<<<1, 1>>>(seed_intervals_fow_rev_compact_gpu, seed_read_pos_fow_rev_compact_gpu, seed_intervals_fow_rev_gpu, n_smems_fow, n_smems_rev, n_smems_fow_rev,  n_seeds_fow_rev,  n_smems_fow_rev_scan, n_ref_pos_fow_rev_gpu, n_smems_max_gpu, n_smems_sum_fow_rev_gpu, total_reads/*n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads*/);

			int2 *swap =  seed_read_pos_fow_rev_compact_gpu;
			seed_read_pos_fow_rev_compact_gpu = seed_read_pos_fow_rev_gpu;
			seed_read_pos_fow_rev_gpu = swap;
//			uint2 *swap2 =  seed_intervals_fow_rev_compact_gpu;
//			seed_intervals_fow_rev_compact_gpu = seed_intervals_fow_rev_gpu;
//			seed_intervals_fow_rev_gpu = swap2;
		}

		cudaDeviceSynchronize();




		//double read_pos_sort_time = realtime_gpu();

		cudaMemcpy(&n_smems_sum_fow_rev, n_smems_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);


//		cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, (uint64_t*)seed_intervals_fow_rev_compact_gpu, (uint64_t*)seed_intervals_fow_rev_gpu,  n_smems_sum_fow_rev, total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size);
//		cudaDeviceSynchronize();
//		fprintf(stderr,"\tread_pos sorted in %.3f seconds\n", realtime_gpu() - read_pos_sort_time);

		
		if (print_stats)
			fprintf(stderr,"\tSMEM seeds filtered in %.3f seconds on GPU\n", realtime_gpu() - n_smems_fow_max_time);
		
		total_filter_seed_intervals_time += realtime_gpu() - n_smems_fow_max_time;

		double locate_seeds_time = realtime_gpu();

//		n_smems_sum[0]=n_smems_sum[0]/2;
//		n_smems_sum[1]=n_smems_sum[1]/2;

		//count_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_compact_gpu, seed_intervals_rev_compact_gpu, n_smems_fow, n_smems_rev, n_seeds_fow, n_seeds_rev,  n_smems_fow_scan, n_smems_rev_scan,  n_ref_pos_fow_gpu, n_ref_pos_rev_gpu, n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads);
		//cudaDeviceSynchronize();
		//fprintf(stderr,"\tNumber of seeds counted in %.3f seconds on GPU\n", realtime_gpu() - count_seeds_time);

		double n_seeds_fow_rev_sum_time = realtime_gpu();
		cub::DeviceReduce::Sum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_sum_fow_rev_gpu, n_smems_sum_fow_rev);
		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tn_seeds_fow_rev summed in %.3f seconds\n", realtime_gpu() - n_seeds_fow_rev_sum_time);
//
//		double n_seeds_rev_sum_time = realtime_gpu();
//		cub::DeviceReduce::Sum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_rev,  &n_seeds_sum_gpu[1], n_smems_sum[1]);
//		cudaDeviceSynchronize();
//		fprintf(stderr,"\tn_seeds_rev summed in %.3f seconds\n", realtime_gpu() - n_seeds_rev_sum_time);

		double n_seeds_fow_rev_scan_time = realtime_gpu();
		cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_fow_rev_scan, n_smems_sum_fow_rev);
		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tn_seeds_fow_rev scanned in %.3f seconds on GPU\n", realtime_gpu() - n_seeds_fow_rev_scan_time);


		seed_sa_idx_fow_rev_gpu = n_seeds_fow_rev;


		//N_BLOCKS = ((n_seeds_sum[0] + n_seeds_sum[1])  + BLOCKDIM - 1)/BLOCKDIM;

		//locate_seeds_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_ref_pos_fow_gpu, seed_ref_pos_rev_gpu, bwt_gpu, n_seeds_sum[0], n_seeds_sum[1]);

		int2 *final_seed_read_pos_fow_rev_gpu;


		cudaMemcpy(&n_seeds_sum_fow_rev, n_seeds_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		//printf("N_seeds_sum_fow_rev %ld\n",n_seeds_sum_fow_rev);

		if(n_seeds_sum_fow_rev > OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))) {
			fprintf(stderr,"n_seeds_sum_fow_rev (%llu) is more than allocated size(%d)\n", n_seeds_sum_fow_rev, OUTPUT_SIZE_MUL*2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)));
			exit(EXIT_FAILURE);
		}

		if (is_smem) {
			locate_seeds_gpu_wrapper<<<1, 1>>>(seed_read_pos_fow_rev_compact_gpu, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev_gpu, seed_read_pos_fow_rev_gpu, n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, bwt_gpu);
			 final_seed_read_pos_fow_rev_gpu = seed_read_pos_fow_rev_compact_gpu;
			 seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;
		}
		else {
			locate_seeds_gpu_wrapper_mem<<<1, 1>>>(seed_read_pos_fow_rev_compact_gpu, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev_gpu, seed_read_pos_fow_rev_gpu, n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, n_ref_pos_fow_rev_gpu, bwt_gpu);
			n_ref_pos_fow_rev_scan = n_smems_fow_rev_scan;
			cudaMemcpy(n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToDevice);
			cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_ref_pos_fow_rev_gpu, n_ref_pos_fow_rev_scan, total_reads);
			cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, seed_sa_idx_fow_rev_gpu, seed_ref_pos_fow_rev_gpu,  n_seeds_sum_fow_rev, total_reads, n_ref_pos_fow_rev_scan, n_ref_pos_fow_rev_scan + 1, 0, n_bits_max_read_size);
			final_seed_read_pos_fow_rev_gpu = seed_read_pos_fow_rev_gpu;
		}

		cudaDeviceSynchronize();
		
		if (print_stats)
			fprintf(stderr,"\tSeeds located on ref in %.3f seconds\n", realtime_gpu() - locate_seeds_time);
		
		total_locate_seeds_time += realtime_gpu() - locate_seeds_time;


		//cudaMemcpy(&n_seeds_sum[1], &n_seeds_rev_scan[n_smems_sum[1]], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (print_stats)
			fprintf(stderr, "n_seed_sum_fow_rev = %d, n_smem_sum_fow = %d\n", n_seeds_sum_fow_rev, n_smems_sum_fow_rev);
		
		fflush(stderr);

		double mem_time1 = realtime_gpu();
		int2 *seed_read_pos_fow_rev = (int2*)calloc(n_seeds_sum_fow_rev, sizeof(int2));
		uint32_t *seed_ref_pos_fow_rev = (uint32_t*)calloc(n_seeds_sum_fow_rev, sizeof(uint32_t));
		uint32_t *n_ref_pos_fow_rev = (uint32_t*)calloc(total_reads, sizeof(uint32_t));


		/*if (is_smem)*/
		//else final_seed_read_pos_fow_rev_gpu = seed_read_pos_fow_rev_gpu;


		cudaMemcpy(seed_ref_pos_fow_rev, seed_ref_pos_fow_rev_gpu, n_seeds_sum_fow_rev*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(seed_read_pos_fow_rev, final_seed_read_pos_fow_rev_gpu, n_seeds_sum_fow_rev*sizeof(int2), cudaMemcpyDeviceToHost);

//				cudaMemcpy(seed_read_pos_fow, smem_intv_read_pos_fow, n_seeds_sum[0]*sizeof(int2), cudaMemcpyDeviceToHost);
//				cudaMemcpy(seed_read_pos_rev, smem_intv_read_pos_rev, n_seeds_sum[1]*sizeof(int2), cudaMemcpyDeviceToHost);
		cudaMemcpy(n_ref_pos_fow_rev, n_ref_pos_fow_rev_gpu, total_reads*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		mem_time1 = (realtime_gpu() - mem_time1);
		
		if (print_stats)
			fprintf(stderr,"\tTime spent in cudaMemcpy is %.3f seconds\n", mem_time0 + mem_time1);
		total_mem_time += mem_time0 + mem_time1;

		if (print_stats)
			fprintf(stderr, "Total processing time of the batch on GPU is %.3f seconds\n", realtime_gpu() - gpu_batch_time);
		total_gpu_time += (realtime_gpu() - gpu_batch_time);

		double print_time = realtime_gpu();

		if(print_out){
			int i, j , k;
			int total_n_ref_pos_fow_rev = 0;
			char sign[2] = {'+', '-'};
			uint32_t seed_pos;
			for (i = 0, j = 0, k = 0; i < total_reads; i++) {
				int y;
				total_n_ref_pos_fow_rev += n_ref_pos_fow_rev[i];
				if (is_smem) fprintf(stdout, "/*===================================SMEM seeds in read no. %d (Read[begin, End] -> starting position(s) on the reference)===================================*/\n", reads_processed + i  + 1);
				else fprintf(stdout, "/*===================================MEM seeds in read no. %d (Read[begin, End] -> starting position(s) on the reference)===================================*/\n", reads_processed + i  + 1);
				int prev_seed_begin = -1, prev_seed_end = -1;
				for (y = 0;  y < n_ref_pos_fow_rev[i] && j < n_seeds_sum_fow_rev; j++, y++) {
					//printf("\nN_ref_pos_fow_rev: %lu\n",n_seeds_fow_rev[i]);
					if (prev_seed_begin == seed_read_pos_fow_rev[j].x && prev_seed_end == (seed_read_pos_fow_rev[j].y)) {
						fprintf(stdout,", %c%llu", sign[((uint32_t)seed_read_pos_fow_rev[j].y) >> 31],seed_ref_pos_fow_rev[j]);
					}
					else {
						fprintf(stdout,"\n");
						if (((uint32_t)seed_read_pos_fow_rev[j].y) >> 31 == 1){
							seed_pos = 2 * bwt->seq_len - seed_ref_pos_fow_rev[j] - ((seed_read_pos_fow_rev[j].y << 1 >> 1) - (seed_read_pos_fow_rev[j].x << 1 >> 1));
							//printf("Seq_len: %llu and seed_pos: %llu and final %llu\n", 2*bwt->seq_len,seed_ref_pos_fow_rev[j], seed_pos);
							fprintf(stdout, "[GPUSeed] Read[%d, %d] -> %llu", seed_read_pos_fow_rev[j].x << 1 >> 1, (seed_read_pos_fow_rev[j].y << 1 >> 1), seed_pos);
						}
						else {
							fprintf(stdout, "[GPUSeed] Read[%d, %d] -> %llu", seed_read_pos_fow_rev[j].x << 1 >> 1, (seed_read_pos_fow_rev[j].y << 1 >> 1), seed_ref_pos_fow_rev[j]);
						}
					}
					prev_seed_begin = seed_read_pos_fow_rev[j].x;
					prev_seed_end = (seed_read_pos_fow_rev[j].y);
				}
				fprintf(stdout, "\n");
			}
			if (print_stats)
				fprintf(stderr, "total_n_ref_pos_fow = %d, n_seed_sum_fow = %d, n_smem_sum_fow = %d\n",total_n_ref_pos_fow_rev, n_seeds_sum_fow_rev, n_smems_sum_fow_rev);

		}
		else {
			for (int i=0; i<total_reads; i++) printf("\nN_ref_pos_fow_rev: %lu\n",n_ref_pos_fow_rev[i]);
		}

		reads_processed = reads_processed + total_reads;
		cudaFree(read_batch_gpu); cudaFree(read_sizes_gpu); cudaFree(read_offsets_gpu);
		//cudaFree(seed_intervals_fow_rev_gpu);
		//cudaFree(seed_read_pos_fow_rev_gpu);
		//cudaFree(seed_intervals_fow_rev_compact_gpu);
		//cudaFree(seed_read_pos_fow_rev_compact_gpu);
		cudaFree(seed_intervals_pos_fow_rev_gpu);
		cudaFree(seed_intervals_pos_fow_rev_compact_gpu);
		cudaFree(packed_read_batch_fow);cudaFree(packed_read_batch_rev);
		cudaFree(n_smems_fow); cudaFree(n_smems_rev); cudaFree(n_smems_fow_rev_scan);
		//cudaFree(n_smems_fow_rev);
		//cudaFree(n_seeds_fow_rev);
		//cudaFree(smem_intv_l_fow);cudaFree(smem_intv_l_rev); cudaFree(smem_intv_read_pos_fow); cudaFree(smem_intv_read_pos_rev);
		cudaFree(n_seeds_sum_fow_rev_gpu);
		cudaFree(n_smems_max_gpu);
		cudaFree(cub_scan_temp_storage);
		cudaFree(cub_select_temp_storage);
		cudaFree(cub_sort_temp_storage);
		cudaFree(n_seeds_fow_rev_scan);
		if(!is_smem) cudaFree(seed_ref_pos_fow_rev_gpu);
		//cudaFree(n_ref_pos_fow_rev_gpu);
		cudaFree(n_seeds_is_smem_flag_fow_rev);
		//cudaFree(is_smem_fow_rev_flag);

		//		cudaFree(seed_intervals_l_gpu); cudaFree(seed_intervals_u_gpu);
		//		cudaFree(seed_read_begin_gpu); cudaFree(seed_read_end_gpu);
		//		cudaFree(num_seeds_gpu);
		//		cudaFree(seed_ref_begin_gpu);

		//cudaFree(seed_ref_pos_fow_gpu); cudaFree(seed_ref_pos_rev_gpu);

		//free(seed_intervals_fow); free(seed_intervals_rev);
		free(seed_read_pos_fow_rev);
		free(seed_ref_pos_fow_rev);
		free(n_ref_pos_fow_rev);
//		free(num_seeds);
//		free(seed_ref_begin);
//		free(read_num); free(read_idx);
		total_print_time += (realtime_gpu() - print_time);
		
		if (print_stats)
			fprintf(stderr, "Total time to print the results of the batch is %.3f seconds\n", realtime_gpu() - print_time);
		fprintf(stderr, "-----------------------------------------------------------------------------------------------------------\n");
	}
	free(read_batch); free(read_sizes); free(read_offsets);
	double mem_time3 = realtime_gpu();
	cudaFree(pre_calc_seed_intervals);
	
	cudaFree(bwt_gpu.bwt);
	cudaFree(bwt_gpu.sa);
	bwt_destroy_gpu(bwt);
	
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

	return u+10;
}

#ifdef __cplusplus
}
#endif