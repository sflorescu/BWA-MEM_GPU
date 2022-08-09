#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include "cub/cub.cuh"
#include "seed_gen.h"
#include "nvToolsExt.h"

#define CHECK(err) if (err != cudaSuccess) { \
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

#define OCC_INTERVAL 0x80

//#define OCC_INTERVAL 0x40

//#define OCC_INTERVAL 0x20

__constant__ bwtint_t_gpu L2_gpu[5];
__constant__ uint32_t ascii_to_dna_table[8];


/* retrieve a character from the $-removed BWT string. Note that
 * bwt_t_gpu::bwt is not exactly the BWT string and therefore this macro is
 * called bwt_B0 instead of bwt_B */

#define bwt_bwt1(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 8) + 8])

//#define bwt_bwt(b, k) ((b).bwt[(k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 8) + 8 + (k)%OCC_INTERVAL/16])
#define bwt_bwt(b, k) ((b).bwt[((k)>>7<<4) + sizeof(bwtint_t) + (((k)&0x7f)>>4)])


#define bwt_B0(b, k) (bwt_bwt(b, k)>>((~(k)&0xf)<<1)&3)

#define bwt_occ_intv(b, k) ((b).bwt + ((k)>>7<<4))

//#define bwt_occ_intv(b, k) ((b).bwt + (k)/OCC_INTERVAL*((OCC_INTERVAL>>4) + 8))

//#define bwt_bwt(b, k) ((b).bwt[(k)/OCC_INTERVAL * (OCC_INTERVAL/(sizeof(uint32_t)*8/2) + sizeof(bwtint_t)/4*4) + sizeof(bwtint_t)/4*4 + (k)%OCC_INTERVAL/16])
//#define bwt_occ_intv(b, k) ((b).bwt + (k)/OCC_INTERVAL * (OCC_INTERVAL/(sizeof(uint32_t)*8/2) + sizeof(bwtint_t)/4*4))


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
//#if defined(DEBUG) || defined(_DEBUG)   
   bool abort = false;
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
//#endif  
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
	
	bwtint_t_gpu n;
	uint32_t *p, *end;
	
	if (k == bwt.seq_len) return L2_gpu[c+1] - L2_gpu[c];
	if (k == (bwtint_t_gpu)(-1)) return 0;
	if (k >= bwt.primary) --k; // because $ is not in bwt

	// retrieve Occ at k/OCC_INTERVAL
	n = ((bwtint_t_gpu*)(p = bwt_occ_intv(bwt, k)))[c];
	p += sizeof(bwtint_t_gpu); // jump to the start of the first BWT cell

	// calculate Occ up to the last k/32
	end = p + (((k>>5) - ((k&~(OCC_INTERVAL-1))>>5))<<1);
	for (; p < end; p += 2) n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);

	// calculate Occ
	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
	if (c == 0) n -= ~k&31; // corrected for the masked bits

	return n;
}

__device__ inline uint4 find_occ_gpu(const bwt_t_gpu bwt, bwtint_t_gpu l, bwtint_t_gpu u, uint8_t c)
{

	bwtint_t_gpu _l, _u;
	_l = (l >= bwt.primary)? l-1 : l;
	_u = (u >= bwt.primary)? u-1 : u;
	if (_u/OCC_INTERVAL != _l/OCC_INTERVAL || l == (bwtint_t_gpu)(-1) || u == (bwtint_t_gpu)(-1)) {
		l = bwt_occ_gpu(bwt, l, c);
		u = bwt_occ_gpu(bwt, u, c);
	} else {
		bwtint_t_gpu m, n, i, j;
		uint32_t *p;
		if (l >= bwt.primary) --l;
		if (u >= bwt.primary) --u;
		n = ((bwtint_t_gpu*)(p = bwt_occ_intv(bwt, l)))[c];
		p += sizeof(bwtint_t_gpu);
		// calculate *ok
		j = l >> 5 << 5;
		for (i = l/OCC_INTERVAL*OCC_INTERVAL; i < j; i += 32, p += 2)
			n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
		m = n;
		n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~l&31)<<1)) - 1), c);
		if (c == 0) n -= ~l&31; // corrected for the masked bits
		l = n;
		// calculate *ol
		j = u >> 5 << 5;
		for (; i < j; i += 32, p += 2)
			m += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
		m += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~u&31)<<1)) - 1), c);
		if (c == 0) m -= ~u&31; // corrected for the masked bits
		u = m;
	}

	return make_uint4(l >> 32, l, u >> 32, u);
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

__device__ inline bwtint_t_gpu bwt_inv_psi_gpu(const bwt_t_gpu bwt, bwtint_t_gpu k) {
	bwtint_t_gpu x = k - (k > bwt.primary);
	x = bwt_B0(bwt, x);
	x = L2_gpu[x] + bwt_occ_gpu(bwt, k, x);
	return k == bwt.primary ? 0 : x;
}

#define THREADS_PER_SMEM 1



__global__ void seeds_to_threads(int2 *final_seed_read_pos_fow_rev, bwtint_t_gpu *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t n_smems_sum_fow_rev, uint32_t *final_seed_scores_gpu) {

    int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (tid >= (n_smems_sum_fow_rev)*THREADS_PER_SMEM) return;

    int n_seeds = n_seeds_fow_rev_scan[tid+1] - n_seeds_fow_rev_scan[tid];
    int2 seed_read_pos = seed_read_pos_fow_rev[tid];

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
		//printf("S2T Seed %d: [%d,%d] sa_idx: %lld Score %d\n",offset + i + idx,  final_seed_read_pos_fow_rev[offset + i + idx].x, final_seed_read_pos_fow_rev[offset + i + idx].y, seed_sa_idx_fow_rev_gpu[offset + i + idx],n_seeds);
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
		//next_seed_interval = seed_intervals_fow_rev[tid+1];
	//}

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
				//printf("[x,y] = [%d,%d] nseeds: %d\n", final_seed_read_pos_fow_rev[offset + seed_count].x,final_seed_read_pos_fow_rev[offset + seed_count].y, n_seeds);				
				seed_count++;
			// debugging
			//printf("final_seed_pos.y = %d\n", final_seed_read_pos_fow_rev[offset + seed_count].y);
        	}
        }

        return;

}
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
		bwtint_t_gpu bits_to_append = bwt.sa_upper_bits[bwt.pack_size * idx / 32];
		// for general case: mask instead of 0x1
		bits_to_append = ((bits_to_append >> (idx % 32)) & 0x1) << 32;
		seed_ref_pos_fow_rev_gpu[tid] = (((bwtint_t_gpu) bwt.sa[idx]) | bits_to_append) + itr;
        return;

}


__global__ void locate_seeds_gpu_wrapper(int2 *final_seed_read_pos_fow_rev, bwtint_t_gpu *seed_sa_idx_fow_rev_gpu, uint32_t *n_seeds_fow_rev_scan, uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev,uint32_t *n_smems_sum_fow_rev, uint32_t *n_seeds_sum_fow_rev, bwt_t_gpu bwt, uint32_t *final_seed_scores_gpu) {


	int BLOCKDIM =128;
	int N_BLOCKS = (n_smems_sum_fow_rev[0]*THREADS_PER_SMEM  + BLOCKDIM - 1)/BLOCKDIM;

	n_seeds_fow_rev_scan[n_smems_sum_fow_rev[0]] = n_seeds_sum_fow_rev[0];

	seeds_to_threads<<<N_BLOCKS, BLOCKDIM>>>(final_seed_read_pos_fow_rev, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev, seed_read_pos_fow_rev, n_smems_sum_fow_rev[0], final_seed_scores_gpu);

	bwtint_t_gpu *seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;

	N_BLOCKS = (n_seeds_sum_fow_rev[0]  + BLOCKDIM - 1)/BLOCKDIM;

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


__global__ void count_seed_intervals_gpu(uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow_rev, bwtint_t_gpu *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan, uint32_t* n_ref_pos_fow_rev,  uint32_t n_smems_max, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int thread_read_num = tid/n_smems_max;
	int offset_in_read = tid - (thread_read_num*n_smems_max);
	if(offset_in_read >= n_smems_fow_rev[thread_read_num]) return;
	int intv_idx = n_smems_fow_rev_scan[thread_read_num];
	//int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].y - seed_intervals_fow_rev[intv_idx + offset_in_read].x + 1;
	int n_intervals = seed_intervals_fow_rev[intv_idx + offset_in_read].x >> 1;
	//printf("Seed %d Interval %u\n",intv_idx + offset_in_read, n_intervals);
	//if (n_intervals > 0) n_seeds_fow_rev[intv_idx + offset_in_read] = 1;
	//else if (n_intervals == 0) n_seeds_fow_rev[intv_idx + offset_in_read] = 0;
	n_seeds_fow_rev[intv_idx + offset_in_read] = n_intervals;
	if (n_intervals > 0)  atomicAdd(&n_ref_pos_fow_rev[thread_read_num], n_intervals);
	//f (n_intervals > 0)  atomicAdd(&n_ref_pos_fow_rev[thread_read_num], 1);

	return;

}


__global__ void count_seed_intervals_gpu_mem(uint2 *seed_intervals_fow_rev, int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow,  uint32_t *n_smems_rev, bwtint_t_gpu *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,uint32_t *n_ref_pos_fow_rev, uint32_t n_smems_max, int n_tasks) {

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
			seed_intervals_fow_rev[intv_idx + offset_in_read - 1] = make_uint2 (0, 0);
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

		if(offset_in_read >= n_smems_fow[thread_read_num] || offset_in_read == 0) {
			return;
		}
		int intv_idx = n_smems_fow_rev_scan[thread_read_num];
		if(offset_in_read == n_smems_fow[thread_read_num] - 1){
			seed_intervals_fow_rev[intv_idx + offset_in_read] = seed_intervals_fow_rev_compact[intv_idx + offset_in_read];
		}
		int seed_begin_pos = seed_read_pos_fow_rev_compact[intv_idx + offset_in_read].x;
		int comp_seed_begin_pos = seed_read_pos_fow_rev_compact[intv_idx + offset_in_read - 1].x;
		//if(seed_begin_pos == comp_seed_begin_pos && ((seed_intervals_fow_rev_compact[intv_idx + offset_in_read].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read].x + 1) == (seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].y - seed_intervals_fow_rev_compact[intv_idx + offset_in_read - 1].x + 1))) {
		uint32_t seed_sa_intv = seed_intervals_fow_rev_compact[intv_idx + offset_in_read].x >> 1; 
		// debugging
		// if (seed_sa_intv != 0)
		//     	printf("seed_sa_intv = %u\n", seed_sa_intv);
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

__global__ void filter_seed_intervals_gpu_wrapper(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev,  int2 *seed_read_pos_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev, bwtint_t_gpu* n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan,  uint32_t* n_ref_pos_fow_rev, uint32_t *n_smems_max, uint32_t *n_smems_sum_fow_rev, void *cub_sort_temp_storage, size_t cub_sort_storage_bytes, int total_reads, int n_bits_max_read_size, int is_smem) {

	// uint32_t n_smems_max_val = n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1];
	uint32_t n_smems_max_val = n_smems_max[0];
	int n_tasks = n_smems_max_val*total_reads;
	n_smems_sum_fow_rev[0] = n_smems_sum_fow_rev[0]/2; // divided by 2 because of flags
	int BLOCKDIM = 128;
	//int N_BLOCKS = (2*n_tasks + BLOCKDIM - 1)/BLOCKDIM;
	int N_BLOCKS = (n_tasks + BLOCKDIM - 1)/BLOCKDIM;

	filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);
	cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact, (uint64_t*)seed_read_pos_fow_rev, (uint64_t*)seed_intervals_fow_rev_compact, (uint64_t*)seed_intervals_fow_rev,  n_smems_sum_fow_rev[0], total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size);
	//count_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev, n_smems_fow_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val << 1, 2*n_tasks);
	count_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev, n_smems_fow_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val, n_tasks);

}

__global__ void filter_seed_intervals_gpu_wrapper_mem(uint2 *seed_intervals_fow_rev_compact, int2 *seed_read_pos_fow_rev_compact, uint2 *seed_intervals_fow_rev, uint32_t *n_smems_fow, uint32_t *n_smems_rev, uint32_t *n_smems_fow_rev, bwtint_t_gpu *n_seeds_fow_rev, uint32_t *n_smems_fow_rev_scan, uint32_t *n_ref_pos_fow_rev, uint32_t *n_smems_max, uint32_t *n_smems_sum_fow_rev, int total_reads) {

	// uint32_t n_smems_max_val = n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1];
	uint32_t n_smems_max_val = n_smems_max[0];
	int n_tasks = n_smems_max_val*total_reads;
	n_smems_sum_fow_rev[0] = n_smems_sum_fow_rev[0]/2; // divided by 2 because of flags
	int BLOCKDIM = 128;
	//int N_BLOCKS = (2*n_tasks + BLOCKDIM - 1)/BLOCKDIM;
	int N_BLOCKS = (n_tasks + BLOCKDIM - 1)/BLOCKDIM;

	//filter_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);

	filter_seed_intervals_gpu_mem<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev_compact, seed_read_pos_fow_rev_compact, seed_intervals_fow_rev, n_smems_fow, n_smems_rev, n_smems_fow_rev_scan, n_smems_max_val, n_tasks);
	count_seed_intervals_gpu_mem<<<N_BLOCKS, BLOCKDIM>>>(seed_intervals_fow_rev, seed_read_pos_fow_rev_compact, n_smems_fow, n_smems_rev, n_seeds_fow_rev, n_smems_fow_rev_scan, n_ref_pos_fow_rev, n_smems_max_val, n_tasks);

}


#define N_SHUFFLES 30
__global__ void find_seed_intervals_gpu(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow_rev,
		int2 *seed_read_pos_fow_rev, uint32_t *read_num, uint32_t *read_idx, bwtint_t_gpu *is_smem_fow_rev_flag, uint2* pre_calc_intervals, uint32_t *n_smems_fow,  uint32_t *n_smems_rev,int min_seed_size, bwt_t_gpu bwt, int pre_calc_seed_len, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	//printf("tid : %d Block idx : %u, Blockdimx:%d and threaidx:%u: \n",tid,blockIdx.x, blockDim.x, threadIdx.x);
	//if (tid >= 2*n_tasks) return;
	if (tid >= n_tasks) return;
	//if (tid == 754428) 
	//printf("Read number %lu\n", read_num[tid]);
	int thread_read_num = read_num[tid];
	int read_len = read_sizes[thread_read_num];
	int read_off = read_offsets[thread_read_num];
	//thread_read_num * (MAX_READ_LENGTH - min_seed_size);
	uint32_t thread_read_idx = read_idx[tid];
	int is_active = 0;
	int is_smem = 1;
	int is_shfl[N_SHUFFLES];
	int only_next_time = 0;
	uint32_t neighbour_active[N_SHUFFLES];
	uint32_t prev_intv_size[N_SHUFFLES];

	int m;
	for (m = 0; m < N_SHUFFLES; m++) {
		// second condition (before read nums) is not needed but makes it faster on the Tesla K40
		is_shfl[m] = ((tid%32) - m > 0) ? 1 : 0;
		if (is_shfl[m]) is_shfl[m] = tid - (m+1) < 0 ? 0 : (thread_read_num == read_num[tid - (m+1)]) ? 1 : 0;
		prev_intv_size[m] = 0;
		neighbour_active[m] = 1;
	}

	int i, j;
	int base;
	bwtint_t_gpu l, u;

	if (tid < n_tasks) {
		// int intv_idx = (2*(read_offsets[thread_read_num] - (thread_read_num*min_seed_size))) + read_len - min_seed_size - 1;
		// int intv_idx = ((read_offsets[thread_read_num] - (thread_read_num*min_seed_size))) + read_len - min_seed_size - 1;
		int intv_idx = read_offsets[thread_read_num] - thread_read_num*(min_seed_size-1) + read_len - min_seed_size;
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
		//uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
		// instead of l=1, sa_intv is assigned to 0
		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(0,0) : pre_calc_intervals[pre_calc_seed];
		
		uint32_t curr_intv_size = prev_seed_interval.x >> 1;
		bwtint_t_gpu l_prev, u_prev;
	    u_prev = (( ((bwtint_t_gpu) prev_seed_interval.x) & 0x1) << 32) | ((bwtint_t_gpu) prev_seed_interval.y);	
		l_prev = u_prev - (bwtint_t_gpu) (prev_seed_interval.x >> 1) + 1;


		int beg_i = i;

		if (curr_intv_size > 0) {
			is_active = 1;

			l = l_prev, u = u_prev;
			for (; i >= start; i--) {
				/*get the base*/
				if (is_active) {

				int reg_no = i >> 3;
				int reg_pos = i & 7;
				int reg = seq[reg_no];
				int base = (reg >> (28 - (reg_pos << 2)))&15;

				if (base > 3) {
					is_active = 0;
					break;
				}				

				uint4 intv = find_occ_gpu(bwt, l_prev - 1, u_prev, base);

				bwtint_t_gpu intv_x = (bwtint_t_gpu) intv.x << 32;
				bwtint_t_gpu intv_y = intv.y;
				bwtint_t_gpu occ_l = intv_x | intv_y;
				bwtint_t_gpu intv_z = (bwtint_t_gpu) intv.z << 32;
				bwtint_t_gpu intv_w = intv.w;
				bwtint_t_gpu occ_u = intv_z | intv_w;

				l = L2_gpu[base] + occ_l + 1; // modified to accomodate 64 bits for l
				u = L2_gpu[base] + occ_u; // modified to accomodate 64 bits for u

				// beg_i = (i == start) ? i - 1 : i;
				}


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

			if (l > u || base > 3) {
				is_active = 0;
			}

			curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;

			if (is_active) {
                                l_prev = l, u_prev = u;
                                beg_i = i-1;
                        }


		}
	}
	if (read_len - thread_read_idx - beg_i + start - 1 >= min_seed_size && is_smem) {
		atomicAdd(&n_smems_fow[thread_read_num], 1);
		
		uint32_t sa_intv_upp_u; 
		uint32_t low_u = u_prev; // register with lower 32-bits of u
		if (l_prev <= u_prev) {
			sa_intv_upp_u = (u_prev - l_prev + 1) << 1; // store the s_intv = u - l + 1 (occupies the leftmost 31 bits)
			sa_intv_upp_u |= (u_prev >> 32); // store the 33rd bit of u in the rightmost bit
		}
		else
			sa_intv_upp_u = 0; // sa_intv = 0, negatives would cause a problem 

		seed_intervals_fow_rev[intv_idx - thread_read_idx] = make_uint2(sa_intv_upp_u, low_u); // modified to take sa_intv and u
				
		seed_read_pos_fow_rev[intv_idx - thread_read_idx] = make_int2 (beg_i - start + 1, read_len - thread_read_idx);
		// this is now uint64_t and needs to validate two values of intervals-pos while being traversed in Cub::DeviceSelect::Flagged
		is_smem_fow_rev_flag[intv_idx - thread_read_idx] = 0x0000000100000001;
	}

	}

	return;

}

__global__ void find_seed_intervals_gpu2(uint32_t *packed_read_batch_fow,  uint32_t *packed_read_batch_rev, uint32_t *read_sizes, uint32_t *read_offsets, uint2 *seed_intervals_fow_rev,
		int2 *seed_read_pos_fow_rev, uint32_t *read_num, uint32_t *read_idx, bwtint_t_gpu *is_smem_fow_rev_flag, uint2* pre_calc_intervals, uint32_t *n_smems_fow,  uint32_t *n_smems_rev,int min_seed_size, bwt_t_gpu bwt, int pre_calc_seed_len, int n_tasks) {

	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	//printf("tid : %d Block idx : %u, Blockdimx:%d and threaidx:%u: \n",tid,blockIdx.x, blockDim.x, threadIdx.x);
	//if (tid >= 2*n_tasks) return;
	if (tid >= n_tasks) return;
	//if (tid == 754428) 
	//printf("Read number %lu\n", read_num[tid]);
	int thread_read_num = read_num[tid];
	int read_len = read_sizes[thread_read_num];
	int read_off = read_offsets[thread_read_num];
	//thread_read_num * (MAX_READ_LENGTH - min_seed_size);
	uint32_t thread_read_idx = read_idx[tid];
	int is_active = 0;
	int is_smem = 1;
	int is_shfl[N_SHUFFLES];
	int only_next_time = 0;
	uint32_t neighbour_active[N_SHUFFLES];
	uint32_t prev_intv_size[N_SHUFFLES];

	int m;
	for (m = 0; m < N_SHUFFLES; m++) {
		// second condition (before read nums) is not needed but makes it faster on the Tesla K40
		is_shfl[m] = ((tid%32) - m > 0) ? 1 : 0;
		if (is_shfl[m]) is_shfl[m] = tid - (m+1) < 0 ? 0 : (thread_read_num == read_num[tid - (m+1)]) ? 1 : 0;
		prev_intv_size[m] = 0;
		neighbour_active[m] = 1;
	}

	int i, j;
	int base;
	bwtint_t_gpu l, u;

	if (tid < n_tasks) {
		// int intv_idx = (2*(read_offsets[thread_read_num] - (thread_read_num*min_seed_size))) + read_len - min_seed_size - 1;
		// int intv_idx = ((read_offsets[thread_read_num] - (thread_read_num*min_seed_size))) + read_len - min_seed_size - 1;
		int intv_idx = read_offsets[thread_read_num] - thread_read_num*(min_seed_size-1) + read_len - min_seed_size;
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
		//uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(1,0) : pre_calc_intervals[pre_calc_seed];
		// instead of l=1, sa_intv is assigned to 0
		uint2 prev_seed_interval = j < pre_calc_seed_len ? make_uint2(0,0) : pre_calc_intervals[pre_calc_seed];
		
		uint32_t curr_intv_size = prev_seed_interval.x >> 1;
		bwtint_t_gpu l_prev, u_prev;
	    u_prev = (( ((bwtint_t_gpu) prev_seed_interval.x) & 0x1) << 32) | ((bwtint_t_gpu) prev_seed_interval.y);	
		l_prev = u_prev - (bwtint_t_gpu) (prev_seed_interval.x >> 1) + 1;


		int beg_i = i;

		if (curr_intv_size > 0) {
			is_active = 1;

			l = l_prev, u = u_prev;
			for (; i >= start; i--) {
				/*get the base*/
				if (is_active) {

				int reg_no = i >> 3;
				int reg_pos = i & 7;
				int reg = seq[reg_no];
				int base = (reg >> (28 - (reg_pos << 2)))&15;

				if (base > 3) {
					is_active = 0;
					break;
				}				

				uint4 intv = find_occ_gpu(bwt, l_prev - 1, u_prev, base);

				bwtint_t_gpu intv_x = (bwtint_t_gpu) intv.x << 32;
				bwtint_t_gpu intv_y = intv.y;
				bwtint_t_gpu occ_l = intv_x | intv_y;
				bwtint_t_gpu intv_z = (bwtint_t_gpu) intv.z << 32;
				bwtint_t_gpu intv_w = intv.w;
				bwtint_t_gpu occ_u = intv_z | intv_w;

				l = L2_gpu[base] + occ_l + 1; // modified to accomodate 64 bits for l
				u = L2_gpu[base] + occ_u; // modified to accomodate 64 bits for u

				// beg_i = (i == start) ? i - 1 : i;
				}


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

			if (l > u || base > 3) {
				is_active = 0;
			}

			curr_intv_size =  l <= u ? u - l + 1 : curr_intv_size;

			if (is_active) {
                                l_prev = l, u_prev = u;
                                beg_i = i-1;
                        }


		}
	}
	if (read_len - thread_read_idx - beg_i + start - 1 >= min_seed_size && is_smem) {
		atomicAdd(&n_smems_fow[thread_read_num], 1);
			//seed_intervals_fow_rev[intv_idx - thread_read_idx] = make_uint2(prev_seed_interval.x, prev_seed_interval.y);
			// uint32_t low_u = u_prev; // register with lower 32-bits of u
			// uint32_t sa_intv_upp_u = (u_prev - l_prev + 1) << 1; // store the s_intv = u - l + 1 (occupies the leftmost 31 bits)
			// sa_intv_upp_u |= (u_prev >> 32); // store the 33rd bit of u in the rightmost bit
		
			uint32_t sa_intv_upp_u; 
			uint32_t low_u = u_prev; // register with lower 32-bits of u
			if (l_prev <= u_prev) {
				sa_intv_upp_u = (u_prev - l_prev + 1) << 1; // store the s_intv = u - l + 1 (occupies the leftmost 31 bits)
				sa_intv_upp_u |= (u_prev >> 32); // store the 33rd bit of u in the rightmost bit
			}
			else
				sa_intv_upp_u = 0; // sa_intv = 0, negatives would cause a problem 

			// debugging
			//if (tid < 50) {
			//	printf("find_sa_intv = %u\n", sa_intv_upp_u >> 1);
			//} 

			
			seed_intervals_fow_rev[intv_idx - thread_read_idx] = make_uint2(sa_intv_upp_u, low_u); // modified to take sa_intv and u
					
			seed_read_pos_fow_rev[intv_idx - thread_read_idx] = make_int2 (beg_i - start + 1, read_len - thread_read_idx);
			// is_smem_fow_rev_flag[intv_idx - thread_read_idx] = 0x00010001;
			// this is now uint64_t and needs to validate two values of intervals-pos while being traversed in Cub::DeviceSelect::Flagged
			is_smem_fow_rev_flag[intv_idx - thread_read_idx] = 0x0000000100000001;
		}

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
		// if (offset_in_read >= read_sizes[read_no] - min_seed_len) return;
		if (offset_in_read > read_sizes[read_no] - min_seed_len) return;
		thread_read_num[read_offsets[read_no] - (read_no*(min_seed_len-1)) + offset_in_read] = read_no;
		//printf("Readno %lu\n",thread_read_num[read_offsets[read_no] - (read_no*(min_seed_len-1)) + offset_in_read]);
		thread_read_idx[read_offsets[read_no] - (read_no*(min_seed_len-1)) + offset_in_read] = offset_in_read;

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
		uint4 intv = find_occ_gpu(bwt, l - 1, u, ch);

		bwtint_t_gpu intv_x = (bwtint_t_gpu) intv.x << 32;
		bwtint_t_gpu intv_y = intv.y;
		bwtint_t_gpu intv_l = intv_x | intv_y;
		bwtint_t_gpu intv_z = (bwtint_t_gpu) intv.z << 32;
		bwtint_t_gpu intv_w = intv.w;
		bwtint_t_gpu intv_u = intv_z | intv_w;


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
	//printf("[GPU] Primary: %llu and seq_len: %llu\n",primary,bwt->seq_len);
	//printf("[GPU] Sa_intv: %d and bwt_size: %llu\n",bwt->sa_intv,bwt->bwt_size);

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


void bwt_destroy_gpu(bwt_t_gpu *bwt)
{

	if (bwt == 0) return;
	cudaFreeHost(bwt->sa); cudaFreeHost(bwt->sa_upper_bits); cudaFreeHost(bwt->bwt);
	cudaFreeHost(bwt->L2); cudaFreeHost(bwt);

	//free(bwt->sa); free(bwt->sa_upper_bits); free(bwt->bwt);
	//free(bwt->L2);
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
	//cudaDeviceSynchronize();
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

mem_seed_v_gpu *seed_gpu(gpuseed_storage_vector *gpuseed_data, int n_reads, int64_t n_processed, bseq1_t *seqs) {
	
	//fprintf(stderr, "I go to seed with n_reads: %d and Processed: %ld\n",n_reads, n_processed);

	int min_seed_size = gpuseed_data->min_seed_size;
	int is_smem = gpuseed_data->is_smem;
	int print_stats = 0;
	int c;
	
	
	double total_time = realtime_gpu();
	mem_seed_v_gpu *from_gpu_results = (mem_seed_v_gpu*)(malloc(sizeof(mem_seed_v_gpu)));
	from_gpu_results->n_ref_pos_fow_rev_results = (uint32_t*)calloc(n_reads, sizeof(uint32_t));	
	from_gpu_results->n_ref_pos_fow_rev_prefix_sums = (uint32_t*)calloc(n_reads, sizeof(uint32_t));
	uint32_t *n_ref_pos_fow_rev_prefix_sums_gpu;
	uint32_t *n_ref_pos_fow_rev_prefix_sums_gpu_results;
	gpuErrchk(cudaMalloc(&(n_ref_pos_fow_rev_prefix_sums_gpu), n_reads*sizeof(uint32_t)));
	gpuErrchk(cudaMalloc(&(n_ref_pos_fow_rev_prefix_sums_gpu_results), n_reads*sizeof(uint32_t)));

	void *cub_scan_temp_storage_end = NULL;
	size_t cub_scan_storage_bytes_end = 0;

	cudaError_t result;

	int nStreams = 2;
	cudaStream_t stream[nStreams];

  	for (int i = 0; i < nStreams; ++i){
		result = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

	if (from_gpu_results == NULL) {
        fprintf(stderr, "Fatal: failed to allocate %zu bytes.\n", n_reads * sizeof(mem_seed_v));
        abort();
    }

	//uint2 *pre_calc_seed_intervals;
	//gpuErrchk(cudaMalloc(&pre_calc_seed_intervals, (1 << (gpuseed_data->pre_calc_seed_len<<1))*sizeof(uint2)));

	if (!gpuseed_data->pre_calc_seed_intervals_flag) {
		gpuErrchk(cudaMalloc(&gpuseed_data->pre_calc_seed_intervals, (1 << (gpuseed_data->pre_calc_seed_len<<1))*sizeof(uint2)));	
		int threads_per_block_pre_calc_seed = 128;
    	int num_blocks_pre_calc_seed = ((1 << (gpuseed_data->pre_calc_seed_len<<1)) + threads_per_block_pre_calc_seed - 1)/threads_per_block_pre_calc_seed;
    	pre_calc_seed_intervals_gpu<<<num_blocks_pre_calc_seed, threads_per_block_pre_calc_seed>>>(gpuseed_data->pre_calc_seed_intervals, gpuseed_data->pre_calc_seed_len, gpuseed_data->bwt_gpu, (1 << (gpuseed_data->pre_calc_seed_len<<1)));
		gpuseed_data->pre_calc_seed_intervals_flag = 1;
		//gpuErrchk(cudaMallocHost((uint2**)&(gpuseed_data->pre_calc_seed_intervals), (1 << (gpuseed_data->pre_calc_seed_len<<1))*sizeof(uint2)));
		//gpuErrchk(cudaMemcpy(gpuseed_data->pre_calc_seed_intervals, pre_calc_seed_intervals, (1 << (gpuseed_data->pre_calc_seed_len<<1))*sizeof(uint2), cudaMemcpyDeviceToDevice));
	}
	//else {
		//gpuErrchk(cudaMemcpy(pre_calc_seed_intervals, gpuseed_data->pre_calc_seed_intervals, (1 << (gpuseed_data->pre_calc_seed_len<<1))*sizeof(uint2), cudaMemcpyHostToDevice));

	//}

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
	int reads_processed = 0;
	int max_read_size = 0;
	int read_count = seqs[0].id;
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
				if (reads_loaded == n_reads) {
					all_done = 1;
					break;
				}				
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
		bwtint_t_gpu *n_seeds_fow_rev;

		uint32_t *thread_read_num, *thread_read_idx;
		int2  *smem_intv_read_pos_fow_rev;
		uint32_t *n_smems_sum_fow_rev_gpu;
		uint32_t n_smems_sum_fow_rev;
		uint32_t *n_seeds_sum_fow_rev_gpu;
		uint32_t n_seeds_sum_fow_rev;

		uint32_t* n_smems_max_gpu;

		uint32_t *n_smems_fow_rev_scan;
		uint32_t *n_seeds_fow_rev_scan;

		void *cub_scan_temp_storage = NULL;
		size_t cub_scan_storage_bytes = 0;
		void *cub_select_temp_storage = NULL;
		size_t cub_select_storage_bytes = 0;
		void *cub_sort_temp_storage = NULL;
		size_t cub_sort_storage_bytes = 0;

		CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_fow_rev_scan, (read_batch_size_8 - (total_reads*(min_seed_size-1)))));
		
		if (print_stats)
			fprintf(stderr, "ExclusiveSum bytes for n_smems = %d\n", cub_scan_storage_bytes);

		int max_output_size = 2*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads));

		max_output_size = max_output_size > (2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)) + (read_batch_size >> 3) + 2*total_reads + read_batch_size_8 >> 2 + (read_batch_size_8 - ((min_seed_size-1)*total_reads))) ? max_output_size : (2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)) + (read_batch_size >> 3) + 2*total_reads + read_batch_size_8 >> 2 + (read_batch_size_8 - ((min_seed_size-1)*total_reads)));

		gpuErrchk(cudaMalloc(&read_batch_gpu, read_batch_size_8));
		gpuErrchk(cudaMalloc(&read_sizes_gpu, total_reads*sizeof(uint32_t)));
		gpuErrchk(cudaMalloc(&read_offsets_gpu,total_reads*sizeof(uint32_t)));
		gpuErrchk(cudaMalloc(&n_smems_fow,total_reads*sizeof(uint32_t)));
		n_smems_fow_rev = read_sizes_gpu;
		gpuErrchk(cudaMalloc(&n_smems_fow_rev_scan,(total_reads+1)*sizeof(uint32_t)));

		gpuErrchk(cudaMalloc(&packed_read_batch_fow,(read_batch_size_8 >> 3)*sizeof(uint32_t)));
		gpuErrchk(cudaMalloc(&n_seeds_fow_rev_scan, ((2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))) + 1)*sizeof(uint32_t)));
		thread_read_num = n_seeds_fow_rev_scan;
		thread_read_idx = &n_seeds_fow_rev_scan[read_batch_size_8 - ((min_seed_size-1)*total_reads)];
		gpuErrchk(cudaMalloc(&cub_scan_temp_storage,cub_scan_storage_bytes));
		n_smems_sum_fow_rev_gpu = &n_smems_fow_rev_scan[total_reads];
		gpuErrchk(cudaMalloc(&n_seeds_sum_fow_rev_gpu, sizeof(uint32_t)));
		gpuErrchk(cudaMalloc(&n_smems_max_gpu, sizeof(uint32_t)));

		uint2 *seed_intervals_fow_rev_gpu;
		int2 *seed_read_pos_fow_rev_gpu;
		uint2 *seed_intervals_fow_rev_compact_gpu;
		int2 *seed_read_pos_fow_rev_compact_gpu;
		bwtint_t_gpu *is_smem_fow_rev_flag;
		uint32_t *n_ref_pos_fow_rev_gpu;
		uint32_t *n_ref_pos_fow_rev_scan;
		bwtint_t_gpu *seed_ref_pos_fow_rev_gpu;
		bwtint_t_gpu *seed_sa_idx_fow_rev_gpu;

		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint32_t*)seed_intervals_fow_rev_gpu, (uint32_t*)is_smem_fow_rev_flag, (uint32_t*)seed_intervals_fow_rev_compact_gpu, n_smems_sum_fow_rev_gpu, 2*(read_batch_size_8 - (total_reads*(min_seed_size-1)))));
		if (print_stats)
			fprintf(stderr, "Flagged bytes = %d\n", cub_select_storage_bytes);

		if (is_smem) {
			CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, (uint64_t*)seed_intervals_fow_rev_compact_gpu, (uint64_t*)seed_intervals_fow_rev_gpu,   2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)), total_reads, n_smems_fow_rev_scan, n_smems_fow_rev_scan + 1, 0, n_bits_max_read_size));
			if (print_stats)
				fprintf(stderr, "Sort bytes = %d\n", cub_sort_storage_bytes);
		} else {
			CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, seed_sa_idx_fow_rev_gpu, seed_ref_pos_fow_rev_gpu,  OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)), total_reads, n_ref_pos_fow_rev_scan, n_ref_pos_fow_rev_scan + 1, 0, n_bits_max_read_size));
			if (print_stats)
				fprintf(stderr, "Sort bytes = %d\n", cub_sort_storage_bytes);
		}

		uint2 *seed_intervals_pos_fow_rev_gpu;
		gpuErrchk(cudaMalloc(&seed_intervals_pos_fow_rev_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint2)));
		seed_read_pos_fow_rev_gpu = (int2*)seed_intervals_pos_fow_rev_gpu;
		seed_intervals_fow_rev_gpu = &seed_intervals_pos_fow_rev_gpu[(read_batch_size_8 - ((min_seed_size-1)*total_reads))];
		uint2 *seed_intervals_pos_fow_rev_compact_gpu;
		gpuErrchk(cudaMalloc(&seed_intervals_pos_fow_rev_compact_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(uint2)));
		seed_read_pos_fow_rev_compact_gpu = (int2*)seed_intervals_pos_fow_rev_compact_gpu;
		seed_intervals_fow_rev_compact_gpu = &seed_intervals_pos_fow_rev_compact_gpu[(read_batch_size_8 - ((min_seed_size-1)*total_reads))];

		bwtint_t_gpu *n_seeds_is_smem_flag_fow_rev;
		gpuErrchk(cudaMalloc(&n_seeds_is_smem_flag_fow_rev, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(bwtint_t_gpu)));
		n_seeds_fow_rev = n_seeds_is_smem_flag_fow_rev;
		is_smem_fow_rev_flag = &n_seeds_is_smem_flag_fow_rev[(read_batch_size_8 - ((min_seed_size-1)*total_reads))];

		if(!is_smem) gpuErrchk(cudaMalloc(&seed_ref_pos_fow_rev_gpu, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(bwtint_t_gpu)));

		gpuErrchk(cudaMalloc(&cub_select_temp_storage,cub_select_storage_bytes));
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
		double assign_threads_for_fow_pack_time_start = realtime_gpu();
		
		double assign_threads_for_fow_pack_time = realtime_gpu() - assign_threads_for_fow_pack_time_start;

		double fow_pack_time_start = realtime_gpu();
		int N_BLOCKS = ((read_batch_size_8 >> 3)  + BLOCKDIM - 1)/BLOCKDIM;
		pack_4bit_fow<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)read_batch_gpu, packed_read_batch_fow, /*thread_read_num, read_offsets_gpu, read_sizes_gpu,*/ (read_batch_size_8 >> 3));
		//pack_4bit_rev<<<N_BLOCKS, BLOCKDIM, 0 , stream[1]>>>((uint32_t*)read_batch_gpu, packed_read_batch_fow, /*thread_read_num, read_offsets_gpu, read_sizes_gpu,*/ (read_batch_size_8 >> 3));
		
		double fow_pack_time = realtime_gpu() - fow_pack_time_start;
		double assign_threads_time_start = realtime_gpu();
		N_BLOCKS = ((total_reads*max_read_size) + BLOCKDIM - 1)/BLOCKDIM;

		prepare_batch<<<N_BLOCKS, BLOCKDIM>>>(thread_read_num, thread_read_idx, read_offsets_gpu, read_sizes_gpu, min_seed_size, total_reads*max_read_size, 0, max_read_size);
		
		double assign_threads_time = realtime_gpu() - assign_threads_time_start;

		total_batch_prep_time += (realtime_gpu() - batch_prep_time);

		if (print_stats) {
			fprintf(stderr,"Batch prepared for processing on GPU in %.3f seconds\n", realtime_gpu() - batch_prep_time);
			fprintf(stderr,"\tThread assignment for forward batch packing on GPU in %.3f seconds\n", assign_threads_for_fow_pack_time);
			fprintf(stderr,"\tForward batch packed on GPU in %.3f seconds\n",fow_pack_time);
			fprintf(stderr,"\tThread assignment for computing seed intervals on GPU in %.3f seconds\n",assign_threads_time);
			fprintf(stderr, "Processing %d reads on GPU...\n", total_reads);
		}


		double find_seeds_time = realtime_gpu();
		int n_seed_cands = read_batch_size - (total_reads*(min_seed_size-1));
		gpuErrchk(cudaMemset(is_smem_fow_rev_flag, 0, (read_batch_size_8 - ((min_seed_size-1)*total_reads))*sizeof(bwtint_t_gpu)));
		gpuErrchk(cudaMemset(n_smems_fow, 0, total_reads*sizeof(uint32_t)));

		N_BLOCKS = ((n_seed_cands) + BLOCKDIM - 1)/BLOCKDIM;	
		//find_seed_intervals_gpu2<<<N_BLOCKS, BLOCKDIM, 0, stream[1]>>>(packed_read_batch_fow, packed_read_batch_rev,  read_sizes_gpu, read_offsets_gpu, seed_intervals_fow_rev_gpu,
				//seed_read_pos_fow_rev_gpu, thread_read_num, thread_read_idx, is_smem_fow_rev_flag, gpuseed_data->pre_calc_seed_intervals, n_smems_fow, n_smems_rev, min_seed_size, gpuseed_data->bwt_gpu, gpuseed_data->pre_calc_seed_len,  n_seed_cands);
					
		
		find_seed_intervals_gpu<<<N_BLOCKS, BLOCKDIM>>>(packed_read_batch_fow, packed_read_batch_rev,  read_sizes_gpu, read_offsets_gpu, seed_intervals_fow_rev_gpu,
				seed_read_pos_fow_rev_gpu, thread_read_num, thread_read_idx, is_smem_fow_rev_flag, gpuseed_data->pre_calc_seed_intervals, n_smems_fow, n_smems_rev, min_seed_size, gpuseed_data->bwt_gpu, gpuseed_data->pre_calc_seed_len,  n_seed_cands);

			
		if (print_stats)
			fprintf(stderr,"\tIntervals of SMEM seeds computed in %.3f seconds on GPU\n", realtime_gpu() - find_seeds_time);
		total_find_seed_intervals_time += realtime_gpu() - find_seeds_time;


		double n_smems_fow_max_time = realtime_gpu();	

		void *cub_red_temp_storage = NULL;
		size_t cub_red_storage_bytes = 0;
		
		CubDebugExit(cub::DeviceReduce::Max(cub_red_temp_storage, cub_red_storage_bytes, n_smems_fow, &n_smems_max_gpu[0], total_reads));		
		gpuErrchk(cudaMalloc(&cub_red_temp_storage,cub_red_storage_bytes));	
		CubDebugExit(cub::DeviceReduce::Max(cub_red_temp_storage, cub_red_storage_bytes, n_smems_fow, &n_smems_max_gpu[0], total_reads));		

		if (print_stats)
			fprintf(stderr,"\tMax in n_smems_fow found in %.3f seconds\n", realtime_gpu() - n_smems_fow_max_time);

		double filter_seeds_time = realtime_gpu();
		cudaError_t err = cudaSuccess;
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (uint32_t*)seed_intervals_fow_rev_gpu, (uint32_t*)is_smem_fow_rev_flag, (uint32_t*)seed_intervals_fow_rev_compact_gpu, n_smems_sum_fow_rev_gpu, 2*(read_batch_size_8 - (total_reads*(min_seed_size-1)))));
		CubDebugExit(cub::DeviceSelect::Flagged(cub_select_temp_storage, cub_select_storage_bytes, (int32_t*)seed_read_pos_fow_rev_gpu, (uint32_t*)is_smem_fow_rev_flag, (int32_t*)seed_read_pos_fow_rev_compact_gpu, n_smems_sum_fow_rev_gpu, 2*(read_batch_size_8 - (total_reads*(min_seed_size-1)))));

		//N_BLOCKS = (total_reads + BLOCKDIM - 1)/BLOCKDIM;
		
		// sum_arrays<<<N_BLOCKS, BLOCKDIM>>>(n_smems_fow, n_smems_rev, n_smems_fow_rev, total_reads);
		// this has a different outcome for some reason..
		n_smems_fow_rev = n_smems_fow;

		double n_smems_fow_rev_scan_time = realtime_gpu();
		CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_smems_fow_rev, n_smems_fow_rev_scan, total_reads));
		
		
		if (print_stats)
			fprintf(stderr,"\tn_smems_fow_rev scanned in %.3f seconds on GPU\n", realtime_gpu() - n_smems_fow_rev_scan_time);
		
		gpuErrchk(cudaMemset(n_ref_pos_fow_rev_gpu, 0, total_reads*sizeof(uint32_t)));
		
		gpuErrchk(cudaMemcpy(&n_smems_sum_fow_rev, n_smems_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		/*int2 *check1 = (int2*)malloc(n_smems_sum_fow_rev * sizeof(int2));
		uint2 *check2 = (uint2*)malloc(n_smems_sum_fow_rev * sizeof(uint2));
		int2 *check3 = (int2*)malloc(n_smems_sum_fow_rev * sizeof(int2));
		uint2 *check4 = (uint2*)malloc(n_smems_sum_fow_rev  * sizeof(uint2));		
		cudaMemcpy(check3, seed_read_pos_fow_rev_compact_gpu, n_smems_sum_fow_rev *sizeof(int2), cudaMemcpyDeviceToHost);
		cudaMemcpy(check4, seed_intervals_fow_rev_compact_gpu, n_smems_sum_fow_rev *sizeof(uint2), cudaMemcpyDeviceToHost);
		

		for(int p = 0; p < n_smems_sum_fow_rev ; p++){
				printf("Before [Seed (%d) [%d %d] --> Inter [%d,%d]\n", p, check3[p].x, check3[p].y, check4[p].x>>1, check4[p].y);
			printf("\n");
		}*/

		if (is_smem) {			
			filter_seed_intervals_gpu_wrapper<<<1, 1>>>(seed_intervals_fow_rev_compact_gpu, seed_read_pos_fow_rev_compact_gpu, seed_intervals_fow_rev_gpu, seed_read_pos_fow_rev_gpu, n_smems_fow, n_smems_rev, n_smems_fow_rev, n_seeds_fow_rev, n_smems_fow_rev_scan,  n_ref_pos_fow_rev_gpu, n_smems_max_gpu, n_smems_sum_fow_rev_gpu, cub_sort_temp_storage, cub_sort_storage_bytes, total_reads, n_bits_max_read_size, is_smem/*n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads*/);

		}
		else {
			filter_seed_intervals_gpu_wrapper_mem<<<1, 1>>>(seed_intervals_fow_rev_compact_gpu, seed_read_pos_fow_rev_compact_gpu, seed_intervals_fow_rev_gpu, n_smems_fow, n_smems_rev, n_smems_fow_rev,  n_seeds_fow_rev,  n_smems_fow_rev_scan, n_ref_pos_fow_rev_gpu, n_smems_max_gpu, n_smems_sum_fow_rev_gpu, total_reads/*n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1], (n_smems_max[0] > n_smems_max[1] ? n_smems_max[0] : n_smems_max[1])*total_reads*/);

			int2 *swap =  seed_read_pos_fow_rev_compact_gpu;
			seed_read_pos_fow_rev_compact_gpu = seed_read_pos_fow_rev_gpu;
			seed_read_pos_fow_rev_gpu = swap;

		}
		//cudaMemcpy(check1, seed_read_pos_fow_rev_gpu, n_smems_sum_fow_rev*sizeof(int2), cudaMemcpyDeviceToHost);
		//cudaMemcpy(check2, seed_intervals_fow_rev_gpu, n_smems_sum_fow_rev*sizeof(uint2), cudaMemcpyDeviceToHost);
		
		//gpuErrchk(cudaMemcpy(&n_smems_sum_fow_rev, n_smems_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		/*int2 *check1 = (int2*)malloc(n_smems_sum_fow_rev * sizeof(int2));
		uint2 *check2 = (uint2*)malloc(n_smems_sum_fow_rev * sizeof(uint2));

		cudaMemcpy(check1, seed_read_pos_fow_rev_gpu, n_smems_sum_fow_rev*sizeof(int2), cudaMemcpyDeviceToHost);
		cudaMemcpy(check2, seed_intervals_fow_rev_gpu, n_smems_sum_fow_rev*sizeof(uint2), cudaMemcpyDeviceToHost);
		

		for(int p = 0; p < n_smems_sum_fow_rev; p++){
				printf("After [Seed (%d) [%d %d] --> Inter [%d,%d]\n", p, check1[p].x, check1[p].y, check2[p].x>>1, check2[p].y);
			printf("\n");
		}*/

		if (print_stats)
			fprintf(stderr,"\tSMEM seeds filtered in %.3f seconds on GPU\n", realtime_gpu() - n_smems_fow_max_time);
		total_filter_seed_intervals_time += realtime_gpu() - n_smems_fow_max_time;

		double locate_seeds_time = realtime_gpu();


		double n_seeds_fow_rev_sum_time = realtime_gpu();

		void *cub_red2_temp_storage = NULL;
		size_t cub_red2_storage_bytes = 0;
						
		CubDebugExit(cub::DeviceReduce::Sum(cub_red2_temp_storage, cub_red2_storage_bytes, n_seeds_fow_rev, n_seeds_sum_fow_rev_gpu, n_smems_sum_fow_rev));
		gpuErrchk(cudaMalloc(&cub_red2_temp_storage,cub_red2_storage_bytes));	
		CubDebugExit(cub::DeviceReduce::Sum(cub_red2_temp_storage, cub_red2_storage_bytes, n_seeds_fow_rev, n_seeds_sum_fow_rev_gpu, n_smems_sum_fow_rev));

		if (print_stats)
			fprintf(stderr,"\tn_seeds_fow_rev summed in %.3f seconds\n", realtime_gpu() - n_seeds_fow_rev_sum_time);


		double n_seeds_fow_rev_scan_time = realtime_gpu();
		CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_seeds_fow_rev, n_seeds_fow_rev_scan, n_smems_sum_fow_rev));
		
		if (print_stats)
			fprintf(stderr,"\tn_seeds_fow_rev scanned in %.3f seconds on GPU\n", realtime_gpu() - n_seeds_fow_rev_scan_time);

		seed_sa_idx_fow_rev_gpu = n_seeds_fow_rev;

		int2 *final_seed_read_pos_fow_rev_gpu;

		gpuErrchk(cudaMemcpy(&n_seeds_sum_fow_rev, n_seeds_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		if(n_seeds_sum_fow_rev > OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads))) {
			fprintf(stderr,"n_seeds_sum_fow_rev (%llu) is more than allocated size(%d)\n", n_seeds_sum_fow_rev, OUTPUT_SIZE_MUL*2*(read_batch_size_8 - ((min_seed_size-1)*total_reads)));
			exit(EXIT_FAILURE);
		}

		uint32_t *final_seed_scores_gpu;
		cudaMalloc(&final_seed_scores_gpu, n_seeds_sum_fow_rev*sizeof(uint32_t));

		if (is_smem) {
			locate_seeds_gpu_wrapper<<<1, 1>>>(seed_read_pos_fow_rev_compact_gpu, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev_gpu, seed_read_pos_fow_rev_gpu, n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, gpuseed_data->bwt_gpu, final_seed_scores_gpu);

			final_seed_read_pos_fow_rev_gpu = seed_read_pos_fow_rev_compact_gpu;
			seed_ref_pos_fow_rev_gpu = seed_sa_idx_fow_rev_gpu;
		}
		else {
			locate_seeds_gpu_wrapper_mem<<<1, 1>>>(seed_read_pos_fow_rev_compact_gpu, seed_sa_idx_fow_rev_gpu, n_seeds_fow_rev_scan, seed_intervals_fow_rev_gpu, seed_read_pos_fow_rev_gpu, n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, n_ref_pos_fow_rev_gpu, gpuseed_data->bwt_gpu, final_seed_scores_gpu);
			n_ref_pos_fow_rev_scan = n_smems_fow_rev_scan;
			gpuErrchk(cudaMemcpy(n_smems_sum_fow_rev_gpu, n_seeds_sum_fow_rev_gpu, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
			CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage, cub_scan_storage_bytes, n_ref_pos_fow_rev_gpu, n_ref_pos_fow_rev_scan, total_reads));
			CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(cub_sort_temp_storage, cub_sort_storage_bytes, (uint64_t*)seed_read_pos_fow_rev_compact_gpu, (uint64_t*)seed_read_pos_fow_rev_gpu, seed_sa_idx_fow_rev_gpu, seed_ref_pos_fow_rev_gpu,  n_seeds_sum_fow_rev, total_reads, n_ref_pos_fow_rev_scan, n_ref_pos_fow_rev_scan + 1, 0, n_bits_max_read_size));
			final_seed_read_pos_fow_rev_gpu = seed_read_pos_fow_rev_gpu;
		}

		if (print_stats)
			fprintf(stderr,"\tSeeds located on ref in in %.3f seconds\n", realtime_gpu() - locate_seeds_time);
		
		total_locate_seeds_time += realtime_gpu() - locate_seeds_time;

		if (print_stats)
			fprintf(stderr, "n_seed_sum_fow_rev = %d, n_smem_sum_fow = %d\n", n_seeds_sum_fow_rev, n_smems_sum_fow_rev);
		fflush(stderr);

		double mem_time1 = realtime_gpu();

		if (!reads_processed) {
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
		gpuErrchk(cudaFree(seed_intervals_pos_fow_rev_gpu));
		gpuErrchk(cudaFree(seed_intervals_pos_fow_rev_compact_gpu));
		gpuErrchk(cudaFree(packed_read_batch_fow)); gpuErrchk(cudaFree(packed_read_batch_rev));
		gpuErrchk(cudaFree(n_smems_fow)); gpuErrchk(cudaFree(n_smems_fow_rev_scan));

		gpuErrchk(cudaFree(n_seeds_sum_fow_rev_gpu));
		gpuErrchk(cudaFree(n_smems_max_gpu));
		gpuErrchk(cudaFree(cub_scan_temp_storage));
		gpuErrchk(cudaFree(cub_select_temp_storage));
		gpuErrchk(cudaFree(cub_red_temp_storage));
		gpuErrchk(cudaFree(cub_red2_temp_storage));	
		gpuErrchk(cudaFree(cub_sort_temp_storage));
		gpuErrchk(cudaFree(n_seeds_fow_rev_scan));
		gpuErrchk(cudaFree(final_seed_scores_gpu));
		if(!is_smem) gpuErrchk(cudaFree(seed_ref_pos_fow_rev_gpu));
		gpuErrchk(cudaFree(n_seeds_is_smem_flag_fow_rev));

	}
	from_gpu_results->file_bytes_skip = file_bytes;
	//fprintf(stderr,"[GPUSeed] File bytes %llu, ref: %llu\n",from_gpu_results->file_bytes_skip, file_bytes);
	
	CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage_end, cub_scan_storage_bytes_end, n_ref_pos_fow_rev_prefix_sums_gpu, n_ref_pos_fow_rev_prefix_sums_gpu_results, n_reads));
	
	gpuErrchk(cudaMalloc(&cub_scan_temp_storage_end,cub_scan_storage_bytes_end));

	CubDebugExit(cub::DeviceScan::ExclusiveSum(cub_scan_temp_storage_end, cub_scan_storage_bytes_end, n_ref_pos_fow_rev_prefix_sums_gpu, n_ref_pos_fow_rev_prefix_sums_gpu_results, n_reads));

	gpuErrchk(cudaMemcpy(from_gpu_results->n_ref_pos_fow_rev_prefix_sums, n_ref_pos_fow_rev_prefix_sums_gpu_results, n_reads*sizeof(uint32_t), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(cub_scan_temp_storage_end));
	//mem_print_gpuseed(from_gpu_results, n_reads);
	//mem_rev_print_gpuseed(from_gpu_results, n_reads);
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
	//printf("I return from GPUSeed\n");
	return from_gpu_results;
}

#ifdef __cplusplus
}
#endif