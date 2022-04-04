/*
 * popcount.h
 *
 *  Created on: Nov 7, 2013
 *      Author: hxin
 */

#ifndef POPCOUNT_H_
#define POPCOUNT_H_

#ifndef align_16
	#define align_16 __attribute__((aligned(16)))
#endif

#include <stdint.h>
#include <nmmintrin.h>

#ifdef __cplusplus
extern "C" {
#endif
uint32_t popcount1_m128i_sse(__m128i reg);

uint32_t popcount11_m128i_sse(__m128i reg);

uint32_t popcount1_sse(uint8_t* buffer, int chunks16);

uint32_t popcount11_sse(uint8_t* buffer, int chunks16);

uint32_t builtin_popcount(uint8_t* buffer, int chunks16);

uint32_t popcount1(uint8_t *buffer, int chunks16);

uint32_t popcount11(uint8_t *buffer, int chunks16);
#ifdef __cplusplus
}
#endif
#endif /* POPCOUNT_H_ */
