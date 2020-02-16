//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <immintrin.h>
#include <iostream>

using namespace std;
// This code sample performs dot-product operations on 8-bit operands.
// The dot-product operation is performed first using a single fused instruction,
// followed by the operation using a sequence of 3 instructions.
 
int main() {

   uint8_t  op1_int8[64];
   int8_t   op2_int8[64];
   int32_t  op3_int[16];
   int16_t  op4_int16[32];
   int32_t  result[16];

   __m512i v1_int8;
   __m512i v2_int8;
   __m512i v3_int;
   __m512i v4_int16;
   __m512i vresult;

   // Choose some sample values
   for (int i = 0; i < 64; i++)
   {
       op1_int8[i]  = 3;
       op2_int8[i]  = 4;
   }
   for (int i = 0; i < 16; i++)
   {
       op3_int[i]   = 1;
   }
   for (int i = 0; i < 32; i++)
   {
       op4_int16[i] = 1;
   }

   //Load 512-bits of integer data
   v1_int8 =_mm512_loadu_si512(op1_int8);
   v2_int8 =_mm512_loadu_si512(op2_int8);
   v3_int =_mm512_loadu_si512(op3_int);
   v4_int16 =_mm512_loadu_si512(op4_int16);

   // PERFORM THE DOT PRODUCT OPERATION USING FUSED INSTRUCTION
   vresult = _mm512_dpbusds_epi32(v3_int,v1_int8,v2_int8);

   _mm512_storeu_si512((void *) result, vresult);

   cout << endl;
   printf("RESULTS USING FUSED INSTRUCTION: \n");
   for (int j = 15; j >= 0; j--){
       cout << result[j]<<" ";
   }
   cout << endl;

   // PERFORM THE DOT PRODUCT OPERATION USING A SEQUENCE OF 3 INSTRUCTIONS

   // Vertically multiply two 8-bit integers,
   // then horizontally add adjacent pairs of 16-bit integers

   __m512i vresult1 = _mm512_maddubs_epi16(v1_int8,v2_int8);

   // Upconvert to 32-bit and horizontally add neighbors. Multiply by 1.
   __m512i vresult2 = _mm512_madd_epi16(vresult1,v4_int16);

   // Add packed 32-bit integers
   vresult = _mm512_add_epi32(vresult2, v3_int);

   _mm512_storeu_si512((void *) result, vresult);

   cout << endl;
   printf("RESULTS USING SEQUENCE OF 3 INSTRUCTIONS: \n");
   for (int j = 15; j >= 0; j--)
       cout << result[j]<<" ";
   cout << endl;
}
