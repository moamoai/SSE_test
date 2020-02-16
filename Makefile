
sqrt: 
	g++ -msse2 -lm benchmark.cpp -o sqrt

test_avx: 
	gcc -mavx2 test_avx.c -o test_avx
