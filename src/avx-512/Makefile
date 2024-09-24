CC     = gcc-9 
SRCS   = fp12_avx.c test.c
FLAGS  = -O2 -mavx512ifma -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native
OUTPUT = test 

all:
	$(CC) $(SRCS) -o $(OUTPUT) $(FLAGS) -DPROFILING

clean:
	rm -r test
