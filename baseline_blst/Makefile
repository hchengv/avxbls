CC     = gcc-9 
SRCS   = $(wildcard *.c) assembly.S rdtsc64.S
FLAGS  = -O2 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native
OUTPUT = test 

all:
	$(CC) $(SRCS) -o $(OUTPUT) $(FLAGS) -DPROFILING

clean:
	rm -r pairing 
