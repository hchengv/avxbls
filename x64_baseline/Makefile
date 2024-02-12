CC     = clang 
SRCS   = $(wildcard *.c) assembly.S rdtsc64.S
FLAGS  = -O0 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native
OUTPUT = pairing 

all:
	$(CC) $(SRCS) -o $(OUTPUT) $(FLAGS) -DPROFILING

clean:
	rm -r pairing 