CC     = clang 
SRCS   = $(wildcard *.c) assembly.S rdtsc64.S
FLAGS  = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native
OUTPUT = pairing 

all:
	$(CC) $(SRCS) -o $(OUTPUT) $(FLAGS)

clean:
	rm -r pairing 