CC     = clang 
SRCS   = $(wildcard *.c) asm/add_mod_384.S
FLAGS  = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native
OUTPUT = pairing 

all:
	$(CC) $(SRCS) -o $(OUTPUT) $(FLAGS)

clean:
	rm -r pairing 