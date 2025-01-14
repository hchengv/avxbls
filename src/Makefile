CC     = gcc-9 
SRCS   = $(wildcard *.c) assembly.S rdtsc64.S
FLAGS  = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native -z noexecstack
OUTPUT = test 

all:
	$(CC) $(SRCS) -o $(OUTPUT) $(FLAGS)

clean:
	rm -r test 
