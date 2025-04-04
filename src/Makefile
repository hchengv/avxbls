CC     = gcc
SRCS   = $(wildcard *.c) assembly.S rdtsc64.S
FLAGS  = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native -z noexecstack

profiling:
	$(CC) $(SRCS) -o profiling $(FLAGS) -DPROFILING

bench:
	$(CC) $(SRCS) -o bench $(FLAGS) -DBENCHMARK

all:
	$(CC) $(SRCS) -o test $(FLAGS)

clean:
	rm -r test bench profiling
