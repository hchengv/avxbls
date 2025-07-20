CC      = gcc
SRCS    = $(wildcard *.c) $(wildcard *.S)
FLAGS   = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native -z noexecstack
OUT_DIR = ./bin

all: bench

bench: $(wildcard *.c) $(wildcard *.S)
	$(CC) $(SRCS) -o bench      $(FLAGS) -DTEST

clean:
	rm -r bench
