CC      = gcc
SRCS    = $(wildcard ./src/*.c) $(wildcard ./src/*.S)
FLAGS   = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native -z noexecstack
OUT_DIR = ./bin

all: test bench

test:
	$(CC) $(SRCS) -o $(OUT_DIR)/test      $(FLAGS) -DTEST

bench:
	$(CC) $(SRCS) -o $(OUT_DIR)/bench     $(FLAGS) -DBENCHMARK

clean:
	rm -r $(OUT_DIR)/*
