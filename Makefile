CC      = gcc
SRCS    = $(wildcard ./src/*.c) $(wildcard ./src/*.S)
FLAGS   = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native -z noexecstack
OUT_DIR = ./bin

all: test benchmark profiling

test:
	$(CC) $(SRCS) -o $(OUT_DIR)/test      $(FLAGS)

bench:
	$(CC) $(SRCS) -o $(OUT_DIR)/bench     $(FLAGS)

profiling:
	$(CC) $(SRCS) -o $(OUT_DIR)/profiling $(FLAGS)

clean:
	rm -r $(OUT_DIR)/*
