CC     = gcc
SRCS   = $(wildcard *.c) assembly.S rdtsc64.S
FLAGS  = -O3 -fomit-frame-pointer -m64 -mbmi2 -march=native -fwrapv -mtune=native -z noexecstack

ifeq "${MODE}" "benchmark"
FLAGS += -DBENCHMARK
endif 

ifeq "${MODE}" "profiling"
FLAGS += -DPROFILING
endif 

all: 
	$(CC) $(SRCS) -o test $(FLAGS) 

clean:
	rm -r test
