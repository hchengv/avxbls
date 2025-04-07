# avxbls: an AVX-512 implementation of optimal ate pairing on BLS12-381

<!--- ==================================================================== --->

## Overview 

AVXBLS is based on [BLST](https://github.com/supranational/blst) and provides a
latency-optimized AVX-512 implementation of the optimal ate pairing on
BLS12-381. 

<!--- ==================================================================== --->

## Usage 

- Clone the repo:

  ```sh
  git clone https://github.com/ulhaocheng/avxbls ./avxbls
  cd ./avxbls
  ```

- Compile the AVX-512 implementation: 

  ```sh
  make
  ```

- Execute test, benchmark, and profiling: 

  ```sh
  ./bin/test
  ./bin/bench
  ./bin/profiling
  ```

<!--- ==================================================================== --->