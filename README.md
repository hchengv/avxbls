# AVXBLS: fast AVX-512 implementation of the optimal ate pairing on BLS12-381

<!--- ==================================================================== --->

## Overview 

AVXBLS is a software library containing a latency-optimized AVX-512
implementation of the optimal ate pairing on BLS12-381 curve, and is based on
the widely-used [BLST](https://github.com/supranational/blst) library.

<!--- ==================================================================== --->

## Usage 

- Clone the repo:

  ```sh
  git clone https://github.com/hchengv/avxbls ./avxbls
  cd ./avxbls
  ```

- Compile the AVX-512 implementation: 

  ```sh
  make
  ```

- Execute test and benchmark: 

  ```sh
  ./bin/test
  ./bin/bench
  ```

<!--- ==================================================================== --->
