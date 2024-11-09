# avxbls: an AVX-512 implementation of optimal ate pairing on BLS12-381

<!--- ==================================================================== --->

## Overview 

AVXBLS is on the basis of [BLST](https://github.com/supranational/blst) and
provides a latency-optimized AVX-512 implementation of optimal ate pairing on
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
  cd ./src
  make  
  ```

- Test and benchmark: 

  ```sh
  ./src/test 
  ```

<!--- ==================================================================== --->