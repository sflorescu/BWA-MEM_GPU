# BWA-MEM_GPUSEED_GASAL2 - A GPU accelerated implementation of BWA-MEM using the GPUSeed and GASAL2 libraries #

## Introduction

**BWA-MEM_GPUSEED_GASAL2** is a modified implementation of **BWA-MEM** ([https://bio-bwa.sourceforge.net/](https://bio-bwa.sourceforge.net/)) that uses two GPU-accelerated libraries to perform both the seeding and extension parts on CUDA enabled GPUs.

For our datasets, we found that our implementation showed speedups between 2× and 2.8× over the baseline program for multithreaded execution. For the alignment results, we found that our integrated program has around 2% of lines in the main result different from the baseline program, and if we filter alignments with mapping quality below 20, the percentage of different lines reduces to around 1%. 

<!-- GETTING STARTED -->
## Getting Started

This is a paragraph of text.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Nvidia GPU preferably with compute 60 or newer and at least 8GB of VRAM. 
* CUDA Toolkit 11.0+
* GCC 9+
* G++ 9+

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/sflorescu/BWA-MEM_GPUSeed_GASAL2.git
   ```
2. Initialize and load GASAL2 submodule
   ```sh
   git submodule init GASAL2/
   git submodule update GASAL2/
   ```
3. Configure and compile GASAL2 library
   ```sh
   cd GASAL2/
   ./configure.sh <path to cuda installation directory>
   make GPU_SM_ARCH=<GPU SM architecture> MAX_SEQ_LEN=<maximum sequence length> N_CODE=<code for "N", e.g. 0x4E if the bases are represented by ASCII characters> [N_PENALTY=<penalty for aligning "N" against any other base]
   ```
   Example for compute 75 (i.e., RTX 2080 Ti):
   ```sh   
   make GPU_SM_ARCH=sm_75 MAX_SEQ_LEN=300 N_CODE=4 N_PENALTY=1
   ```
   For more information on the GASAL2 compilation parameters check the GASAL2 [README](GASAL2/README.md)
4. Configure GPUSeed library for specific GPU architecture by moving to the GPUSeed directory
   ```sh
   cd src/GPUSeed/
   ```
   Then change line 4 of the GPUSeed **Makefile** to your specific GPU architecture
   ```sh
   GPU_SM_ARCH=sm_XX
   ```
   
   Example for compute 75 (i.e., RTX 2080 Ti):
      ```sh
   GPU_SM_ARCH=sm_75
   ```
5. In the root directory of the repository change the 5th line according to the path of your CUDA toolkit installation
   ```sh
   CUDA_LIB_DIR=<path to cuda installation directory>
   ```
   
   Example:
      ```sh
   CUDA_LIB_DIR=/usr/local/cuda-11.0/lib64/
   ```
6. Compile complete project by running 
   ```sh
   make all
   ```

### Building the FMD-index

The GPUSeed library uses a modified FMD-index to find seeds of the query in the reference. In order to build the index of the reference run the following command in the root directory of the repo.
```sh
./build_index.sh <path to reference fasta file>
```
This step is only required once for each new reference file.

## Running the library

In order to perform alignments using the accelerated library we use the following command:
```sh
./bwa-gasal2 gase_aln -t <number of CPU threads> -l <query length> <reference fasta file>  <query fasta file>
```

The following command can be used to see a list of all available alignment options:
```sh
./bwa-gasal2 gase_aln -l 1
```

### Current Limitations ###

* Maximum number of queries that can be processed in one run depends on system's RAM.
* Maximum number of CPU threads that can be used depends on GPU VRAM.
* Library only accepts queries in the FASTA format.
* BWA-MEM like re-seeding is not performed.

   
## Related Libraries ##
BWA-MEM: <https://github.com/lh3/bwa>

GPUSeed: <https://github.com/nahmedraja/GPUseed>

GASAL2 with BWA-MEM <https://github.com/j-levy/bwa-gasal2>

## Further Reading ##
This library was developed as part of a Master's degree thesis at the Delft University of Technology within the Accelerated Big Data Systems group. Since we are currently working on a paper regarding this project we urge the interested reader to take a look at the complete thesis for further information. The thesis which presents in detail the work performed in the context of this project can be found at the following [link](https://repository.tudelft.nl/islandora/object/uuid:4dd99ea2-6955-4e39-8e40-4198da4667f4).