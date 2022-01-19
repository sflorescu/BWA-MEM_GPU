OBJ_DIR=./obj/
SRC_DIR=./src/
LIB_DIR=./libs/
LOBJS= seed_gen.o
LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
GPUSEED_INCLUDE_DIR = ./include/
INCLUDES= $(GPUSEED_INCLUDE_DIR)
AR=ar

object: makedir
	nvcc -c -O3 -std=c++11 -I ./cub/cub -lineinfo -Xcompiler -Wall -Xptxas  -Werror -rdc=true --gpu-architecture=compute_61 --gpu-code=sm_61 --ptxas-options=-v -o $(OBJ_DIR)/seed_gen.o seed_gen.cu

libseed.a: object
		echo $(LOBJS_PATH)
		$(AR) -csr $(LIB_DIR)$@ $(LOBJS_PATH)

clean:
	rm -rf *.exe *.o obj/ libs/ include/

makedir:
	@mkdir -p $(INCLUDES)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(OBJ_DIR)
	@cp *.h $(INCLUDES)