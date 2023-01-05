#makefile

include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

all: naive float mixed

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_75 ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_75 ex_particle_CUDA_float_seq.cu -o particlefilter_float

mixed: ex_particle_CUDA_mixed_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O0 -use_fast_math -arch sm_75 ex_particle_CUDA_mixed_seq.cu -o particlefilter_mixed
	
trace: ex_particle_CUDA_mixed_seq.cu
	$(CC) -DTRACE -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_75 ex_particle_CUDA_mixed_seq.cu -o particlefilter_mixed

clean:
	rm particlefilter_naive particlefilter_float particlefilter_mixed
