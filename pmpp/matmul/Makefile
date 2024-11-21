# Compiler and flags
NVCC = nvcc
CFLAGS = -I./include -O2

# Source directory and output directory
SRC_DIR = src
OUT_DIR = bin

# Get all CUDA source files in the source directory
SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Create an output executable for each source file
EXECUTABLES = $(patsubst $(SRC_DIR)/%.cu, $(OUT_DIR)/%, $(SOURCES))

# Default target
all: $(EXECUTABLES)

# Rule to compile each CUDA source file into an executable
$(OUT_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(OUT_DIR)
	$(NVCC) $(CFLAGS) $< -o $@

# Clean target to remove compiled files
clean:
	rm -rf $(OUT_DIR)
