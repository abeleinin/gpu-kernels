# Define variables
CUDA_COMPILER = nvcc                # CUDA compiler
CUDA_FLAGS = -std=c++11             # CUDA flags (add any extra flags as needed)
JPEG_LIB = -ljpeg                   # JPEG library linking flag
LIB_PATH = /usr/lib                 # Library path (if needed)
INCLUDE_PATH = /usr/include         # Include path for headers (if needed)
TARGET = blur
SRC = blur.cu

# The default rule, when 'make' is run
all: $(TARGET)

# Compile the CUDA file into an executable, linking with libjpeg
$(TARGET): $(SRC)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $< -o $@ -L$(LIB_PATH) $(JPEG_LIB) -I$(INCLUDE_PATH)

# Clean the build files
clean:
	rm -f $(TARGET)
