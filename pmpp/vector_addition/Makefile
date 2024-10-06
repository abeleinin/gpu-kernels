CUDA_COMPILER = nvcc
CUDA_FLAGS = 
TARGET = vector_addition
SRC = vector_addition.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
