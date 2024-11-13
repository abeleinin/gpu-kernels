CUDA_COMPILER = nvcc
CUDA_FLAGS = 
TARGET = naive_matmul
SRC = naive_matmul.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
