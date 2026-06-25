CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2
TARGET   := autograd_engine

SRCS := src/Tensor.cpp        \
        src/GraphEngine.cpp   \
        src/MemoryPool.cpp    \
        src/Optimizer.cpp     \
        src/ops/Add.cpp       \
        src/ops/MatMul.cpp    \
        src/ops/ReLU.cpp      \
        src/ops/MSELoss.cpp   \
        main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
