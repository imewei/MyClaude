
CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -fopenmp
LIBS = -lmpi

all: simulation

simulation: simulation.cpp
	$(CXX) $(CXXFLAGS) -o simulation simulation.cpp $(LIBS)

clean:
	rm -f simulation
