CXX = g++-5
CXXFLAGS = -g -Wall -Wreorder -std=c++11
LDADD =

OBJ = main.o GAP.o Lagrangian.o LocalSearch.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o LagrHeu 

clean:
	rm -rf $(OBJ) LagrHeu

.PHONY: clean
