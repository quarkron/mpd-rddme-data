# Single-core port. `make` checks the 500-line cap on every file here, then builds lmcpu.
# No -fopenmp: the column is one core, and the binary cannot use more.
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter
SRC := lattice.cpp scene_io.cpp bath_hop.cpp structure.cpp smear.cpp body.cpp \
       chain_common.cpp chain_moves.cpp chain_kinetics.cpp main.cpp
HDR := $(wildcard *.h)

all: linecheck lmcpu

lmcpu: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

# every file in this folder, sources, headers and scripts alike
linecheck:
	@fail=0; for f in $$(ls *.cpp *.h *.py *.sh Makefile README.md 2>/dev/null); do \
	  n=$$(wc -l < $$f); if [ $$n -gt 500 ]; then echo "LINECAP: $$f has $$n lines (> 500)"; fail=1; fi; done; \
	  [ $$fail -eq 0 ] || exit 1; echo "linecheck: every file <= 500 lines"

clean:
	rm -f lmcpu

.PHONY: all linecheck clean
