#!/bin/bash

# Run cProfile to collect a file named test.prof in the local directory.
# Eg. python -m cProfile -o test.prof main.py


python -c "import pstats ; p = pstats.Stats('test.prof') ; p.strip_dirs().sort_stats(-1).print_stats()" > print_stats.out