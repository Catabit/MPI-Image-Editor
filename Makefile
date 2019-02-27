FILT = sharpen
COL = bw	
NP = 4

build: homework
homework: homework.c homework.h
	mpicc -o homework homework.c homework.h -lm -Wall -g
test: homework homework.c
	mpirun -np $(NP) homework in/lenna_bw.pgm output.pgm $(FILT)
	compare -compose src output.pgm ref/lenna_bw_$(FILT).pgm diff.pgm &>/dev/null
	diff -q output.pgm ref/lenna_bw_$(FILT).pgm
test_color: homework homework.c
	mpirun -np $(NP) homework in/lenna_color.pnm output.pnm $(FILT)
	compare -compose src output.pnm ref/lenna_color_$(FILT).pnm diff.pnm &>/dev/null
	diff -q output.pnm ref/lenna_color_$(FILT).pnm

clean:
	rm -f homework output.pgm

valgrind: homework
	mpiexec -n $(NP) valgrind --leak-check=yes ./homework in/lenna_$(COL).pgm output.pgm $(FILT)

valgrind2:homework
	mpiexec -n $(NP) valgrind --leak-check=yes --show-reachable=yes --log-file=nc.vg.%p ./homework in/lenna_$(COL).pgm output.pgm $(FILT)

compare:
	compare -compose src output.pgm ref/lenna_$(COL)_$(FILT).pgm diff.pgm &>/dev/null
	diff -q output.pgm ref/lenna_$(COL)_$(FILT).pgm
