set terminal png font "arial"
set style data histograms
set style histogram rowstacked
set boxwidth 0.5
set style fill solid 0.5 border -1
set xlabel '# nodes'
#set yrange [0:80]
set linetype 1 lc rgb "red"
set linetype 2 lc rgb "blue"

if (ARG3 eq "serial") LEVEL="MPI-only"; else LEVEL="MPI+openMP"
FILENAME=sprintf("time_%s_%s_%s.txt",ARG1,ARG2,ARG3)
TITLE=sprintf("n=%s, %s, %s", ARG1,ARG2,LEVEL)
OUTPUT=sprintf("plot_%s_%s_%s.png",ARG1,ARG2,ARG3)
set title TITLE
set output OUTPUT

plot FILENAME using 3:xtic(1) title "computation", '' using 2 title "communication"
