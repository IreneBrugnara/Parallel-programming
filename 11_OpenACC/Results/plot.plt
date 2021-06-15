set terminal png font "arial" # size 1280,800
set style data histograms
set style histogram rowstacked
set boxwidth 0.5
set style fill solid 0.5 border -1
set xlabel '# nodes'
set yrange [0:2]
set linetype 1 lc rgb "orange"
set linetype 2 lc rgb "purple"

FILENAME="time_N10000.txt"
set title "n=10000 with OpenACC"
set output "plot_N10000_openacc.png"

plot FILENAME using ($2-$3):xtic(1) title "total - comm.", '' using 3 title "comm."
