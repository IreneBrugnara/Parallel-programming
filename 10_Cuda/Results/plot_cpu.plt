set terminal png  # size 1280,800
set style data histograms
set style histogram rowstacked
set boxwidth 0.5
set style fill solid 0.5 border -1
set xlabel '# nodes'
#set yrange [0:110]
set linetype 1 lc rgb "yellow"
set linetype 2 lc rgb "red"

FILENAME="time_N10000_cpu.txt"
set title "N=10000 on CPU"
set output "plot_N10000_cpu.png"

plot FILENAME using 2:xtic(1) title "comm.", '' using 3 title "total - comm."
