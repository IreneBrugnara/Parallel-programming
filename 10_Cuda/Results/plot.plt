set terminal png  # size 1280,800
set style data histograms
set style histogram rowstacked
set boxwidth 0.5
set style fill solid 0.5 border -1
set xlabel '# nodes'
#set yrange [0:3]
set linetype 1 lc rgb "green"
set linetype 2 lc rgb "blue"

FILENAME="time_N10000_gpu.txt"
set title "N=10000 on GPU"
set output "plot_N10000_gpu_zoomed.png"

plot FILENAME using 3:xtic(1) title "comm.", '' using ($2-$3) title "total - comm."
