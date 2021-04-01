set terminal png size 1280,800 font Courier
set output "com_N1200_nonblocking.png"
set style data histograms
set style histogram rowstacked
set boxwidth 0.5
set style fill solid border -1
set title "N=1200 non-blocking"
set xlabel 'P'

filename="time_N1200_nonblocking.dat"
plot filename using 2:xtic(1) title "communication", '' using 3 title "computation"
