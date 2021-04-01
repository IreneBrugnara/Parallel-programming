set terminal png size 1280,800 font Courier
set output "scal_N1200_nonblocking.png"
set title "N=1200 non-blocking"
set xlabel 'P'
set ylabel 'time to solution'
set ytics nomirror
set y2tics
set y2label 'scalability'
set y2range [20:160]

getValue(row,col,filename) = system('awk ''{if (NR == '.row.') print $'.col.'}'' '.filename.'')
filename="time_N1200_nonblocking.dat"
first_val=getValue(2,2,filename)+getValue(2,3,filename)

plot filename using 1:($2+$3) title "T(P)" with linespoints pointtype 7 axis x1y1, \
     filename using 1:((first_val*20/($2+$3))) title "T(1)/T(P)" with linespoints pointtype 6 axes x1y2

