# DIM variable must be passed as parameter to this script
# example: gnuplot -e "DIM=1200" plot_binary.plt
set terminal png
set output "image_binary.png"
unset colorbox
set palette rgb 33,13,10
set size square
plot 'solution.dat' binary array=(DIM+2,DIM+2) format="%double" flipy with image
