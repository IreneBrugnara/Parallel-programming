set terminal png
set output "image.png"
unset colorbox
set palette rgb 33,13,10
set size square
plot 'solution.dat' with image
