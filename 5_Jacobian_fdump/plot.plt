set terminal png
set output "image.png"
unset colorbox
set palette rgb 33,13,10
set size square
set yrange [*:*] reverse
plot 'solution.txt' matrix with image
