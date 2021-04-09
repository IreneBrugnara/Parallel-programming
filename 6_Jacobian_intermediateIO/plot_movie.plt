unset colorbox
set palette rgb 33,13,10
set size square
set yrange [*:*] reverse
FILES = system("ls -1 solution_it*.txt")

do for [data in FILES] { plot data matrix with image; pause 0.5 }

