; Sample pen-plotter job: a 60 mm square with a circle inside.
G21 ; millimetres
G90 ; absolute
G0 Z5
G0 X0 Y0
G1 Z0
G1 X60 Y0
G1 X60 Y60
G1 X0 Y60
G1 X0 Y0
G0 Z5
G0 X30 Y15
G1 Z0
G2 X30 Y15 I0 J15
G0 Z5
