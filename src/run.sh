#!/bin/sh

EX=../data/E1

#run with default parameters: -L 3 -ucmT 0.1 -ransacT 2.6 -score SOD -fast -r 10
./flowsac $EX/src.png $EX/tar.png $EX/match.txt $EX/ucm.png $EX/sed.edge $EX/flow.flo sintel 


#run with fast mode
#./flowsac $EX/src.png $EX/tar.png $EX/match.txt $EX/ucm.png $EX/sed.edge $EX/flow.flo sintel -fast

