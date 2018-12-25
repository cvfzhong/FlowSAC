#!/bin/sh

# name of the dataset ( sintel or kitti), some parameters will be set according to the dataset name
DATASET=kitti

# estimation method (flowsac | epicflow | ricflow)
METHOD=flowsac

#use this line for flowsac-fast
#METHOD="flowsac -fast"

#optional set more parameters
#METHOD="flowsac -fast -r 10 "

#list of flow pairs
SUBSET=x
#LIST=../data/$DATASET-train/list-$SUBSET.txt
LIST=/fan/data/FlowSAC/data/$DATASET-train/list-$SUBSET.txt

#file name for writing the results
OUTFILE=../data/results/$DATASET-[$METHOD]-$SUBSET


#run for the four matching methods: flowfields, cpm, dcflow, and discreteflow

echo flowfields "($DATASET-$SUBSET-[$METHOD])"
./batch $LIST flowfields $DATASET -o "$OUTFILE-ff.txt" -method $METHOD

echo cpm "($DATASET-$SUBSET-[$METHOD])"
./batch $LIST cpm $DATASET -o "$OUTFILE-cpm.txt" -method $METHOD 

echo dcflow "($DATASET-$SUBSET-[$METHOD])"
./batch $LIST dcflow $DATASET -o "$OUTFILE-dcf.txt" -method $METHOD

echo discreteflow "($DATASET-$SUBSET-[$METHOD])"
./batch $LIST discreteflow $DATASET -o "$OUTFILE-df.txt" -method $METHOD


