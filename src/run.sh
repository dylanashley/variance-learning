#!/bin/sh

POWERS_OF_TWO=(0.0000305176
               0.0000610352
               0.0001220703
               0.0002441406
               0.0004882812
               0.0009765625
               0.0019531250
               0.0039062500
               0.0078125000
               0.0156250000
               0.0312500000
               0.0625000000
               0.1250000000
               0.2500000000
               0.5000000000
               1.0000000000)

# assert command line arguments valid
if [ "$#" -gt "1" ]
    then
        echo 'usage: ./run.sh [RESULTS_DIR]'
        exit
    fi

# get folder name for results
if [ "$#" == "1" ]
    then
        RESULTS_DIR=$1
    else
        RESULTS_DIR=$(date +%Y-%m-%dT%H:%M:%S%z)
    fi

# get location of scripts
RUN_SCRIPT=$(dirname $0)'/run.py'
PLOT_SCRIPT=$(dirname $0)'/plot.py'

# build run tasks file
CHECK_FILE_NAME='first_predictions.npy'
TASKS_FILE="$RESULTS_DIR"'/tasks.sh'
if [ ! -f $TASKS_FILE ]
    then
        DATA_DIR="$RESULTS_DIR"'/1'
        mkdir -p "$DATA_DIR" 2>/dev/null
        CHECK_FILE="$DATA_DIR"'/'"$CHECK_FILE_NAME"
        STDOUT_FILE="$DATA_DIR"'/stdout'
        STDERR_FILE="$DATA_DIR"'/stderr'
        ARGS="${POWERS_OF_TWO[4]} 0.95 ${POWERS_OF_TWO[6]} 0.95 --directory $DATA_DIR --seed-count 100 --step-count 3000000 --step-interval 30"
        echo "if [ ! -f $CHECK_FILE ]; then python $RUN_SCRIPT $ARGS > $STDOUT_FILE 2> $STDERR_FILE; fi" >> $TASKS_FILE

        for VAL_ALPHA in "${POWERS_OF_TWO[@]}"
            do
                VAR_ALPHA='0.0000000000'
                DATA_DIR="$RESULTS_DIR"'/0/'"$VAL_ALPHA"'/'"$VAR_ALPHA"
                mkdir -p "$DATA_DIR" 2>/dev/null
                CHECK_FILE="$DATA_DIR"'/'"$CHECK_FILE_NAME"
                STDOUT_FILE="$DATA_DIR"'/stdout'
                STDERR_FILE="$DATA_DIR"'/stderr'
                ARGS="$VAL_ALPHA 0.95 $VAR_ALPHA 0.95 --directory $DATA_DIR --seed-count 30 --step-count 3000000 --step-interval 3000"
                echo "if [ ! -f $CHECK_FILE ]; then python $RUN_SCRIPT $ARGS > $STDOUT_FILE 2> $STDERR_FILE; fi" >> $TASKS_FILE
            done

        for VAR_ALPHA in "${POWERS_OF_TWO[@]}"
            do

                DATA_DIR="$RESULTS_DIR"'/0/'"${POWERS_OF_TWO[4]}"'/'"$VAR_ALPHA"
                mkdir -p "$DATA_DIR" 2>/dev/null
                CHECK_FILE="$DATA_DIR"'/'"$CHECK_FILE_NAME"
                STDOUT_FILE="$DATA_DIR"'/stdout'
                STDERR_FILE="$DATA_DIR"'/stderr'
                ARGS="${POWERS_OF_TWO[4]} 0.95 $VAR_ALPHA 0.95 --directory $DATA_DIR --seed-count 30 --step-count 3000000 --step-interval 3000"
                echo "if [ ! -f $CHECK_FILE ]; then python $RUN_SCRIPT $ARGS > $STDOUT_FILE 2> $STDERR_FILE; fi" >> $TASKS_FILE
            done
    fi

# run tasks
parallel --verbose :::: $TASKS_FILE

# remove blank files
find $RESULTS_DIR -type f -empty -delete

# run plotting script
python $PLOT_SCRIPT $RESULTS_DIR
