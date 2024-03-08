#!/bin/bash
DIR='output'
FILE='data_log.log'

# Loop over the directories in DIR
for directory in "$DIR"/*; do
    if [ -d "$directory" ]; then
        # Execute model counting on every FILE in the directory
        echo
        echo "Executing model count on $directory/$FILE"
        python3 src/model_counting.py $directory/$FILE
    fi
done
