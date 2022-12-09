#!/bin/bash
N=8
END=18894 # Number of peptides in DBAASP as per 02.03.2022
DIR=../../temp_data/json

mkdir -p $DIR


for i in $(seq 1 $END); do 
	( 
	curl -X GET "https://dbaasp.org/peptides/$i" -H "accept: application/json">$DIR/$i.json 
	if [[ "$?" -eq 0 ]]; then
		exit 0
	fi
	) &
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
done

wait

echo "all done"

find $DIR  -size 0 -print -delete

