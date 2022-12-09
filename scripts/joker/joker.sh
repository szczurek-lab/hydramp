#!/bin/bash

HYDRAMP_PATH="/home/paulina/AMP/hydramp"
JOKER_PATH="/home/paulina/AMP/Joker"
DATA_FOLDER="temp_data"

$JOKER_PATH/joker -i $HYDRAMP_PATH/$DATA_FOLDER/positives_test.fasta -p $JOKER_PATH/example/pattern.pat -o $HYDRAMP_PATH/$DATA_FOLDER/positives_test_joker_output.fasta
$JOKER_PATH/joker -i $HYDRAMP_PATH/$DATA_FOLDER/negatives_test.fasta -p $JOKER_PATH/example/pattern.pat -o $HYDRAMP_PATH/$DATA_FOLDER/negatives_test_joker_output.fasta

awk 'BEGIN{RS=">"}{print $1"\t"$2;}' $HYDRAMP_PATH/$DATA_FOLDER/positives_test_joker_output.fasta | tail -n+2 > $HYDRAMP_PATH/$DATA_FOLDER/positives_test_joker_output.tsv
awk 'BEGIN{RS=">"}{print $1"\t"$2;}' $HYDRAMP_PATH/$DATA_FOLDER/negatives_test_joker_output.fasta | tail -n+2 > $HYDRAMP_PATH/$DATA_FOLDER/negatives_test_joker_output.tsv
