#!/usr/bin/bash

array=( msg gcam )
for i in "${array[@]}"
do
    echo $i
    aneris test-$i/inputfile.xlsx --history history.csv --regions test-$i/regiondef.xlsx --rc rc.yaml --output_path . --output_prefix $i
done
