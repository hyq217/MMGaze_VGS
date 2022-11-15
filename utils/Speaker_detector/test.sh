#!/bin/bash

# videofile=$1
path=$1

# out_dir=$2
files=$(ls $path)

for videofile in $files
do
   	echo "==============================Now is working on file:====================="
   	echo $videofile
        a = $(echo $videofile | awk  '{ string=substr($0,0,3); print string; }')
        echo $a

 
done



