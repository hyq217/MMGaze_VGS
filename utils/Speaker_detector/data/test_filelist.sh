#!/bin/bash
# get all filename in specified path

# run： bash getFileName.sh Your_Folder_Path
 
path=$1
files=$(ls $path)
for filename in $files
do
 echo $filename >> filename.txt
done

