#!/usr/local/bin/bash

count=0
listfile="/tmp/listfile.txt"
rm -f ${listfile}
amount_of_files_to_keep=500

cd $(dirname $0)

for f in ../pictures/image/*.png; do
  count=$(( ${count} + 1 ))
  echo "${f}" >> "${listfile}"
done

if [ "${count}" -gt "${amount_of_files_to_keep}" ]; then

  difference=$(( $count - $amount_of_files_to_keep ))
  files_to_remove=$(cat ${listfile} | sort | head -n "${difference}")
  arr=(${files_to_remove})

  for i in "${arr[@]}"; do
    rm -f "${i}"
  done

fi
