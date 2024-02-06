#!/bin/bash

proc() {
 dir=$1
 ls $dir/*/*.out -1
 grep Summ $dir/*/*.out > $1.dat
}

for d in "$@"; do
  proc $d
done