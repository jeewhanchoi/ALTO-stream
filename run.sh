#!/bin/bash

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux

export OMP_NUM_THREADS=56
export KMP_AFFINITY=granularity=fine,compact,1,0

function special_execute() { 
    [ "$1" -eq "-123" ] && echo "flagY" || echo "flagN"; 
    shift; 
    set -x; "$@"; set x;
  }

function get_path_to_tensor() {
    declare -A path_to_tensors=(
        ["flickr"]="~/hpctensor/flickr-4d.tns"
        ["chicago"]="~/hpctensor/chicago-crime-comm.tns"
        ["uber"]="~/hpctensor/uber.tns"
        ["deli"]="~/hpctensor/delicious-4d.tns"
        ["patents"]="~/hpctensor/patents.tns"
        ["lbnl"]="~/hpctensor/lbnl-network.tns"
        ["nips"]="~/hpctensor/nips.tns"
        ["lanl"]="~/hpctensor/lanl_one.tns"

    )
    echo "${path_to_tensors[${1}]}"
}

function get_streaming_mode() {
  declare -A streaming_mode=(
    ["uber"]="0"
    ["chicago"]="0"
    ["flickr"]="3" 
    ["deli"]="3" 
    ["patents"]="0" 
    ["lbnl"]="4"
    ["nips"]="3"
    ["lanl"]="0"
  )
  echo "${streaming_mode[${1}]}"
}

function get_constraint() {
    declare -A constraint_type=(
        ["nonneg"]="-z nonneg"
        ["l1"]="-z l1"
        ["simplex"]="-z simplex"
    )

    [[ -z "$1" ]] &&  echo "" || echo "${constraint_type[${1}]}"
}

make

declare -a random_seeds=(55 35 62 2245 1251321)
#declare -a random_seeds=(1251321)

make

for seed in ${random_seeds[@]}; do
    cmd="./cpd128 --rank 16 --epsilon 1e-5 -x $seed -m 50 -a $(get_streaming_mode $2) -l $1 $(get_constraint $3) -i $(get_path_to_tensor $2)"
    echo $cmd
    eval "$cmd"
done

#for seed in ${random_seeds[@]}; do
#    cmd="./cpd128 --rank 16 --epsilon 1e-5 --timeslice-limit 16 --nnz-threshold 1000000 -x $seed -m 50 -a $(get_streaming_mode $2) -l $1 $(get_constraint $3) -i $(get_path_to_tensor $2)"
#    echo $cmd
#    eval "$cmd"
#done
