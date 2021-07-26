#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

function usage () {
    cat << EOF
Help message
    -t, --thr distance threshold in A to consider the ligand as an IPPI
    -r, --rec input recfile to process (default: holo_ppis.rec)
    -h, --help print this help message and exit
EOF
}

REC=holo_ppis.rec
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--thr) THR="$2"; shift ;;
        -r|--rec) REC="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done


echo "threshold: $THR"
echo ""
N=$(recinf $REC)
for I in $(seq 0 $((N-1))); do
    ENTRY=$(recsel -n $I $REC)
    # echo $ENTRY
    PDB=$(echo $ENTRY | recsel -p pdb)
    LIGANDS=$(echo $ENTRY | recsel -P ligand | sort -u)
    echo $PDB
    for LIGAND in $(echo $LIGANDS | tr ' ' '_'); do
        LIGCLASHES=$(echo $ENTRY | grep -A1 "$(echo $LIGAND | tr '_' ' ')" | grep ligclash | awk '{print $2}')
        ISPPI=$(echo $LIGCLASHES | awk -v"THR=$THR" 'BEGIN{OUT=0} {if ($1<THR){OUT=1}} END{print OUT}')
        echo "ligand: $LIGAND"
        echo "IPPI: $ISPPI"
    done
    echo ""
done
