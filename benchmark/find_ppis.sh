#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

# set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

function usage () {
    cat << EOF
Find all PPIs for PDBs in holo directory of the Chen Benchmark
    -h, --help print this help message and exit
    -r, --rec recfile containing homologous PDB codes as generated
              by find_homologous.sh (default: holo_homologs.rec)
EOF
}

REC=holo_homologs.rec
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--rec) REC="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

N=$(recinf $REC)
for I in $(seq 0 $((N-1))); do
    ENTRY=$(recsel -n $I $REC)
    PDB=$(echo $ENTRY |  recsel -P 'pdb')
    HOMOLOGS=$(echo $ENTRY | recsel -P 'homo' | tr '\n' ' ' | sed 's/ $//')
    # echo "./find_ppi.py -p $PDB --homo $HOMOLOGS"
    if [ ! -z $HOMOLOGS ]; then
        ./find_ppi.py -p $PDB --homo $(echo $HOMOLOGS)
    fi
done
