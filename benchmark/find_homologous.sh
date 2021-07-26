#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
EOF
}

N=1  # Default value
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done


for PDB in holo/*.pdb; do
    echo ""
    echo "pdb: $PDB"
    HOMOLOGS=$(pdbsearch-seq -f =(pdb2fasta $PDB 'polymer.protein' | sed 's/?//g') -n 1000 | grep -v "input_sequence: " | awk '{if (NR>1){print "+ "$1} else {print "homo: "$1}}')
    echo $HOMOLOGS
done > holo_homologs.rec
