#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

# set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
# set -o noclobber  # prevent overwritting redirection

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    --dcc rec file with DCC results (default: dcc.rec)
    --ippi rec file with IPPI info (ippi_rec/ippi_3.rec)
    --out output rec file (default: dcc_split.rec)
EOF
}

DCC=dcc.rec  # Default value
IPPI=ippi_rec/ippi_3.rec
OUTFILE=dcc_split.rec
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dcc) DCC="$2"; shift ;;
        --ippi) IPPI="$2"; shift ;;
        --out) OUTFILE="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done



N=$(recinf $DCC)
for I in $(seq 0 $((N-1))); do
    ENTRY=$(recsel -n $I $DCC)
    SCOP=$(echo $ENTRY | recsel -P scop)
    LIGMAP=$(grep $SCOP ligand_mapping.txt | awk -F_ '{print $2}' | sed 's/.mol2//' | awk '{if ($3!="****"){chain=$3}print $1,$4"_"$2"_"chain}')
    while read LIG; do
        echo "scop: $SCOP"
        LIGNAME=$(echo $LIG | awk '{print $2}')
        LIGID=$(echo $LIG | awk '{print $1}')
        recsel -e "pdb~'$SCOP'" $IPPI | grep -A1 "$LIGNAME"
        echo $ENTRY | grep "_$LIGID:" # | sed "s/_$LIGID//"
        echo ""
    done < =(echo $LIGMAP)
done > $OUTFILE
