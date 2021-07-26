#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# See: https://search.rcsb.org/#search-api

function usage () {
    cat << EOF
Retrieve PDB codes above a given sequence identity threshold from a given sequence
    -h, --help print this help message and exit
    -i, --id sequence identity cutoff (default: 0.9)
    -s, --seq sequence to search for
    -f, --fasta read the sequence from a fasta file instead of the -s option
    -n, --num maximum number of results (default: 100)
EOF
}

# SEQ=MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLPARTVETRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMNCKCVIS
ID=0.9
N=100
FASTA=None
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--seq) SEQ="$2"; shift ;;
        -f|--fasta) FASTA=$2; shift ;;
        -i|--id-cutoff) ID=$2; shift ;;
        -n|--num) N=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done
[[ $FASTA != None ]] && SEQ=$(grep -v '^>' $FASTA | sed '/^$/d' | tr -d '\n')
echo "input_sequence: $SEQ"

http https://search.rcsb.org/rcsbsearch/v1/query\?json\="{\"query\": {\"type\": \"terminal\",    \"service\": \"sequence\",    \"parameters\": {      \"evalue_cutoff\": 1,      \"identity_cutoff\": $ID,      \"target\": \"pdb_protein_sequence\",      \"value\": \"$SEQ\"  }  },  \"request_options\": {    \"scoring_strategy\": \"sequence\", \"pager\": {  \"start\": 0, \"rows\": $N}  },  \"return_type\": \"polymer_entity\"}" \
    | jq -r '.result_set[] | .identifier + " " + (.services[].nodes[].match_context[].sequence_identity|tostring)' \
    | awk -F'[_ ]' '{print $1,$3}' | sort -k2,2gr -k1,1
