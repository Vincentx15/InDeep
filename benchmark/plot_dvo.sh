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
    -r|--rec, recfile
    -t|--type, type of prediction to plot: cav, cavlig or lig
    -c|--cutoff, cutoff in Angstrom for the DCC
    -h, --help print this help message and exit
EOF
}

RECFILE=chen_scores.rec  # Default value
CUTOFF=4
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -r|--rec) RECFILE="$2"; shift ;;
        -t|--type) TYPE="$2"; shift ;;
        -c|--cutoff) CUTOFF="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

AWKCMDKALA='{if (DOPRINT==1){print $2; DOPRINT=0}} /dcc_holo_'$TYPE'_kala_[0-9]*:/{if ($2<'$CUTOFF'){DOPRINT=1}}'
AWKCMDINDEEP='{if (DOPRINT==1){print $2; DOPRINT=0}} /dcc_holo_'$TYPE'_indeep_[0-9]*:/{if ($2<'$CUTOFF'){DOPRINT=1}}'
paste -d',' \
    =(awk -v TYPE=$TYPE $AWKCMDKALA $RECFILE) \
    =(awk -v TYPE=$TYPE $AWKCMDINDEEP $RECFILE) \
    | np -d',' "
A[A==''] = np.inf
A[A=='\n'] = np.inf
plt.hist(A, bins=20, range=(0,1), label=('kala', 'indeep'))
plt.legend()
plt.grid()
plt.show()
"
