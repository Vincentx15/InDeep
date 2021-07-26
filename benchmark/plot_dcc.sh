#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
# set -o noclobber  # prevent overwritting redirection

function usage () {
    cat << EOF
Help message
    -r, --rec input rec file with dcc values (default: dcc.rec)
    -h, --help print this help message and exit
EOF
}

INREC='dcc.rec'  # Default value
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--rec) INREC="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

OUTDIR='DCC_figs'
[ -d $OUTDIR ] || mkdir $OUTDIR

function plot_success() {
TITLE=$1
XMAX=$2
PRINTYLABEL=$3
PRINTLEGEND=$4
    np -d',' "
A[A==''] = 100
A[A=='\n'] = 100
n = A.shape[0]
print(n)
fig, ax = plt.subplots()
ax.hist(A, bins=np.linspace(1, $XMAX, $XMAX)-.5, cumulative=True,
        label=['InDeep apo', 'Kalasanty apo', 'InDeep holo', 'Kalasanty holo'])
ax.grid()
ax.set_yticks([n*i for i in np.linspace(0, 0.6, 6)])
fig.canvas.draw()
ylabels = np.asarray([float(item.get_text()) for item in ax.get_yticklabels()])
print(ylabels)
ax.set_yticklabels([ f'{e:.2f}' for e in ylabels / n])
plt.xlabel('DCC (Å)')
if $PRINTYLABEL == 1:
    plt.ylabel('Success rate (SR)')
else:
    plt.ylabel('SR')
if $PRINTLEGEND == 1:
    plt.legend()
plt.title(f'$TITLE')
plt.savefig('$OUTDIR/$(echo $TITLE[1]).pdf')"
}


XMAX=7

paste -d',' =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_apo_cav_indeep:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_apo_cav_kala:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_holo_cav_indeep:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_holo_cav_kala:/{print $2}')  \
            | plot_success 'A. Volsite cavities' $XMAX 1 1

paste -d',' =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_apo_lig_indeep:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_apo_lig_kala:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_holo_lig_indeep:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_holo_lig_kala:/{print $2}')  \
            | plot_success 'B. Ligands' $XMAX 0 0
 
paste -d',' =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_apo_cavlig_indeep:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_apo_cavlig_kala:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_holo_cavlig_indeep:/{print $2}') \
            =(sed 's/_\([0-9]\+\)//g' $INREC | awk '/dcc_holo_cavlig_kala:/{print $2}')  \
            | plot_success 'C. Ligands with VolSite cavity' $XMAX 0 0

for PDF in $(ls $OUTDIR/*.pdf); do
    pdfcrop $PDF $PDF
done

cat << EOF > $OUTDIR/dcc.tex
\documentclass{article}[10pt]
\usepackage{caption}
\usepackage{graphicx}
\usepackage{subcaption}
\captionsetup[subfigure]{labelformat=empty}
\begin{document}
\pagenumbering{gobble}
\begin{figure}
    \begin{subfigure}[b]{0.3\textwidth}
        \includegraphics[width=\textwidth]{A.pdf}
    \end{subfigure}
    \begin{subfigure}[b]{0.3\textwidth}
        \includegraphics[width=\textwidth]{B.pdf}
    \end{subfigure}
    \begin{subfigure}[b]{0.3\textwidth}
        \includegraphics[width=\textwidth]{C.pdf}
    \end{subfigure}
\end{figure}
\end{document}
EOF

cd $OUTDIR
pdflatex dcc.tex
pdfcrop dcc.pdf dcc.pdf
