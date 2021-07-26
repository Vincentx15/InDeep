#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails

INREC=$1

[ ! -d dccs ] && mkdir dccs
[ ! -d dcc_plots ] && mkdir dcc_plots

func plot_success () {
paste $1 $2 | np "A[A>20]=20; plt.title('$1 $2'); bins=np.linspace(0,20,100); plt.hist(A[:,0], label='InDeep', cumulative=True, histtype='step', bins=bins, normed=True); plt.hist(A[:,1], label='Kalasanty', cumulative=True, histtype='step', bins=bins, normed=True); plt.xlabel('DCC (Å)'); plt.legend(); plt.grid(); plt.savefig('dcc_plots/$1:t_$2:t.png')"
}

sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_apo' > dccs/indeep_apo
sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_apo_kala' > dccs/kala_apo

sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_holo' > dccs/indeep_holo
sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_holo_kala' > dccs/kala_holo

sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_apo_lig' > dccs/indeep_apo_lig
sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_apo_lig_kala' > dccs/kala_apo_lig

sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_holo_lig' > dccs/indeep_holo_lig
sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_holo_lig_kala' > dccs/kala_holo_lig

sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_apo_cavlig' > dccs/indeep_apo_cavlig
sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_apo_cavlig_kala' > dccs/kala_apo_cavlig

sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_holo_cavlig' > dccs/indeep_holo_cavlig
sed 's/_\([0-9]\+\)//g' $INREC | recsel -CP 'dcc_holo_cavlig_kala' > dccs/kala_holo_cavlig

plot_success dccs/indeep_apo dccs/kala_apo
plot_success dccs/indeep_holo dccs/kala_holo
plot_success dccs/indeep_apo_lig dccs/kala_apo_lig
plot_success dccs/indeep_holo_lig dccs/kala_holo_lig
plot_success dccs/indeep_apo_cavlig dccs/kala_apo_cavlig
plot_success dccs/indeep_holo_cavlig dccs/kala_holo_cavlig

eog /dev/stdin < =(montage dcc_plots/* -geometry 512x512\>+1+1 -)
