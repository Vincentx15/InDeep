#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-27 16:19:50 (UTC+0200)

import glob
import os

with open("../data/HD-database_short.txt", 'r') as f:
    for line in f:
        line = line.strip()
        PL = line.split(" ")[0]
        if PL == 'L':
            pdbcode = line.split(" ")[1][:4]
            pdbflag = line.split(" ")[1][4]
            ligandname = line.split(" ")[1][:4]+line.split(" ")[1][5:-4]
            tosearch = 'HD-database/'+pdbcode[0]+'/'+pdbcode[1]+'/'+pdbcode[2]+'/'+pdbcode[3]+'/'+ligandname+ '_*.pdb'
            shorts = glob.glob(tosearch)
            for short in shorts:
                short = os.path.split(short)[-1]
                short = short[:4]+pdbflag+short[4:]
                print(PL+" "+short)
        else:
            print(line)
