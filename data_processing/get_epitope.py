#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-23 11:08:27 (UTC+0200)

import os
import pymol
import pymol.cmd as cmd

dbfiles = ['HD-database_AB.txt', 'HD-database_BA.txt']
for dbfilename in dbfiles:
    with open(dbfilename, 'r') as dbfile:
        LPset = {'L', 'P'}
        LPdict = {}
        for line in dbfile:
            line = line.strip()
            print(line)
            LP = line.split(" ")[0]
            pdbfilename = line.split(" ")[1]
            LPdict[LP] = pdbfilename
            pdbcode = pdbfilename[:4]
            path = 'HD-database/%s/%s/%s/%s'%(pdbcode[0], pdbcode[1], pdbcode[2], pdbcode[3])
            cmd.load(path+"/"+pdbfilename, LP)
            cmd.remove('hydrogens')
            LPset -= {LP}
            if len(LPset) == 0:
                cmd.remove('L and not (byres P around 6.)')
                outfilename = path + '/' + os.path.splitext(LPdict['L'])[0] + '-short.pdb'
                cmd.save(outfilename, 'L')
                LPdict = {}
                LPset = {'L', 'P'}
                cmd.delete('all')
