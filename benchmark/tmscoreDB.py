#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-02-17 08:40:18 (UTC+0100)

import gzip
import io
import numpy as np
from scipy.sparse import coo_matrix
import sys
import os
import subprocess
from multiprocessing import Pool


class TMscoreDB(object):
    def __init__(self, recfilename: str = None):
        """__init__.

        :param recfilename:
        :type recfilename: str
        """
        self.recfile = recfilename
        self.train_ind = dict()
        self.test_ind = dict()
        self.rows = []
        self.cols = []
        self.data = []
        if self.recfile is not None:
            self.build()

    def build(self):
        with io.TextIOWrapper(io.BufferedReader(gzip.open(self.recfile, 'r'))) as recfile:
            datapoint = []
            for line in recfile:
                splitout = line.split(':')
                if splitout[0] == 'train':
                    train = splitout[1].strip()
                    datapoint.append(train)
                if splitout[0] == 'test':
                    test = splitout[1].strip()
                    datapoint.append(test)
                if splitout[0] == 'tmscore':
                    tmscore = float(splitout[1].strip())
                    datapoint.append(tmscore)
                if splitout[0] == '\n':
                    self.add(datapoint)
                    datapoint = []
        self.format()

    def readfile(self, recfilename):
        inlist = []
        with open(recfilename, 'r') as recfile:
            traintest = {}
            for line in recfile:
                splitout = line.split(':')
                if splitout[0] == 'ref':
                    train = splitout[1].strip()
                    traintest['ref'] = train
                if splitout[0] == 'model':
                    test = splitout[1].strip()
                    traintest['model'] = test
                if splitout[0] == '\n':
                    inlist.append(traintest)
                    traintest = {}
        return inlist

    def add(self, datapoint: list):
        """add.

        :param datapoint:
        :type datapoint: list
        """
        train, test, tmscore = datapoint
        train = os.path.split(train)[-1]
        test = os.path.split(test)[-1]
        if len(self.train_ind) > 0:
            train_ind_max = max(self.train_ind.values())
        else:
            train_ind_max = -1
        if len(self.test_ind) > 0:
            test_ind_max = max(self.test_ind.values())
        else:
            test_ind_max = -1
        if train not in self.train_ind:
            self.train_ind[train] = train_ind_max + 1
        if test not in self.test_ind:
            self.test_ind[test] = test_ind_max + 1
        sys.stdout.write(f"Append data point: ({self.train_ind[train]}, {self.test_ind[test]})              \r")
        sys.stdout.flush()
        self.rows.append(self.train_ind[train])
        self.cols.append(self.test_ind[test])
        self.data.append(tmscore)
        print(f'{test} {train} {tmscore:.4f}')

    def compute(self, testtrain: tuple):
        """
        Compute TM-score for the given train and test pdb files
        """
        test = testtrain[0]
        train = testtrain[1]
        print(test, train)
        train_check = False
        test_check = False
        if train in self.train_ind:
            train_check = True
        if test in self.test_ind:
            test_check = True
        if not train_check or not test_check:
            process = subprocess.Popen('TMscore %s %s | grep "TM-score    =" | awk \'{print $3}\'' % (test, train), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = process.communicate()
            try:
                tmscore = float(stdout)
            except ValueError:
                tmscore = 0.
        return [train, test, tmscore]

    def format(self):
        n, p = max(self.rows) + 1, max(self.cols) + 1
        self.data = coo_matrix((self.data, (self.rows, self.cols)), shape=(n, p)).toarray()
        del self.rows, self.cols
        print(f'Shape of the data array: {self.data.shape}')

    def reformat_keys(self) -> None:
        """reformat_keys.

        :rtype: None

        Reformat the keys by removing the leading path
        HD-database/1/5/c/8/15c8-LH-P01837-P01869-H.pdb -> 15c8-LH-P01837-P01869-H.pdb
        """
        train_ind = dict()
        for train in self.train_ind:
            train_new = os.path.split(train)[-1]
            train_ind[train_new] = self.train_ind[train]
        test_ind = dict()
        for test in self.test_ind:
            test_new = os.path.split(test)[-1]
            test_ind[test_new] = self.test_ind[test]
        self.train_ind = train_ind
        self.test_ind = test_ind

    def filter(self, train_keys: list, test_keys: list) -> None:
        """filter.

        :param train_keys:
        :type train_keys: list
        :param test_keys:
        :type test_keys: list

        Filter the database with the given keys
        """
        train_ind = dict()
        test_ind = dict()
        train_keys = np.unique(train_keys)
        test_keys = np.unique(test_keys)
        data = np.zeros((len(train_keys), len(test_keys)))
        for i, train in enumerate(train_keys):
            train_ind[train] = i
            for j, test in enumerate(test_keys):
                test_ind[test] = j
                data[i, j] = self.get(train, test)
        self.train_ind = train_ind
        self.test_ind = test_ind
        self.data = data

    def get(self, train: str, test: str) -> float:
        """get.

        :param train:
        :type train: str
        :param test:
        :type test: str
        :rtype: float

        Get the given data
        """
        train = os.path.split(train)[-1]
        test = os.path.split(test)[-1]
        i = self.train_ind[train]
        j = self.test_ind[test]
        return self.data[i, j]

    def print(self, getmax: bool = False, print_keys: bool = False, max_axis=1):
        """print.

        :param getmax:

        Print all the data on stdout
        """
        def sort_keys(mydict):
            kv = mydict.items()
            kv = sorted(kv, key=lambda x: x[1])
            keys = np.asarray([e[0] for e in kv])
            return keys
        if not getmax:
            toprint = self.data.flatten()
        else:
            toprint = self.data.max(axis=max_axis).flatten()
        if print_keys:
            train_keys = sort_keys(self.train_ind)
            test_keys = sort_keys(self.test_ind)
            inds = self.data.argmax(axis=1)
            test_keys = test_keys[inds]
            toprint = np.vstack(((train_keys, test_keys, toprint.astype(np.str)))).T
        np.savetxt(sys.stdout, toprint, fmt='%s')


if __name__ == '__main__':
    import argparse
    import pickle
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-r', '--rec', type=str, default=None, help='Build the DB from a recfile containing the tmscore data')
    parser.add_argument('-o', '--out', default=None, help='Output database filename', type=str)
    parser.add_argument('-d', '--db', type=str, default=None, help='Load the given db pickle file')
    parser.add_argument('-p', '--print', action='store_true', help='Print all the data in the db')
    parser.add_argument('-m', '--max', action='store_true', help='Print the max over axis 1 of the data in the db')
    parser.add_argument('--max_axis', type=int, default=1, help='Axis to take the maximum value over')
    parser.add_argument('-g', '--get', nargs=2, type=str, help='Get the data for the given train, test couple')
    parser.add_argument('-f', '--filter', type=str, help='Filter the database with the given test file containing one couple "train_key test_key" per line')
    parser.add_argument('-k', '--keys', action='store_true', help='Print the keys along with the data')
    parser.add_argument('--format', action='store_true', help='Reformat the keys such as HD-database/1/5/c/8/15c8-LH-P01837-P01869-H.pdb -> 15c8-LH-P01837-P01869-H.pdb')
    parser.add_argument('--model', type=str, help='Compute the TM-score between model and ref (required)', default=None)
    parser.add_argument('--ref', type=str, help='Compute the TM-score between model and ref (required)', default=None)
    parser.add_argument('--file', type=str, help='Read test and train from a rec file with model and ref fields')
    parser.add_argument('--cpu', type=int, help='Run on the given number of cpus (default=1)', default=1)
    args = parser.parse_args()

    def test_out():
        if args.out is None:
            print("Please give on output filename with the --out option e.g.: '--out tmscore.pickle'")
            sys.exit(1)

    if args.db is not None:
        tmscoredb = pickle.load(open(args.db, 'rb'))
    else:
        tmscoredb = TMscoreDB()
    if args.model is not None and args.ref is not None:
        datapoint = tmscoredb.compute(args.model, args.ref)
        tmscoredb.add(datapoint)
    if args.file is not None:
        inlist = tmscoredb.readfile(args.file)
        testtrain_list = []
        for traintest in inlist:
            train = traintest['ref']
            test = traintest['model']
            testtrain_list.append((test, train))
        with Pool(args.cpu) as p:
            datapoints = p.map(tmscoredb.compute, testtrain_list)
        for datapoint in datapoints:
            tmscoredb.add(datapoint)
        tmscoredb.format()
    if args.rec is not None:
        test_out()
        tmscoredb = TMscoreDB(recfilename=args.rec)
    if args.db is not None:
        if args.format:
            test_out()
            tmscoredb.reformat_keys()
        if args.print:
            tmscoredb.print()
        if args.max:
            tmscoredb.print(getmax=True, print_keys=args.keys, max_axis=args.max_axis)
        if args.get is not None:
            data = tmscoredb.get(train=args.get[0], test=args.get[1])
            print(f'{data:.4f}')
        if args.filter is not None:
            test_out()
            traintest = np.genfromtxt(args.filter, dtype=str)
            train = traintest[:, 0]
            test = traintest[:, 1]
            tmscoredb.filter(train, test)
    if args.out is not None:
        pickle.dump(tmscoredb, open(args.out, 'wb'))
