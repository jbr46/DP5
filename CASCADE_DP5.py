#!/usr/bin/env python

# -*- coding: utf-8 -*-
 
import os
import sys

sys.path.insert(0, '/Users/benji/CASCADE_package/CASCADE')

from CASCADE import prediction

def SetupCNMRPred(Isomers, settings):

    jobdir = os.getcwd()

    if not os.path.exists('nmr'):
        os.mkdir('nmr')
    os.chdir('nmr')

    if os.path.exists('CASCADE_inputs.csv'):
        os.remove('CASCADE_inputs.csv')
    with open('CASCADE_inputs.csv', 'w') as inputfile:
        inputfile.write('number,id\n')
        for num, iso in enumerate(Isomers):
            if os.path.exists(iso.BaseName + '.cout'):
                os.remove(iso.BaseName + '.cout')
            inputfile.write(str(num) + ',' + iso.BaseName + '\n')

    os.chdir(jobdir)

    return Isomers


def RunCNMRPred(Isomers, settings):

    print('\nRunning CASCADE NMR prediction locally...')
    
    save_folder = os.getcwd() + '/nmr'
    path_csv = save_folder + '/CASCADE_inputs.csv'
    prediction(save_folder, path_csv)

    return Isomers


def ReadCPred(Isomers, settings):

    jobdir = os.getcwd()
    os.chdir('nmr')

    if 'b' in settings.Workflow:

        for iso in Isomers:
            output_file = iso.BaseName + '.cout'
            with open(output_file, 'r') as cascadefile:
                for shift in cascadefile:
                    iso.PredShifts_CASCADE.append(float(shift.rstrip('\n')))

    else:

        for iso in Isomers:
            output_file = iso.BaseName + '.cout'
            with open(output_file, 'r') as cascadefile:
                for shift in cascadefile:
                    iso.PredShifts.append(float(shift.rstrip('\n')))


    os.chdir(jobdir)

    return Isomers
