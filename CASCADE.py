import pandas as pd
import numpy as np
from rdkit import Chem
from nfp.preprocessing import MolAPreprocessor, GraphSequence

import keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.layers import (Input, Embedding, Dense, BatchNormalization,
                                 Concatenate, Multiply, Add)

from keras.models import Model, load_model

from nfp.layers.layers import (MessageLayer, Squeeze, EdgeNetwork,
                               ReduceBondToPro, ReduceBondToAtom, GatherAtomToBond, ReduceAtomToPro)
from nfp.layers.wrappers import GRUStep
from nfp.models.models import GraphModel
from cascade.apply import predict_NMR_C,predict_NMR_H

import os

def prediction(save_folder, path_csv):

    # modelpath_C = os.path.join('cascade', 'trained_model', 'best_model.hdf5')
    # modelpath_H = os.path.join('cascade', 'trained_model', 'best_model_H_DFTNN.hdf5')
    modelpath_C = '/users/benji/dp5/cascade/trained_model/best_model.hdf5'
    modelpath_H = '/users/benji/dp5/cascade/trained_model/best_model_H_DFTNN.hdf5'

    batch_size = 32
    atom_means = pd.Series(np.array([0,0,97.74193,0,0,0,0,0,0,0]).astype(np.float64), name='shift')
    NMR_model_C = load_model(modelpath_C, custom_objects={'GraphModel': GraphModel,
                                                'ReduceAtomToPro': ReduceAtomToPro,
                                                'Squeeze': Squeeze,
                                                'GatherAtomToBond': GatherAtomToBond,
                                                'ReduceBondToAtom': ReduceBondToAtom})
    NMR_model_H = load_model(modelpath_H, custom_objects={'GraphModel': GraphModel,
                                                'ReduceAtomToPro': ReduceAtomToPro,
                                                'Squeeze': Squeeze,
                                                'GatherAtomToBond': GatherAtomToBond,
                                                'ReduceBondToAtom': ReduceBondToAtom})
    # NMR_model_C.summary()
    # NMR_model_H.summary()

    #Loading DATA/
    data = pd.read_csv(path_csv)
    data.columns

    # C predicting NMR
    for i, ID in enumerate(data.id):
        pred_data_C = pd.DataFrame()
        mols, weightedPrediction, spreadShift = predict_NMR_C(ID, NMR_model_C)
        weightedPrediction['ID'] = ID
        pred_data_C = pd.concat([pred_data_C,weightedPrediction])

    # H predicitions
        pred_data_H = pd.DataFrame()
        try:
            mols, weightedPrediction, spreadShift = predict_NMR_H(ID, NMR_model_H)
            weightedPrediction['ID'] = ID
            pred_data_H = pd.concat([pred_data_H, weightedPrediction])
        except:
            pass

        predictions = pd.concat([pred_data_C, pred_data_H])

        # TODO:
        # Need to save the results of the NMR prediction into output files, ordered by the atom index. 
        # Need to fill in zeros for atoms other than C and H.
        name = str(ID) + ".cout"
        output_file = os.path.join(save_folder, name)
        with open(output_file, "w",  encoding='utf-8', errors='ignore') as output:
            idx = 1
            for label, shift in zip(predictions.atom_index, predictions.Shift):
                while idx < label:
                    output.write(str(0))
                    output.write('\n')
                    idx += 1
                output.write(str(shift))
                output.write('\n')
                idx += 1

            




def save_results(save_folder, ID, final_list):
    """ This function saves the final shift predictions into an output file with the lowest energy conformer"""

    name = str(ID) + ".cout"
    output_file = os.path.join(save_folder, name)
    with open(output_file, "w",  encoding='utf-8', errors='ignore') as output:
        for shift in final_list:
            output.write(str(shift))
            output.write('\n')
    return output


def create_shift_list(ID, C_shifts, H_shifts):
    """This function recreates the shift list based on the index labelling of the molecule (including H atoms)"""

    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(ID + '.sdf')
    # c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    x_h_connectivity_dict = get_x_h_connectivity(connectivity_list, atom_list)

    final_list = []
    c_count = 0 
    h_count = 0
    h_done = [] 
    for idx, atom in enumerate(mol.GetAtoms()):

        if atom.GetSymbol() == "C":
            final_list.append(C_shifts[c_count])
            c_count += 1 

        elif atom.GetSymbol() == "H":
            final_list.append(H_shifts[h_count])
            h_count += 1
            # Check if the ID is in the connectivity list of the
            # for k,v in x_h_connectivity_dict.items():
            #     if idx in v and idx not in h_done:
            #         for i in v:           # iterate over the number of H and add them to list and also to done list
            #             final_list.append(H_shifts[h_count])
            #             h_done.append(i)
            #         h_count += 1
        else:
            final_list.append(0)
    return final_list

def get_molecule_data(compound_path):
    """ This function returns the list of atoms of the molecules 
    and a list of strings with each line of the compound document as one string 
    because reading the SDF files with pandas causes an error"""
    
    ################ atom_list ################
    index_list_C =[]
    index_list_H =[]
    docline_list = []
    start = False
    name = compound_path.split("/")[-1].split(".")[0]

    # save each string line of the document in a list
    with open(compound_path) as f:
        for i in f:
            docline_list.append(i)

    # get index location of C and H and atom list
    atom_list = []
    stop_list = ["1","0","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
    for i in docline_list:

        if start:
            if "C" in i:
                index_list_C.append(counter) 
            if "H" in i:
                index_list_H.append(counter)
            if (i.split()[0] in stop_list):   # to find the end of the atom defining list
                break
            atom = i.split()[3]
            if atom != "0":
                atom_list.append(atom)
                counter += 1

        if "V2000" in i:   # select when to start looking
            start = True
            counter = 0    
            
    ################ connectivity_list ################
    # # To get the reference right add one empty string to the list -> need to check that there is something wrong
    # atom_list_ = [" "] + atom_list
    atom_list_ = atom_list
    start_line = len(atom_list_)+4
    end_line = len(atom_list_)+4 + len(atom_list_)
    connectivity_list = []

    for idx, i in enumerate(docline_list):
        if idx >= start_line and "M  END" not in i:
            add_list = i.split()
            ### if there are more than 100 connection the first and second columns are connected
            ### therefore I need to manually split them
            if len(add_list) ==3:
                part_1 = str(int(i[:3]))
                part_2 = str(int(i[3:6]))
                part_3 = str(int(i[6:9]))
                part_4 = str(int(i[9:12]))
                add_list =[part_1,part_2,part_3,part_4]

            ### For some reason sometimes it is too long and sometimes it is too short
            if add_list[0]=="M" or add_list[0]==">" or add_list[0]=="$$$$":
                pass
            else:
                connectivity_list.append(add_list)
        if  "M  END" in i:
            break  
    ################ mol ################
    # save each string line of the document in a list
    with open(compound_path[:-4]+".mol", "w",  encoding='utf-8', errors='ignore') as output:  # Path to broken file
        with open(compound_path) as f:
            for element in f:
                if "END" in element:
                    output.write(element)
                    break
                else:
                    output.write(element)

    mol = Chem.MolFromMolFile(compound_path[:-4]+".mol")
    return atom_list, connectivity_list, docline_list, name, mol


def get_x_h_connectivity(connectivity_list, atom_list):
    """ This function checks the connectifity list and creates a dictionary with all the 
    X atoms that are connected to hydrogens with their labelled numbers"""
    x_h_connectivity_dict = {}
    # print(connectivity_list)
    for i in connectivity_list:
        selected_atom_nr = int(i[0])-1
        selected_connection_nr = int(i[1])-1
        atom = atom_list[selected_atom_nr]
        connection = atom_list[selected_connection_nr]
        num_connection = atom_list[int(i[2])]
        # print(atom, selected_atom_nr, connection, selected_connection_nr)
        # check atom X-H bonds and add them to dictionary
        if (atom =="O" or atom =="S" or atom =="N" or atom =="P" or atom == "C") and connection == "H":
            found_H_nr = [selected_connection_nr]
            found_X_nr = selected_atom_nr
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(x_h_connectivity_dict[found_X_nr]) == list
                x_h_connectivity_dict[found_X_nr]+=found_H_nr
            except:
                x_h_connectivity_dict[found_X_nr]=found_H_nr
        # check atom X-H bonds and add them to dictionary
        if atom =="H" and (connection =="O" or connection =="S" or connection =="N" or connection =="P" or connection =="C"):
            found_X_nr = selected_connection_nr
            found_H_nr = [selected_atom_nr]
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(x_h_connectivity_dict[found_X_nr]) == list
                x_h_connectivity_dict[found_X_nr]+=found_H_nr
            except:
                x_h_connectivity_dict[found_X_nr]=found_H_nr
       
    return x_h_connectivity_dict