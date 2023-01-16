########################################## General ##########################################
# Load Model
from gnn_models.mpnn_proposed import nmr_mpnn_PROPOSED
from gnn_models.mpnn_baseline import nmr_mpnn_BASELINE

import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import argparse
import ast
from model import training, inference
from dataset import GraphDataset

from dataset import GraphDataset
from dgllife.utils import RandomSplitter
from dgl.data.utils import split_dataset
from util import collate_reaction_graphs
from torch.utils.data import DataLoader
from model import training, inference
from helpers import *


def load_std_mean(target,graph_representation):
    """This functions returns the train_y_mean, train_y_std of the train dataset for either H or C"""
    #Load Train data (to get the train_y_mean and std)
    batch_size = 128
    data = GraphDataset(target, graph_representation)
    all_train_data_loader_C = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    train_y = np.hstack([inst[-2][inst[-1]] for inst in iter(all_train_data_loader_C.dataset)])
    train_y_mean, train_y_std = np.mean(train_y), np.std(train_y)
    # print(train_y_mean, train_y_std)
    return train_y_mean, train_y_std

def load_model(target, save_path):
    """ """
    data_sample = GraphDataset_sample(target, save_path)

    node_dim = data_sample.node_attr.shape[1]
    edge_dim = data_sample.edge_attr.shape[1]
    net = nmr_mpnn_PROPOSED(node_dim, edge_dim, readout_mode, node_embedding_dim, readout_n_hidden_dim).cuda()

    if target == "1H":
        model_path = "/home/jbr46/nmr_sgnn/model/1H_sparsified_proposed_proposed_set2set_1.pt"
        net.load_state_dict(torch.load(model_path))
    elif target == "13C":
        model_path = "/home/jbr46/nmr_sgnn/model/13C_sparsified_proposed_proposed_set2set_1.pt"
        net.load_state_dict(torch.load(model_path))
    return net

########################################## C related ##########################################

def save_as_npy_for_13C(mol):
    """ This functions executes all the code for creating the npy file for inference"""
    target = "13C" #"1H"

    atom_list = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
    charge_list = [1, 2, 3, -1, -2, -3, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]

    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    max_graph_distance = 20

    rdBase.DisableLog('rdApp.error') 
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    mol_dict = {'n_node': [],
                    'n_edge': [],
                    'node_attr': [],
                    'edge_attr': [],
                    'src': [],
                    'dst': [],
                    'shift': [],
                    'mask': [],
                    'smi': [],
                    # 'h_c_connectivity': [],
                    'h_x_connectivity': []}

    message_passing_mode = "proposed"
    readout_mode = "proposed_set2set"
    graph_representation = "sparsified"
    target = "13C"
    memo = ""
    fold_seed = 0
    data_split = [0.95, 0.05]
    batch_size = 128
    random_seed = 27407
    node_embedding_dim = 256 
    node_hidden_dim = 512 
    readout_n_hidden_dim = 512 



    Chem.rdmolops.SanitizeMol(mol)
    si = Chem.FindPotentialStereo(mol)
    for element in si:
        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
            mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
            mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
    assert '.' not in Chem.MolToSmiles(mol)

    # Write dummy shift (0.0) into the shift property and set mask for carbons True
    atom_selection_list = ["C"]#"H"
    for j, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() in atom_selection_list:
            atom.SetProp('shift', str(0.0))
            atom.SetBoolProp('mask', 1)
        else:
            atom.SetProp('shift', str(0.0))
            atom.SetBoolProp('mask', 0)

    mol = Chem.RemoveHs(mol)
    mol_dict = add_mol_sparsified_graph(mol_dict, mol)

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    if target == '13C': mol_dict['shift'] = np.hstack(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])

    #save the paramenters
    # for key in mol_dict.keys(): print(key, mol_dict[key].shape, mol_dict[key].dtype)
    folder = os.getcwd()
    save_path = os.path.join(folder, "sample_%s.npz"%(target))
    np.savez_compressed('./sample_%s.npz'%(target), data = [mol_dict])
    return save_path



def inference_C(target, net, save_path, train_y_mean_C, train_y_std_C):
    # Load data sample
    data_sample = GraphDataset_sample(target, save_path)
    test_sample_loader = DataLoader(dataset=data_sample, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    # inference
    test_y_pred_C, time_per_mol_C = inference(net, test_sample_loader, train_y_mean_C, train_y_std_C)
    return test_y_pred_C


########################################## H related ##########################################

def save_as_npy_for_1H(mol):

    ### Generate SDF File
    mol = Chem.AddHs(mol, addCoords=True)
    sdf_path = mol2SDF(mol)

    ### Generate connectivity list
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(sdf_path)
    # c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    x_h_connectivity_dict = get_x_h_connectivity(connectivity_list, atom_list) 


    target = "1H" #"13C" 

    atom_list = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
    charge_list = [1, 2, 3, -1, -2, -3, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]

    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    max_graph_distance = 20

    rdBase.DisableLog('rdApp.error') 
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

    mol_dict = {'n_node': [],
                    'n_edge': [],
                    'node_attr': [],
                    'edge_attr': [],
                    'src': [],
                    'dst': [],
                    'shift': [],
                    'mask': [],
                    'smi': [],
                    # 'h_c_connectivity': [],
                    'h_x_connectivity': []}

    target = "1H"
    message_passing_mode = "proposed"
    readout_mode = "proposed_set2set"
    graph_representation = "sparsified"
    memo = ""
    fold_seed = 0
    model_path = "/content/nmr_sgnn/model/1H_sparsified_proposed_proposed_set2set_1.pt"
    data_split = [0.95, 0.05]
    batch_size = 128
    random_seed = 27407

    node_embedding_dim = 256 
    node_hidden_dim = 512 
    readout_n_hidden_dim = 512 

    ### Check on Mol
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    si = Chem.FindPotentialStereo(mol)
    for element in si:
        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
            mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
            mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
    assert '.' not in Chem.MolToSmiles(mol)

    # Filter to get just the atoms that are connected to hydrogens
    H_ID_list = [k for k,v in x_h_connectivity_dict.items() if v !=[] ] 

    # Set shift = 0.0 for all C where H is connected
    for j, atom in enumerate(mol.GetAtoms()):
        if atom.GetIdx() in H_ID_list:
            atom.SetProp('shift', str(0.0))
            atom.SetBoolProp('mask', 1)
        else:
            atom.SetProp('shift', str(0.0))
            atom.SetBoolProp('mask', 0)

    # generate Mol_dict with all the information
    mol_dict = add_mol_sparsified_graph(mol_dict, mol)
    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    if target == '1H': mol_dict['shift'] = np.array(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])
    # mol_dict["h_c_connectivity"].append(c_h_connectivity_dict)
    mol_dict["h_x_connectivity"].append(x_h_connectivity_dict)


    #save the paramenters
    folder = os.getcwd()
    save_path = os.path.join(folder, "sample_%s.npz"%(target))
    np.savez_compressed('./sample_%s.npz'%(target), data = [mol_dict])
    return save_path, x_h_connectivity_dict



def inference_H(target, net, save_path, train_y_mean_H, train_y_std_H):
    # Load data sample
    data_sample = GraphDataset_sample(target, save_path)
    test_sample_loader = DataLoader(dataset=data_sample, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    # inference
    test_y_pred_H, time_per_mol_H = inference(net, test_sample_loader, train_y_mean_H, train_y_std_H)
    return test_y_pred_H

########################################## Reconstruct ##########################################

def create_shift_list(mol, x_h_connectivity_dict, test_y_pred_C, test_y_pred_H):
    """This function recreates the shift list based on the index labelling of the molecule (including H atoms)"""
    final_list = []
    c_count = 0 
    h_count = 0
    h_done = [] 
    for idx, atom in enumerate(mol.GetAtoms()):

        if atom.GetSymbol() == "C":
            final_list.append(test_y_pred_C[c_count])
            c_count += 1 

        elif atom.GetSymbol() == "H":
            # Check if the ID is in the connectivity list of the
            for k,v in x_h_connectivity_dict.items():
                if idx in v and idx not in h_done:
                    for i in v:           # iterate over the number of H and add them to list and also to done list
                        final_list.append(test_y_pred_H[h_count])
                        h_done.append(i)
                    h_count += 1 
        else:
            final_list.append(0)
    return final_list


def save_results(save_folder, ID, final_list):
    """ This function saves the final shift predictions into an output file with the lowest energy conformer"""

    name = str(ID) + ".sout"
    output_file = os.path.join(save_folder, name)
    with open(output_file, "w",  encoding='utf-8', errors='ignore') as output:
        for shift in final_list:
            output.write(str(shift))
            output.write('\n')
    return output


def main_execute(data_df, save_folder):
    """ Given a pandas dataframe with columns SMILES and sample-id it calculates 
    all the sdf files including the nmr shifts for every C and H atom"""
    ############## General ##############
    graph_representation = "sparsified"
    target = "13C" 
    train_y_mean_C, train_y_std_C = load_std_mean(target,graph_representation)
    target = "1H" 
    train_y_mean_H, train_y_std_H = load_std_mean(target,graph_representation)

    SMILES_list = data_df["SMILES"]
    sample_id = data_df["sample-id"]

    for smi, ID in tqdm(zip(SMILES_list[:5],sample_id[:5])):
        print(smi, ID)
        mol = MolFromSmiles(smi)
        ############## For 13C ##############
        target = "13C" 
        save_path_C = save_as_npy_for_13C(mol)
        net_C = load_model(target, save_path_C)
        test_y_pred_C = inference_C(net_C, save_path_C, train_y_mean_C, train_y_std_C)

        ############## For 1H ##############
        target = "1H" 
        save_path_H, x_h_connectivity_dict = save_as_npy_for_1H(mol)
        net_H = load_model(target, save_path_H)
        test_y_pred_H = inference_H(net_H, save_path_H, train_y_mean_H, train_y_std_H)

        ############## Reconstruction ##############

        mol = Chem.AddHs(mol)
        final_list = create_shift_list(mol, c_h_connectivity_dict, test_y_pred_C, test_y_pred_H)

        output_sdf = save_results_sdf_file(mol, save_folder, ID, final_list)
        print(output_sdf)