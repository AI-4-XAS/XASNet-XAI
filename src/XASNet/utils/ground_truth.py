import re 
from typing import Any, Dict, List, Optional
import os.path as osp
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .graph_preprocessing import GraphDataProducer
from scipy.signal import find_peaks
from .auc_roc import auc



class OrcaAnlyser():
    """
    Takes the orca's output files and process them to output
    the molecular orbitals and excitation states.    

    """
    def __init__(self,
                 path_orca_output: str,
                 path_orca_spectrum: str
                 ):
        """
        Args:
            path_orca_output: Path to orca output of tddft results. The output contains the 
                calculated excitation states and molecular orbitals.
            path_orca_spectrum: Path to the orca NEXAFS spectrum. The output file contains 
                excitation energies vs. oscillator strength.
        """
        assert osp.exists(path_orca_output), "orca output doesn't exist in the directory"
        assert osp.exists(path_orca_spectrum), "orca spectrum output doesn't exist in the directory"
        self.read_output = open(path_orca_output, "r").readlines()
        self.path_orca_spectrum = path_orca_spectrum

    def _group_by_match(self, read_output):
        buffer = []
        for line in read_output:
            if re.match('\w+ ATOM POPULATIONS', line):
                if buffer: yield buffer
                buffer = [line]
            elif re.match('\w+ REDUCED ORBITAL POPULATIONS', line):
                break
            else:
                buffer.append(line)
        yield buffer

    def mo_contributions(self):
        for spec_lines in \
            self._group_by_match(self.read_output):
            lines = spec_lines[3:]
        
        MO_dict = {}
        for i, line in enumerate(lines):
                match2 = re.match(r'\s*[0-9]\d*\s*[0-9]\d*\s*[0-9]\d*\s*[0-9]\d*\s*[0-9]\d*\s*[0-9]\d*', line)
                match3 = re.match(r'\s*[0-9]\d*\s\w\s*\d', line)
                if match2:
                    MOs = line.split()
                    atom_contib = {}
                elif match3:
                    i = line.split()
                    atom_contib[i[1] + ' ' + i[0]] = i[2:]
                    MO_dict[tuple(MOs)] = atom_contib
        df_MO_dict = {}
        for key, value in MO_dict.items():
            df_MO_dict[key] = pd.DataFrame(value, dtype=float)

        mo_df = pd.concat(df_MO_dict.values(), ignore_index=True).fillna(0)
        return mo_df
    
    def _highest_MO_contr(self):
        self.mo_df = self.mo_contributions()
        cols = self.mo_df.columns.tolist()
        highest_contb = self.mo_df[cols].to_numpy().argsort()[:, :-3-1:-1]
        return highest_contb
    
    def _excitation_finder(self, read_output):
        rocis_dict = {}
        for line in read_output:
            match1 = re.search('STATE\s*\d*[1-9]', line)
            match2 = re.search(r"^\s*[\d+]+[A-Za-z]\s*->", line)
            if match1:
                i = line.split()
                energy = i[0] + ' ' + i[1] + ' ' + i[-4] #"{:.2f}".format(float(i[-2].strip('eV')))  
                transitions_dict = {}
            elif match2:
                transitions_dict[line.split()[-3]] = [line.split()[0], line.split()[2]]
                if energy in rocis_dict.keys():
                    rocis_dict[energy].update(transitions_dict)
                else:
                    rocis_dict[energy] = transitions_dict
        return rocis_dict
    
    def _tddft_spectrum(self):
        tddft_spectrum = pd.read_csv(self.path_orca_spectrum, 
                                          delim_whitespace=True, 
                                          header=None)
        return tddft_spectrum
    
    def give_excitations(self) -> Dict[str, dict]:
        excitations = self._excitation_finder(self.read_output)
        df_excitations = pd.DataFrame.from_dict(excitations).fillna(0)
        all_excitations = {}
        for col in df_excitations.columns:
            trans_excitations = {}
            dict_excitations = dict(df_excitations.loc[df_excitations[col] != 0, col])
            for key, value in dict_excitations.items():
                trans_excitations[key] = [int(re.findall(r'\d+', value[i])[0]) for i in range(2)]
                all_excitations[col] = trans_excitations
        
        highest_contb = self._highest_MO_contr()

        for key, value in all_excitations.items():
            for k, v in value.items():
                value[k] = [tuple(self.mo_df.columns[highest_contb[v[0]]]),
                            tuple(self.mo_df.columns[highest_contb[v[1]]]),
                        tuple(self.mo_df.loc[v[1], self.mo_df.columns[highest_contb[v[1]]]])]
        
        df_spectrum = self._tddft_spectrum()
        dict_all_excitations = {}
        for key, row in zip(all_excitations.keys(), df_spectrum[1]):
            k = key.split(':')[-1]
            dict_all_excitations[f'energy = {k} / osc = {row}'] = all_excitations[key]

        return dict_all_excitations

    
class Contributions():
    """
    Class to get the contribution of donor 
    and acceptor atoms to a specific peak.

    """
    def __init__(self, 
                 excitations: Dict[str, dict],
                 cam_contr: pd.DataFrame,
                 peak: float,
                 atom_labels: List[str]):
        """
        Args:
            excitations (dict): dictionary of all excitations with mo contributions
            cam_contr (pd.DataFrame): data frame of cam contribution in all energies
        """
        self.excitations = excitations
        self.cam_contr = cam_contr
        self.peak = peak
        self.atom_labels = atom_labels

    def _peak_contributions(self):
        masks = (self.peak-1, self.peak+1)
        gt_contributions = {}
        for key, value in self.excitations.items():
            
            energy = float(key.split()[2])
            osc = float(key.split()[-1])
            
            if masks[0] < energy < masks[1]:
                probs = list(value.keys())
                acceptor = []
                donor = []
                acc_orbital = []
                for v in value.values():
                    donor.append(v[0][0])
                    acceptor.append(v[1])
                    acc_orbital.append(v[2])
                gt_contributions[(energy, osc)] = pd.DataFrame(
                    zip(donor, acceptor, probs, acc_orbital),
                    columns=['donor', 'acceptor', 'probabilities', 'acc_orbital']
                )
        cam_contributions = self.cam_contr.where((self.cam_contr.energies > masks[0]) & \
                                    (self.cam_contr.energies < masks[1])).dropna()

        return gt_contributions, cam_contributions
    
    def _refine_contributions(self, df_weights):
        cont_dic = {}
        for atom in self.atom_labels:
            if atom in df_weights.index:
                cont_dic[atom] =  df_weights.loc[atom][0]
            else:
                cont_dic[atom] =  0
        df_weights = pd.DataFrame.from_dict(
            {"atoms": list(cont_dic.keys()),
             "weights": list(cont_dic.values())}
             )
        
        df_weights["weights"] = MinMaxScaler().fit_transform(
            np.array(df_weights["weights"]).reshape(-1, 1)
            )

        return df_weights
    
    def don_acc_contrs(self):
        gt_contributions, _ = self._peak_contributions()
        # remove empty contributions 
        if gt_contributions:
            dfs_donor = []
            dfs_acc = []
            for key, value in gt_contributions.items():
                value['probabilities'] = pd.to_numeric(value['probabilities'])
                value = value.drop(value[value.probabilities < 0.3].index)
                osc_weight = key[1]
                donors = []
                accs = []
                probs = []
                acc_orbs = []
                for _, row in value.iterrows():
                    donors.append(row['donor'])
                    probs.append(row['probabilities'] * osc_weight)
                    weighted_orbitals = [orb * osc_weight * row['probabilities'] \
                                        for orb in row['acc_orbital']]
                    accs.append([*row['acceptor']])
                    acc_orbs.append([*weighted_orbitals])
                    
                dfs_donor.append(pd.DataFrame(
                    zip(donors, probs),
                    columns=['atoms', 'weights']
                ))
                dfs_acc.append(pd.DataFrame(
                    zip(chain(*accs), chain(*acc_orbs)), 
                    columns=['atoms', 'weights']
                )) 
    
            acceptor_contributions = pd.concat(dfs_acc, 
                                            axis=0).groupby('atoms').sum()
            donor_contributions = pd.concat(dfs_donor, 
                                            axis=0).groupby('atoms').sum()

            acceptor_contributions = self._refine_contributions(acceptor_contributions)
            donor_contributions = self._refine_contributions(donor_contributions)

            return acceptor_contributions, donor_contributions
        else:
            #return empty df if no contributions
            return pd.DataFrame(), pd.DataFrame()

        
    def cam_contrs(self):
        _, cam_contributions = self._peak_contributions()
        cols = cam_contributions.columns
        df_cam = cam_contributions[cols[2:]]
        df_ener_osc = cam_contributions[cols[:2]]
        atom_cam_weights = df_cam.multiply(df_ener_osc['osc'], 
                                                axis=0).sum().to_frame().reset_index()
        atom_cam_weights.columns = ['atoms', 'weights']
        atom_cam_weights['weights'] = MinMaxScaler().fit_transform(
            np.array(atom_cam_weights['weights']).reshape(-1, 1)
            )
        
        return atom_cam_weights
    

class GraphGroundTruth:
    """
    Base class to calculate the atomic contributions in each excitation state and 
    set the cam values of atoms for comparison. 
    """
    def __init__(self):
        self.excitations = None
        self.cam_data = None
        self.don_contr = None
        self.acc_contr = None
        self.cam_contr = None

    def set_excitation(self, path_orca_output,
                   path_orca_spectrum):
        orca_analyzer = OrcaAnlyser(path_orca_output,
                            path_orca_spectrum)
        self.excitations = orca_analyzer.give_excitations()

    def set_cam(self, x_pred, y_pred, 
                cam, atom_labels):
        self.cam_data = pd.DataFrame(np.c_[x_pred, y_pred, cam.T], 
            columns=['energies', 'osc', *atom_labels])

    def set_atomContributions(self, excitations, 
                              cam_data, peak, 
                              atom_labels):
        contributions = Contributions(
            excitations, 
            cam_data, 
            peak, 
            atom_labels
            )
        self.don_contr, self.acc_contr = contributions.don_acc_contrs()
        self.cam_contr = contributions.cam_contrs()


class GroundTruthBuilder:
    """
    The builder of atomic contributions in core and virtual orbitals.
    """
    def __init__(self, 
                 model: Any,
                 test_data: Any,
                 graph_index: int,
                 gnn_type: str,                  
                 path_to_orca_data: str):
        """
        Args:
            model: Trained GNN model.
            test_data: Uploaded test dataset of QM9_XAS dataset.
            graph_index: Index of the graph.
            gnn_type: Type of the GNN used for training.
            path_to_orca_data: The path which contains both orca output of tddft 
                calculations (.out) and the NEXAFS spectrum (.stk). 
        """
    
        self.graph = GraphDataProducer(model, gnn_type, 
                                 test_data, graph_index)
        
        self.atom_labels = self.graph.atom_labels()
        self.x_pred, self.y_pred = self.graph.predictions()

        self.graphGT = GraphGroundTruth()

        try:
            self.orca_output = osp.join(
                path_to_orca_data, 
                f"structure_{graph_index}/structure_{graph_index}.out"
                )
            
            self.orca_spectrum = osp.join(
                path_to_orca_data, 
                f"structure_{graph_index}/structure_{graph_index}.out.abs.stk"
                )
        except OSError as e:
            print(e)

    def _buildExcitations(self):
        self.graphGT.set_excitation(self.orca_output, 
                                           self.orca_spectrum)
    
    def _buildCamData(self):
        cam_data = self.graph.cam()
        self.graphGT.set_cam(self.x_pred, self.y_pred, 
                                    cam_data, self.atom_labels)
    
    def buildAtomContributions(self):
        atoms_contr_all_peaks = {}
        self._buildExcitations()
        self._buildCamData()
        peaks, _ = find_peaks(self.y_pred)
        all_peaks = self.x_pred[peaks]
        all_peaks = [peak for peak in all_peaks \
                     if peak > 274 and peak < 296]

        for peak in all_peaks:
            self.graphGT.set_atomContributions(
                self.graphGT.excitations,
                self.graphGT.cam_data,
                peak,
                self.atom_labels
            )
            if not self.graphGT.don_contr.empty:
                atoms_contr_all_peaks[peak] = {
                    "donor" : self.graphGT.don_contr,
                    "acceptor": self.graphGT.acc_contr,
                    "cam": self.graphGT.cam_contr
                }
            else:
                continue           
        return atoms_contr_all_peaks    

class GroundTruthGenerator():
    """
    Generator class to get ground truth data for single 
    and multiple graph data in QM9-XAS test dataset.
    """
    def __init__(self, 
                 model: Any, 
                 test_data: Any,
                 gnn_type: str, 
                 path_to_orca_data: str,
                 return_auc: bool = True
                 ):   
        """
        Args:
            model: Trained GNN model.
            gnn_type:  Type of the GNN used for training.
            test_data: Uploaded test dataset of QM9_XAS dataset.
            path_to_orca_data: The path which contains both orca output of tddft 
                calculations (.out) and the NEXAFS spectrum (.stk). 
            return_auc (bool, optional): If true, AUC scores is also calculated. Defaults to True.
        """
        self.model = model
        self.gnn_type = gnn_type
        self.test_data = test_data
        self.path_to_orca_data = path_to_orca_data
        self.return_auc = return_auc
        self.all_test_inds = [graph.idx for graph in test_data]

    def _calculate_auc(self, atoms_contr_all_peaks):
        auc_dict = {}
        for peak, contrs in atoms_contr_all_peaks.items():
            auc_dict[peak] = {
                    "auc_donor": auc(contrs["cam"]["weights"], 
                                     contrs["donor"]["weights"]),
                    "auc_acceptor": auc(contrs["cam"]["weights"], 
                                        contrs["acceptor"]["weights"])
            }
        return auc_dict

    def __len__(self):
        return len(self.all_test_inds)

    def __getitem__(self, idx):

        if isinstance(idx, int):
            GTBuilder = GroundTruthBuilder(
                self.model,
                self.gnn_type,
                self.test_data,
                self.all_test_inds[idx],
                self.path_to_orca_data
            )
            if self.return_auc:
                return self._calculate_auc(
                    GTBuilder.buildAtomContributions()
                )
            else:
                return GTBuilder.buildAtomContributions()

        elif isinstance(idx, slice):
            all_GTs = []
            for i in np.arange(idx.start, 
                               min(idx.stop, len(self)),
                               idx.step):
                try:
                    GTBuilder = GroundTruthBuilder(
                    self.model,
                    self.gnn_type,
                    self.test_data,
                    self.all_test_inds[i],
                    self.path_to_orca_data
                    )
                    if self.return_auc:
                        all_GTs.append(self._calculate_auc(
                        GTBuilder.buildAtomContributions()
                        ))
                    else:
                        all_GTs.append(
                            GTBuilder.buildAtomContributions()
                        )
                except Exception as e:
                    print(e)
                    print(f"error raised! check data from graph {self.all_test_inds[i]}")
            return all_GTs
