from debyecalculator import DebyeCalculator
from ase.cluster import Icosahedron
import numpy as np
import os
import time
import torch
path='home-local/ratel-ra/GPU_tests'
os.makedirs(path,exist_ok=True)

def writexyz(path,atoms):
        """atoms ase Atoms object"""
        outputfile=path+'/structure.xyz'
        #write(strufile_dir+f'/{filename}.xyz',atoms)
        element_array=atoms.get_chemical_symbols()
        # extract composition in dict form
        composition={}
        for element in element_array:
            if element in composition:
                composition[element]+=1
            else:
                composition[element]=1
        
        coord=atoms.get_positions()
        natoms=len(element_array)  
        line2write='%d \n'%natoms
        line2write+='%s\n'%str(composition)
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(outputfile,'w') as file:
            file.write(line2write)

def build_icosahedron_compute_iq(path,nbshells):
    # create structure
    ico = Icosahedron('Au',nbshells,latticeconstant=4.08)
    nbatoms=len(ico)
    writexyz(path,ico)
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    calc=DebyeCalculator(qmin=0.001,qmax=20,qstep=0.001,qdamp=0.014,device=device)
    start_time = time.time()  # Record start time
    try:
        q,i=calc.iq(structure_source=path+'/structure.xyz')
        end_time = time.time()  # Record end time
        elapsed_time=end_time-start_time
    except:
        elapsed_time=-1
    os.remove(path+'/structure.xyz')
    return device, nbatoms, elapsed_time

path='/GPU_tests'
os.makedirs(path,exist_ok=True)
sizes = np.arange(10, 500, 10)
line2write = "{:<6},{:<10},{:<15},{:<10}\n".format("Size", "Atoms", "Elapsed Time", "Device")  # Header
for size in sizes:
    device, nb_atoms, elapsed_time = build_icosahedron_compute_iq(path, size)
    line2write += "{:<6},{:<10},{:<15.6f},{:<10}\n".format(size, nb_atoms, elapsed_time, device)
with open(path+'/ComputationTimes.csv','w') as file:
    file.write(line2write)