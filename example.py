from ase.build import bulk
from ase.visualize import view
from ase.io import read, write
import numpy as np
import os
import fileinput
from shutil import rmtree
from shutil import copyfile
from acease.ace_calculator import ACEpotentials, initialize_julia


model_path = "path/to/model.json"
python_path = "path/to/python"

initialize_julia(python_path) # pass your python path
calculator = ACEpotentials(model_path)

# create Simple bulk crystal
# lattice constant used before: ao=3.6358
# 3.55 to 3.65
steps_num = 46
step = 0.01
#lat_consts = []
start_val = 3.45
path = os.getcwd()
cur_Etot = ''
output_file = 'lattice_const.out'
bulk_cu = bulk('Cu', 'fcc', a=start_val, cubic=False)

with open(output_file, 'w') as f:
    print('') # empty file

for i in range(steps_num):
	cur_val = start_val + i*step
	cur_val = round(cur_val,3)
	bulk_cu = bulk('Cu', 'fcc', a=cur_val, cubic=False)

	# calc energy
	bulk_cu.calc = calculator
	cur_Etot = bulk_cu.get_potential_energy()

	#read aims.out and take total energy from it
	# os.chdir('..')
	with open(output_file, 'a') as f:
		f.write(str(cur_val) + '   ' + str(cur_val**3) + '   ' + str(cur_Etot) + '\n')