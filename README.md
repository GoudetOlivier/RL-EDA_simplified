# Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning. Replication code and additional results.


To launch the algorithm enter the source_code repo.

The algorithm is in Python 3.11.5. 
All the library required to launch the multivariate RL EDAs are in the file requirement.txt.

Other libraries are required such as Nevergrad (see https://facebookresearch.github.io/nevergrad/) to run the competing algorithms.


An example of python command to run the (sigma,sigma)-RL-EDA version (reference version of the paper) with default hyperparameters for 10 QUBO instances  with n=128 and K=0 and 10 restarts for each instance (100 runs) on GPU device is :

python main_rl_eda.py QUBO 128 0 --verbose 


## Nevergrad competing algorithms

To run a nervergrad algorithm such as DiscreteDE the command line is :

python main_nevergrad.py DiscreteDE QUBO 128 0


## Other EDAs

To run the PBIL algorithm on the same instance the command line is

python main_baseline_edas.py PBIL QUBO 128 0

To run the MIMIC algorithm on the same instance the command line is

python main_baseline_edas.py MIMIC QUBO 128 0


## Nasbench dataset

Install first BBO-DOB with the following lines :

git clone https://github.com/e5120/BB-DOB
cd BB-DOB
pip install -r requirements.txt
pip install -e .

Download nasbench data here : https://github.com/google-research/nasbench
Install the library with the following lines :

git clone https://github.com/google-research/nasbench
cd nasbench
pip install -e .
