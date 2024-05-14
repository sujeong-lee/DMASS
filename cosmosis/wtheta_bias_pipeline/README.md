Cosmosis pipeline to constrain galaxy bias and redshift bias on DMASS wtheta 

to run this pipeline you first need to set up for cosmosis enviroment:
(see installation instructions here https://cosmosis.readthedocs.io/en/latest/intro/installation.html)

source cosmosis-configure


cosmosis params.ini

or 

mpirun -n <number of processes> cosmosis --mpi params.ini


cosmosis-postprocess chain_test_run.txt -o postprocess_output



