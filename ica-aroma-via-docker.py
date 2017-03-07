from __future__ import print_function
# Simple wrapper for the ICA-AROMA python scripts, to hand them absolute paths.
# Required to make the tool work in cbrain via the docker container

# function provided by Tristan A.A., ttaa9 on github

import os, sys
from subprocess import Popen

# Input arguments
args = sys.argv[1:]

# Modify the arguments for existent files and the output dir to be absolute paths
mod_args = [(os.path.abspath(f) if os.path.exists(f) else f) for f in args]
targ_ind = mod_args.index("-out") + 1
mod_args[targ_ind] = os.path.abspath(mod_args[targ_ind])

# Call the ICA-AROMA process
cmd = "python /ICA-AROMA/ICA_AROMA.py " + " ".join(mod_args)
print("Running: " + cmd + "\n")
process = Popen( cmd.split() ) 
sys.exit( process.wait() )


