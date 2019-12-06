#!/usr/bin/env python3

import subprocess

# `cd' into the Perceptron Directory
subprocess.run(["cd","~/IdeaProjects/Perceptron/src"])

# compile the DibDump java file
subprocess.run(["javac","DibDump.java"])

# then, run the Dibdump java file to set inputs and outputs
subprocess.run(["java","DibDump","smallA.bmp","bmpTrialCases.txt"])
subprocess.run(["java","DibDump","smallA.bmp","bmpTruths.txt"])

# compile the trainer, then run it with the correct arguments
# this mandates that the `inputsFile.txt' is formatted correctly
subprocess.run(["javac","Perceptron.java"])
subprocess.run(["javac","Trainer.java"])

# run the trainer
subprocess.run(["java","Trainer","bmpInputsFile.txt","bmpTrialCases.txt","bmpTruths.txt","bmpFinalOuts.txt"])

#open the final bitmap and the original for comparison purposes
subprocess.run(["open","smallA.bmp"])
subprocess.run(["open","out.bmp"])
