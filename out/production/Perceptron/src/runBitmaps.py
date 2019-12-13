#!/usr/bin/env python3

import subprocess

# filename
word = "mediumA.bmp"

# compile the DibDump java file
subprocess.run(["javac","DibDump.java"])

# then, run the Dibdump java file to set inputs and outputs
subprocess.run(["java","DibDump",word,"bmp/bmpTrialCases.txt"])
subprocess.run(["java","DibDump",word,"bmp/bmpTruths.txt"])

# compile the trainer, then run it with the correct arguments
# this mandates that the `inputsFile.txt' is formatted correctly
subprocess.run(["javac","Perceptron.java"])
subprocess.run(["javac","Trainer.java"])

# run the trainer
subprocess.run(["java","Trainer","bmp/bmpInputsFile.txt","bmp/bmpTrialCases.txt","bmp/bmpTruths.txt","bmp/bmpFinalOuts.txt"])

# recompile the new bitmap
subprocess.run(["java","DibDump",word,"tempFile.txt","bmp/bmpFinalOuts.txt"])

#open the final bitmap and the original for comparison purposes
subprocess.run(["open",word])
subprocess.run(["open","out.bmp"])
