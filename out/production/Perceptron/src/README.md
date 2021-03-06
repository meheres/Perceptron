# Perceptron
Honors Advanced Topics in Computer Science: Neural Networks, Dr. Eric R. Nelson

A Perceptron which is able to solve the AND, OR, & XOR logic problems. Additionally, it is capable of handling
 Bitmap files and training to match them. The program trains using back propagation for any number of hidden layers.

## Running Trainer.main()
The Trainer class's main method must be run with four arguments in the ```String[] args```. Currently, the arguments are passed in
 done through the command line, as the following example indicates:
 
 ```java Trainer.java inputsFile trialCasesFile truthsFile outputsFile```
 
 The ```inputsFile``` specifies the basic information for the Perceptron and Trainer classes, as detailed below.
 The ```trialCasesFile``` specifies the inputs for each trial during training. The ```truthsFile``` specifies the expected
  outputs for each trial during training.
 The ```outputsFile``` is where the Perceptron's final result will be printed once training is complete.
 
 ## Inputs File Format
- Each line should contain a series of decimal numbers, separated by a single whitespace.

- The first line of the file should be the number of input nodes.

- The second line of the file should be the number of hidden layer nodes, followed by the number of nodes in each hidden layer.

- The third line of the file should be the number of output nodes.

- The fourth line of the file should be the number of trial cases.

- The fifth line of the file should contain a double value of the starting value for lambda.

- The sixth line of the file should contain a double value for the maximum value of lambda.

- The seventh line of the file should contain a double value of the constant with which lambda will be multiplied when the previous
 error is less than the current error.
 
- The eighth line of the file should contain a double value of the constant with which lambda will be multiplied when the
 previous error is greater than the current error.

- The ninth line of the file should contain an int value of the minimum error to be reached before the Perceptron terminates its
 training.

- The tenth line of the file should contain a double value of the maximum number of iterations before the training terminates
 (times out). 
 
- The eleventh line of the file should contain a double value of the lower bound for the random weights.

- The twelfth and final line of the file should contain a double value of the upper bound for the random weights.
 
## Trial Cases File Format
- The number of lines in this file must be exactly equal to the number of trials.

- Each line represents an individual trial case, with each element in the string representing an input activation. Each activation
 separated by a whitespace. For example, if there are
 three inputs ```input1```, ```input2```, and ```input3```, and only one training case, the file will be a single line with the
  three inputs separated by one whitespace. The number of double elements in each line must be exactly equal to the number of
   input nodes.

## Truths File Format
- The Truths File is formatted in the same manner as the Trial Cases File. The number of lines must be exactly equal to the
 number of trials.
 
 - Each line represents the expected output for the corresponding trial in the Trial Cases File, and each element in each line
  must be separated by a whitespace. The number of elements in each line must be exactly the same as the number of output nodes.


## Outputs File Format
- The Outputs File can be any file without valuable information. It will have the Perceptron's final result printed to it.

## Notes
The code is not written for large-scale distribution, so all files must be correctly formatted as specified
 -- otherwise, there may be runtime errors.  