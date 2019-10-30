# Perceptron
Honors Advanced Topics in Computer Science: Neural Networks, Dr. Eric R. Nelson

Currently a 2-2-1 perceptron which is able to solve the AND, OR, & XOR logic problems.

## Input File Format
The inputs file MUST be formatted correctly for the Simple Network to work.

- Each line should contain a series of decimal numbers, separated by a single whitespace.

- The first line of the file should be the number of input nodes.

- The next line of the file should be the nodes of the input activation layer. In example, if there
  are two input activations with values [input1, input2], the first lines should look like the
  following:  
  ```input1 input2```  
- The next line of the file should be each of the weights for the program. Weights that do not
  exist in the design document but do in the 2D array of activations should also be instantiated as
  "0.0". The line should be each weight in each connectivity layer, written in their order. For
  example, if there are two connectivity layers with values [a0, a1, a2] and [b0, b1], the line
  should look like the following.  
  ```a0 a1 a2 b0 b1```  
- The next line of the file should be the expected outputs for the function. In example, if
  there is supposed to be one expected output of [output1], the line would simply be one number, as
  follows:  
  ```output1```  
- The next line of the file should contain a double value of the starting value for lambda. Currently, the lambda value will be the
 final lambda value because lambda is not set to be adaptive, but will eventually be set to E/10.

- The next line of the file should contain a double value of the minimum error to be reached before the Perceptron terminates its
 training.

- The next line of the file should contain a double value of the maximum number of iterations before the training terminates
 (times out). 
 
- The next line of the file should contain a double value of the upper bound for the random weights.

- The final line of the file should contain a double value of the lower bound for the random weights.
 
## Notes
In the main method, it is crucial to correctly define the inputs, outputs, and hidden layers in
accordance with the inputs file.  
Additionally, it is important to ensure that the full path of the filename is specified in the main
method.