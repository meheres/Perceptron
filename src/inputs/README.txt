The inputs file MUST be formatted correctly for the Simple Network to work.
The lines should be structured as follows:

Line 1: The number of input activations.

Line 2: The number of hidden layers, followed by the number of activations in each hidden layer.
        For example: 2 hidden layers with 4 activations in the first layer and 20 in the second will be written as:
            2 4 20

Line 3: The number of output activations.

Line 4: The number of test cases (n) being used to train the network.

Lines (4+n): An individual test case, structured as the inputs followed by the expected outputs.
             For example: 2 test cases with inputs {0,0} and {1,1} with expected outputs {1} and {0}, respectively.
                Line 5: 0 0 1
                Line 6: 1 1 0

------------
In the main method, it is crucial to correctly define the inputs, outputs, and hidden layers in
accordance with the inputs file.
Additionally, it is important to ensure that the full path of the filename is specified in the main
method.