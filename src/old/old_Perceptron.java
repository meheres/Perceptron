package old;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

/**
 * @author Mihir Sharma
 * @version 2.0
 * Version 1 created on Friday, 9.6.19.
 * File created on Thursday, 9.12.19.
 * Last edited on Friday, 9.20.19.
 * <p>
 * The SimpleNet class is a Java implementation of a  2-2-1 perceptron. The network attempts to solve a basic logic problem,
 * either AND, OR, or XOR, using a two-connectivity network with three layers of activation. The neural network will take in two
 * boolean inputs, represented by 1 (true) and 0 (false). It will output the sum of the dot products of the connectivity layers
 * and the activation layers. When constructing the Simple Network, there should be an input file that provides the inputs,
 * weights, and expected outputs.
 * <p>
 * Currently, the SimpleNet is configured with weights that are accurate for the AND boolean formula.
 * <p>
 * Functions in this class include the following:
 * - readInput, which reads the input from the constructor's filename then populates the inputs, weights, and expected outputs.
 * - runNetwork, which uses the inputs and weights to run the Simple Network for an AND logic table.
 * - main, which instantiates the Simple Network and runs the other two functions. The main also prints the output and its
 * expected comparison.
 */
public class old_Perceptron
{

   int numTrials;                     // The number of trials to be run.
   int minimumError;                  // The minimum error for training.
   int inputNodes;                    // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;            // The number of nodes in each hidden activation layer.
   int outputNodes;                   // The number of nodes in the output activation layer.
   String inputFileName;              // The name of the input file. Should be written as a full path.

   static final int NUM_COLUMNS = 2;  // One for the input layer, one for the output layer.

   BufferedReader bufferedReader;     // bufferedReader is for the input file. It allows for quick line-by-line reading of the file.
   StringTokenizer stringTokenizer;   // stringTokenizer reads each line provided by the BufferedReader, allowing for
                                      // token-by-token reading.

   int numberActivationLayers;        // The number of connectivity layers will be one less than the number of activation layers.
                                      // The number of activation layers will be 2 more than the number of hidden layers, one for
                                      // input activations and one for output activations.

   double[][] inputs;                 // An Array that holds the values for the input activations. Read in first line of input
                                      // file.

   double[][] expectedOutputs;        // An Array that holds the values for the expected outputs, for comparison with the actual
                                      // outputs.

   double[][] activationLayers;       // A 2D Array that represents the different activation layers. First index will be the
                                      // number of activation layers, and the second index will specify which node from the
                                      // activation layer to use.

   double[][][] connectivityLayers;   // A 3D Array that represents the connectivity layers.


   /**
    * SimpleNet is the constructor for the neural network. It creates the activation layers and connectivity layers for the
    * network. The constructor also determines the most number of nodes by iterating over all of the activation layers.
    *
    * @param inputNodes       The inputNodes parameter is an integer which represents the number of nodes in the input activation layer.
    * @param hiddenLayerNodes The hiddenLayerNodes parameter is an array of integers, where each integer in the array represents
    *                         the number of nodes in each hidden layer. The length of the array should be two less than the
    *                         total number of layers.
    * @param outputNodes      The outputNodes parameter is an integer which represents the number of nodes in the output activation
    *                         layer.
    * @param inputFileName    The inputFileName parameter is a String containing the name of the file with weights, inputs, and
    *                         expected output.
    */
   old_Perceptron(int numTrials, int inputNodes, int[] hiddenLayerNodes, int outputNodes, String inputFileName)
   {
      this.numTrials = numTrials;

      try                                                                      // Check for an error when trying to open the file.
      {
         bufferedReader = new BufferedReader(new FileReader(inputFileName));
      }
      catch (IOException e)
      {
         throw new RuntimeException("File not found.");
      }

      this.inputNodes = inputNodes;                                            // Update the Class-level variables for the number of
      this.hiddenLayerNodes = hiddenLayerNodes;                                // inputs, outputs, hidden layers, and total number
      this.outputNodes = outputNodes;                                          // of activation layers.


      this.numberActivationLayers = NUM_COLUMNS + hiddenLayerNodes.length;     // Add 2 to the number of hidden layers for the total
      this.inputFileName = inputFileName;                                      // number of layers (1 input + n hidden + 1 output).
      this.inputs = new double[numTrials][inputNodes];
      this.expectedOutputs = new double[numTrials][outputNodes];

      int maxNumberNodes = inputNodes;
      for (int hiddenLayerNode : hiddenLayerNodes)                             // Determine the most number of nodes in any layer
      {
         maxNumberNodes = Math.max(maxNumberNodes, hiddenLayerNode);
      }

      maxNumberNodes = Math.max(maxNumberNodes, outputNodes);                  // Update the max number of nodes in the class
                                                                               // variables.

      activationLayers = new double[maxNumberNodes][numberActivationLayers];   // The first index of the activationLayers
      connectivityLayers = new double[numberActivationLayers - 1]              // represents the number of nodes in the layer
                                                                               // (limited by the maximum number of
            [numberActivationLayers - 1]                                       // nodes in the neural network, while the
            [numberActivationLayers - 1];                                      // second index is the number of activation layers.
   }


   /**
    * Method readInput uses the input file to populate three class level objects: the array of inputs, the 3D array of weights,
    * and the expected output value.
    */
   void readInput()
   {
      // Ensure that the StringTokenizer doesn't crash the program upon creation
      try
      {
         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " not accepted, terminating.");
      }

      // Iterate over the inputs in the first line then populate the input activation array
      for (int j = 0; j < numTrials; j++)
      {
         for (int i = 0; i < inputNodes; i++)
         {
            this.inputs[j][i] = Double.parseDouble(stringTokenizer.nextToken());
         }
      }

      try
      {
         // Advance the Buffered Reader by one line to begin reading the weights
         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " incorrectly shows weight, terminating.");
      }

      // Iterate over the 3D array and populate it with the weights as written in the weights file
      for (int i = 0; i < connectivityLayers.length; i++)
      {
         for (int j = 0; j < connectivityLayers[i].length; j++)
         {
            for (int k = 0; k < connectivityLayers[i][j].length; k++)
            {
               this.connectivityLayers[i][j][k] = Double.parseDouble(stringTokenizer.nextToken());
            }
         }
      }

      try
      {
         // Advance the Buffered Reader by one line to begin reading the expected outputs.
         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " incorrectly shows expected output, terminating.");
      }

      // Add each expected output to the array of expected outputs.
      for (int j = 0; j < numTrials; j++)
      {
         for (int i = 0; i < outputNodes; i++)
         {
            this.expectedOutputs[j][i] = Double.parseDouble(stringTokenizer.nextToken());
         }
      }
   }


   /**
    * The main method is the manner in which the Simple Network is run. Currently, it creates a 2-2-1 network and then runs the
    * readInput and runNetwork methods on that network. Finally, the network outputs its result for the test value. Currently,
    * it is limited by the input file -- in the event that the input file is formatted incorrectly, the program will shut down.
    *
    * @param args The arguments for the main method.
    */
   public static void main(String[] args)
   {
      int numberInputActivations = 2;                     // Provide input activations, output activations, file path, number of
      int numberOutputActivations = 1;                    // hidden layers, and the number of activations in each hidden layer.

      int numberTrials = 4;

      String inputFileFullPath = "/Users/mihir/IdeaProjects/Neural Networks/Java " +
            "XOR Implementation/src/inputsFile.txt";      // This must be the full file path for the input file.
      int numberHiddenLayers = 1;
      int numberHiddenActivationsLayer1 = 2;

      int[] inputarr = new int[numberHiddenLayers];       // Populates input array.
      inputarr[0] = numberHiddenActivationsLayer1;

      old_Perceptron testNetwork = new old_Perceptron(numberTrials, numberInputActivations, inputarr, numberOutputActivations,
            inputFileFullPath);

      testNetwork.readInput();
      testNetwork.runNetwork(1);
      testNetwork.trainNetwork();

      System.out.println(testNetwork.findTotalError(1)); //DEBUG


      for (int i = 0; i < numberOutputActivations; i++)   // Print all of the results, which will be found in the output activation
                                                          // layer (last column of each row).
      {
         System.out.println("Result: " + testNetwork.activationLayers[i][testNetwork.activationLayers[0].length - 1]);
      }
   }


   /**
    * Method to train the network
    */
   void trainNetwork()
   {
      double avgError = 10.0;
      while (avgError > minimumError)
      {
         double errorBefore = findTotalError(1);
         steepestDescent(expectedOutputs[1]);

      }

   }


   /**
    * Method runNetwork runs the Simple Network class. It follows the design document's guidelines when implementing the network.
    * The method works by using the dot product of a given node's previous activation layer and the connectivity pattern's
    * vector, then running that output through a threshold function. If the current activation layer is the input activation
    * layer, the method assigns the values read from the file.
    */
   void runNetwork(int trial)
   {
      for (int columns = 0; columns < activationLayers[0].length; columns++)
      {
         for (int row = 0; row < activationLayers.length; row++)
         {
            if (columns == 0)
            {
               activationLayers[row][columns] = inputs[trial][row];  // Populate the input activation layer with user-defined
               // input
               // values.
            }
            else
            {
               for (int i = 0; i < activationLayers.length; i++)
               {
                  activationLayers[row][columns] += this.thresholdFunction(activationLayers[i][columns - 1] *
                        connectivityLayers[columns - 1][i][row]); // Do the dot product and limit it with the thresholdFunction.
                                                                  // m will always be n - 1 in this case
               }
            }
         }
      }
   }


   /**
    * Method thresholdFunction limits the output of the dot product in the runNetwork method. Currently, it limits the
    * dot product with a sigmoid function.
    *
    * @param dotProductResult The dotProductsResult parameter is the output of the dot product between the two vectors, as
    *                         explained in the documentation for runNetwork. It will be limited by the threshold function.
    * @return The threshold function will return the limited dot product result. Currently, the function returns exactly what it
    * is given, and simply serves as a placeholder for future updates.
    */
   double thresholdFunction(double dotProductResult)
   {
      return dotProductResult;
    //  return 1 / (1 + Math.exp(-dotProductResult));
   }

// STEEPEST DESCENT CODE:

   double findTotalError(int trial)
   {
      double error = 0.0;
      for (int i = 0; i < outputNodes; i++)
      {
         error += 0.5 * ((expectedOutputs[trial][i] - activationLayers[i][activationLayers[0].length - 1]) *
               (expectedOutputs[trial][i] - activationLayers[i][activationLayers[0].length - 1]));
      }
      return Math.sqrt(error);
   }


   /**
    * Method steepestDescent creates a 3D array of the partial derivatives of weights. It then uses a lambda, or a
    * learning constant, to modify the current weights array to minimize their output. This method uses all of the same variable
    * names as the design document, for easier readability/modification.
    */
   void steepestDescent(double[] inputCases)
   {
      for (int trial = 0; trial < inputs.length; trial++)                               // For all training cases.
      {
         double lambda = 1.0;
         double sumHColumn = 0.0;
         double outputResult = activationLayers[0][activationLayers[0].length - 1];
         double error = expectedOutputs[trial][0] - outputResult;


         for (int j = 0; j < inputs.length; j++)                                        // For one training case.
         {
            for (int J = 0; J < activationLayers[0].length - 1; J++)
            {
               sumHColumn += activationLayers[J][1] * connectivityLayers[1][J][0];
            }

            double finalDeriv = -fDeriv(sumHColumn) * error * activationLayers[j][1];   // Partial for W_{j0}
            connectivityLayers[1][j][0] += -lambda * finalDeriv;
         }

         for (int k = 0; k < NUM_COLUMNS; k++)
         {
            for (int j = 0; j < NUM_COLUMNS; j++)
            {
               double sumAColumn = 0;
               for (int K = 0; K < activationLayers[0].length; K++)
               {
                  sumAColumn += activationLayers[K][0] * connectivityLayers[0][K][j];
               }

               double finalDeriv =
                     -activationLayers[k][0] * fDeriv(sumAColumn) * error * fDeriv(sumHColumn) * connectivityLayers[0][k][j];
               connectivityLayers[0][k][j] += -lambda * finalDeriv;
            }
         }


      }  // Iterates over all training cases
   }


   /**
    * Method fDeriv finds the derivative of the threshold function.
    *
    * @param input
    * @return
    */
   double fDeriv(double input)
   {
      return thresholdFunction(input) * (1 - thresholdFunction(input));
   }


} // public class SimpleNet


/*

public void steepestDescent()
    {
        double[][][] partialWeights = new double[numberActivationLayers - 1]
                                                [numberActivationLayers - 1]
                                                [numberActivationLayers - 1];
        double outputResult = activationLayers[0][activationLayers[0].length - 1];
        double fPrime = Math.exp(-outputResult);
        double sumHColumn = 0.0;
        for (int J = 0; J < activationLayers[0].length - 1; J++)
        {
            sumHColumn += activationLayers[J][1] * connectivityLayers[1][J][0];
        }
        // second column
        for (int j = 0; j < numberActivationLayers - 1; j++) {
            for (int i = 0; i < numberActivationLayers - 1; i++) {
                partialWeights[1][i][j] = (expectedOutputs[0] - outputResult) * fPrime * sumHColumn * activationLayers[1][0];
            }
        }
        // Add the sums of the products of the nodes in the A column and the weights in the A column to an array.
        double[] sumAColumns = new double[numberActivationLayers - 1];
        for (int K = 0; K < activationLayers[0].length; K++)
        {
            for (int i = 0; i < numberActivationLayers - 1; i++) {
                sumAColumns[i] += activationLayers[K][0] * connectivityLayers[0][K][i];

            }
        }
        // first column
        for (int k = 0; k < numberActivationLayers - 1; k++) {
            for (int j = 0; j < numberActivationLayers - 1; j++) {
                partialWeights[0][j][k] = -activationLayers[0][k] *             //A_{k}
                                          (expectedOutputs[0] - outputResult) * // T_{0} - F_{0}
                                          fPrime *
                                          sumHColumn *
                                          sumAColumns[j] *
                                          connectivityLayers[0][j][0];          // w_{j0}
            }
        }
    }


 */