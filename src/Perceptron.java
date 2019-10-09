import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

/**
 * @author Mihir Sharma
 * @version 1.0
 * Created on Friday, 9/6/19
 * <p>
 * The SimpleNet class is a Java implementation of a basic neural network. The network attempts to solve a basic logic problem,
 * either AND, OR, or XOR, using a two-connectivity network with three layers of activation. The neural network will take in two
 * boolean inputs, represented by 1 (true) and 0 (false). It will output what it believes the answer to the logic problem based
 * on the inputs should be. When constructing the Simple Network, there should be an input file that provides the inputs,
 * weights, and expected outputs.
 * <p>
 * Functions in this class include the following:
 * - readInput, which reads the input from the constructor's filename then populates the inputs, weights, and expected outputs.
 * - runNetwork, which uses the inputs and weights to run the Simple Network for an AND logic table.
 * - main, which instantiates the Simple Network and runs the other two functions. The main also prints the output and its
 * expected comparison.
 */
public class Perceptron
{

   int numTrials;                     // The number of trials to be run.
   int minimumError;                  // The minimum error for training.
   int inputNodes;                    // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;            // The number of nodes in each hidden activation layer.
   int outputNodes;                   // The number of nodes in the output activation layer.
   int maxNumberNodes;                // The maximum number of nodes used in the perceptron.

   static final int NUM_COLUMNS = 2;  // One for the input layer, one for the output layer.

   BufferedReader bufferedReader;     // bufferedReader is for the input file. It allows for quick line-by-line reading of the file.
   StringTokenizer stringTokenizer;   // stringTokenizer reads each line provided by the BufferedReader, allowing for
   // token-by-token reading.

   int numberActivationLayers;        // The number of connectivity layers will be one less than the number of activation layers.
   // The number of activation layers will be 2 more than the number of hidden layers, one for
   // input activations and one for output activations.

   double[] inputs;                   // An Array that holds the values for the input activations. Read in first line of input
   // file.

   double[] expectedOutputs;          // An Array that holds the values for the expected outputs, for comparison with the actual
   // outputs.

   double[][] activations;            // A 2D Array that represents the different activation layers. First index will be the
   // number of activation layers, and the second index will specify which node from the
   // activation layer to use.

   public double[][][] weights;       // A 3D Array that represents the connectivity layers.


   /**
    * SimpleNet is the constructor for the neural network. It creates the activation layers and connectivity layers for the
    * network. The constructor also determines the most number of nodes by iterating over all of the activation layers.
    *
    * @param inputNodes
    * @param hiddenLayerNodes
    * @param outputNodes
    */
   public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes, String inputFileName)
   {
      this.inputNodes = inputNodes;
      this.hiddenLayerNodes = hiddenLayerNodes;
      this.outputNodes = outputNodes;
      this.numberActivationLayers = 2 + hiddenLayerNodes.length;  // add 2 to the number of hidden layers for the total number
      // of layers (1 input + n hidden + 1 output)
      this.inputs = new double[inputNodes];
      this.expectedOutputs = new double[outputNodes];
      maxNumberNodes = inputNodes;
      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         maxNumberNodes = Math.max(maxNumberNodes, hiddenLayerNodes[i]);
      }
      maxNumberNodes = Math.max(maxNumberNodes, outputNodes);

      // the first index of the activationLayers represents the number of nodes in the layer (limited by the maximum number of
      // nodes in the neural network, while the second index is the number of activation layers.
      activations = new double[maxNumberNodes][numberActivationLayers];
      // the first index of the connectivity layer represents the
      weights = new double[numberActivationLayers - 1]
                          [numberActivationLayers - 1]
                          [numberActivationLayers - 1];
   }




   /**
    * The main method is the manner in which the Simple Network is run. Currently, it creates a 2-2-1 network and then runs the
    * readInput and runNetwork methods on that network. Finally, the network outputs its result for the test value. Currently,
    * it is limited by the input file -- in the event that the input file is formatted incorrectly, the program will shut down.
    *
    * @param args The arguments for the main method.
    */
   public static void main(String[] args) throws IOException
   {
      int[] inputarr = new int[1];
      double[] inputarr2 = new double[2];
      inputarr2[0] = 1.0;
      inputarr2[1] = 1.0;
      inputarr[0] = 2;
      Perceptron testNetwork = new Perceptron(2, inputarr, 1, "/Users/mihir/IdeaProjects/Neural Networks/Java XOR " +
            "Implementation/src/inputs/inputFile11.txt");
      testNetwork.randomizeWeights();
      testNetwork.runNetwork(inputarr2);
      testNetwork.printResult();
   }



   /**
    * Method runNetwork runs the Simple Network class. It first takes in inputs then runs the network. It follows the design
    * document's guidelines when implementing the network.
    * The method works by using the dot product of a given node's previous activation layer and the connectivity pattern's
    * vector, then running that output through a threshold function. If the current activation layer is the input activation
    * layer, the method assigns the values read from the file.
    */
   public void runNetwork(double[] inputs)
   {
      for (int n = 0; n < inputs.length; n++)
      {
         activations[n][0] = inputs[n];                  // Read inputs and modify input activations
      }

      for (int n = 1; n < maxNumberNodes; n++)
      {
         for (int source = 0; source < maxNumberNodes; source++) // source is the second index of the weights, either k or j
         {
            for (int dest = 0; dest < maxNumberNodes; dest++)    // dest is the third index of the weights, either j or i
            {
               activations[n][source] += thresholdFunction(activations[n - 1][source] * weights[n - 1][source][dest]);
            }
         }
      }
      

     /* for (int k = 0; k < activations[0].length; k++)
      {
         for (int row = 0; row < activations.length; row++)
         {
            if (k == 0)
            {
               activations[row][k] = inputs[row];
            }
            else
            {
               for (int i = 0; i < activations.length; i++)
               {
                  activations[row][k] +=
                        thresholdFunction(activations[i][k - 1] * weights[k - 1][i][row]); // m will always be n - 1 in this case
               }
            }
         }
      }*/
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
      return 1 / (1 + Math.exp(-dotProductResult));
   }

   /**
    * Method fDeriv finds the derivative of the threshold function.
    *
    * @param input The result of the dot product
    * @return a double value of the derivative of the threshold function at the input.
    */
   double fDeriv(double input)
   {
      double thresholdOutput = thresholdFunction(input);
      return  thresholdOutput * (1 - thresholdOutput);
   }

   /**
    * Randomizes the weights in the perceptron.
    */
   void randomizeWeights()
   {
      for (int i = 0; i < weights.length; i++)
      {
         for (int j = 0; j < weights[0].length; j++)
         {
            for (int k = 0; k < weights[0][0].length; k++)
            {
               weights[i][j][k] = 1.0 * (Math.random());
            }
         }
      }
   }

   /**
    * Find the partial derivatives for the gradient descent, then add them to the trial set of weights.
    */
   public double[][][] findPartials()
   {
      double[][][] partials = new double[maxNumberNodes][maxNumberNodes][maxNumberNodes];


      double outputResult = activations[0][activations[0].length - 1];
      double error = expectedOutputs[0] - outputResult;

      for (int j = 0; j < inputs.length; j++)                                   // For one training case.
      {
         double sumHColumn = 0.0;
         for (int J = 0; J < activations[0].length - 1; J++)
         {
            sumHColumn += activations[J][1] * weights[1][J][0];
         }

         double finalDeriv = -fDeriv(sumHColumn) * error * activations[j][1];   // Partial for W_{j0}
         partials[1][j][0] = finalDeriv;
      }

      for (int k = 0; k < maxNumberNodes; k++)
      {
         for (int j = 0; j < maxNumberNodes; j++)
         {
            double sumAColumn = 0;
            double sumHColumn = 0.0;
            for (int K = 0; K < activations.length; K++)
            {
               sumAColumn += activations[K][0] * weights[0][K][j];
               sumHColumn += activations[K][1] * weights[1][K][0];
            }

            double finalDeriv =
                  -activations[k][0] * fDeriv(sumAColumn) * error * fDeriv(sumHColumn) * weights[0][k][j];

            partials[0][k][j] = finalDeriv;
         }
      }
      return partials;
   }

   void printResult()
   {
      System.out.println("Perceptron's result: " + activations[0][activations[0].length - 1]);
   }

   /**
    * Setter function for the weights variable.
    *
    * @param weights The new weights to replace the original ones.
    */
   public void setWeights(double[][][] weights)
   {
      this.weights = weights;
   }
}