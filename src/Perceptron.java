import java.io.BufferedReader;
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
    * Perceptron is the constructor for the neural network. It creates the activation layers and connectivity layers for the
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
      this.numberActivationLayers = NUM_COLUMNS + hiddenLayerNodes.length;  // add 2 to the number of hidden layers for the
                                                                            // total number of layers (1 input + n hidden + 1
                                                                            // output)
      this.inputs = new double[inputNodes];
      this.expectedOutputs = new double[outputNodes];
      maxNumberNodes = inputNodes;
      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         maxNumberNodes = Math.max(maxNumberNodes, hiddenLayerNodes[i]);    // Determined for building the activations
      }
      maxNumberNodes = Math.max(maxNumberNodes, outputNodes);

      // The first index of the activationLayers represents the number of nodes in the layer (limited by the maximum number of
      // nodes in the neural network, while the second index is the number of activation layers.
      activations = new double[numberActivationLayers][];
      activations[0] = new double[inputNodes];
      for (int i = 1; i < activations.length - 1; i++)
      {
         activations[i] = new double[hiddenLayerNodes[i - 1]];
      }
      activations[activations.length - 1] = new double[outputNodes];
      // The first index of the connectivity layer represents the current layer, the second index represents the source node, and
      // the third index represents the destination node.
      weights = new double[numberActivationLayers - 1][][];
      for (int i = 0; i < numberActivationLayers - 1; i++)
      {
         weights[i] = new double[activations[i].length][activations[i + 1].length];
      }
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
      inputarr2[1] = 0.0;
      inputarr[0] = 2;

      double[][][] testWeights;
      testWeights =
            new double[][][] {
               new double[][] {
                     new double[] {1.44, 0.49},
                     new double[] {1.25, 0.05}
               },
               new double[][] {
                     new double[] {0.22, 0.0},
                     new double[] {0.11, 0.0}
               }
            };
      Perceptron testNetwork = new Perceptron(2, inputarr, 1, "/Users/mihir/IdeaProjects/Neural Networks/Java XOR " +
            "Implementation/src/inputs/inputFile11.txt");
      testNetwork.randomizeWeights();
      testNetwork.setWeights(testWeights);
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
      for (int source = 0; source < inputs.length; source++)
      {
         activations[0][source] = inputs[source]; // Read inputs & modify input activations, 0 hardcoded for input activation layer
      }

      for (int n = 1; n < activations.length; n++)
      {
         for (int dest = 0; dest < activations[0].length; dest++) // source is the second index of the weights, either k or j
         {
            for (int source = 0; source < activations[0].length; source++)    // dest is the third index of the weights, either j or i
            {
               activations[n][dest] += f(activations[n - 1][source] * weights[n - 1][source][dest]);
            }
         }
      }
   }

   /**
    * Method thresholdFunction limits the output of the dot product in the runNetwork method. Currently, it limits the
    * dot product with a sigmoid function.
    *
    * @param x The dotProductsResult parameter is the output of the dot product between the two vectors, as
    *                         explained in the documentation for runNetwork. It will be limited by the threshold function.
    * @return The threshold function will return the limited dot product result. Currently, the function returns exactly what it
    * is given, and simply serves as a placeholder for future updates.
    */
   double f(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   /**
    * Method fDeriv finds the derivative of the threshold function.
    *
    * @param x The result of the dot product
    * @return a double value of the derivative of the threshold function at the input.
    */
   double fPrime(double x)
   {
      double thresholdOutput = f(x);
      return  thresholdOutput * (1.0 - thresholdOutput);
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
   public double[][][] findPartials(double truthValue)
   {
      double[][][] partials = new double[maxNumberNodes][maxNumberNodes][maxNumberNodes];

      double outputResult = activations[activations.length - 1][0];
      double error = truthValue - outputResult;

      double sumHColumn = 0.0;
      double sumAColumn = 0.0;

      for (int k = 0; k < maxNumberNodes; k++)  // iterate over source nodes
      {
         for (int j = 0; j < maxNumberNodes; j++)  // iterate over destination nodes
         {
            sumHColumn = 0.0;
            for (int J = 0; J < activations[0].length - 1; J++)
            {
               sumHColumn += (activations[1][J] * weights[1][J][0]);
            }

            double singleOutputPartial = -1.0 * error * fPrime(sumHColumn) * activations[1][j];   // Partial for W_{j0}
            partials[1][j][0] = singleOutputPartial;

            sumAColumn = 0;
            for (int K = 0; K < activations[0].length; K++)
            {
               sumAColumn += activations[0][K] * weights[0][K][j]; // First column
            }
            double multiOutputPartial = -1.0 * activations[0][k] * fPrime(sumAColumn) * error * fPrime(sumHColumn) * weights[1][j][0];
            partials[0][k][j] = multiOutputPartial;
         }
      }
      return partials;
   }

   void printResult()
   {
      System.out.println("Perceptron's result: " + activations[activations.length - 1][0]);
   }

   /**
    * Setter function for the weights variable.
    *
    * @param newWeights The new weights to replace the original ones.
    */
   public void setWeights(double[][][] newWeights)
   {
      for (int i = 0; i < this.weights.length; i++)
      {
         for (int j = 0; j < this.weights[0].length; j++)
         {
            for (int k = 0; k < this.weights[0][0].length; k++)
            {
               weights[i][j][k] = newWeights[i][j][k];
            }
         }
      }
   }

   /**
    * Calculate error calculates the perceptron's error in relation to a provided truth value, using the formula written in the
    * design document.
    *
    * @param truthValue The truth value, or expected output.
    * @return A double value for the error.
    */
   public double calculateError (double truthValue, double networkOutput)
   {
      return (truthValue - networkOutput) * (truthValue - networkOutput);
   }

}