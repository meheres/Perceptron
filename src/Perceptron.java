
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.StringTokenizer;

/**
 * @author Mihir Sharma
 * Created on Friday, November 6th (9.6.19)
 * <p>
 * The Perceptron class is written to follow the documentation as closely as possible.
 * <p>
 * The Perceptron class is a Java implementation of a basic neural network. The network attempts to solve a basic logic problem,
 * either AND, OR, or XOR, using a two-connectivity network with three layers of activation. The neural network takes in a list of double inputs,
 * then uses the provided weights (or randomized ones) to caluculate vector dot products and provide a vector of double outputs.
 * The Perceptron should be constructed within the Trainer.java class, as a basic perceptron during the optimization of weights.
 * <p>
 * Functions in this class include the following:
 * - main, which instantiates the Simple Network and runs the other two functions. The main also prints the output and its expected comparison.
 * - runNetwork, which uses the inputs and weights to run the Simple Network for an AND logic table.
 * - f, which follows the documentation's notation for the threshold function.
 * - fPrime, which is the derivative of the threshold function.
 * - randomizeWeights, which uses a lower bound and an upper bound to randomize the Perceptron's weights.
 * - findPartials, which calculates the partial derivatives as described in the documentation.
 * - printResult, which provides some information by printing the result and the expected output.
 * - setWeights, which allows to easily modify all of the Perceptron's weights at once.
 * - calculateError, which calculates the root of the sum of the squares as shown in the documentation. (Ti - Fi)^2.
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

   double[] inputs;                   // An Array that holds the values for the input activations. Read in first line of input file.

   double[] expectedOutputs;          // An Array that holds the values for the expected outputs, for comparison with the actual outputs.

   double[][] activations;            // A 2D Array that represents the different activation layers. First index will be the
   // number of activation layers, and the second index will specify which node from the
   // activation layer to use.

   public double[][][] weights;       // A 3D Array that represents the connectivity layers.

   public double[][][] partials;      // Partials 3D Array, to avoid the heap allocation problems

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
      // total number of layers (1 input + n hidden + 1 output)
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

      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         if (i == 0)
         {
            weights[i] = new double[inputNodes][hiddenLayerNodes[i]];
         }
         else
         {
            weights[i] = new double[hiddenLayerNodes[i - 1]][hiddenLayerNodes[i]];
         }
      }
      weights[hiddenLayerNodes.length] = new double[hiddenLayerNodes[hiddenLayerNodes.length - 1]][outputNodes];
      partials = new double[weights.length][][];
      for (int m = 0; m < weights.length; m++)
      {
         for (int i = 0; i < weights[m].length; i++)
         {
            partials[m] = new double[weights[m].length][weights[m][i].length]; // Creates weights array via jagged array
         }
      }

   }




   /**
    * NOTE: THIS METHOD NEVER GETS USED. IT IS FOR TESTING PURPOSES ONLY.
    *
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

      double[][][] testWeights =
            new double[][][] {
                  new double[][] {
                        new double[] {0.739674399, 1.665815801},
                        new double[] {1.321837369, 1.8005763}
                  },
                  new double[][] {
                        new double[] {0.089064485},
                        new double[] {0.341737526}
                  }
            };
      Perceptron testNetwork = new Perceptron(2, inputarr, 1, "old/inputsFile.txt");
      testNetwork.setWeights(testWeights);
      testNetwork.runNetwork(inputarr2);
      testNetwork.printResult();
      System.out.println("Weights: " + Arrays.deepToString(testNetwork.weights));
      System.out.println("Activations: " + Arrays.deepToString(testNetwork.activations));


   }



   /**
    * Method runNetwork runs the Simple Network class. It first takes in inputs then runs the network. It follows the design
    * document's guidelines when implementing the network.
    * The method works by using the dot product of a given node's previous activation layer and the connectivity pattern's
    * vector, then running that output through a threshold function. If the current activation layer is the input activation
    * layer, the method assigns the values read from the file.
    *
    * @param inputs A double array of the input activations for the network that is to be run.
    */
   public void runNetwork(double[] inputs)
   {
      for (int source = 0; source < inputs.length; source++)
      {
         activations[0][source] = inputs[source];                                 // Read inputs & modify input activations, 0 hardcoded for
      }                                                                           // input activation layer


      for (int n = 1; n < activations.length; n++)
      {
         for (int dest = 0; dest < activations[n].length; dest++)                 // source is the second index of the weights, either k or j
         {
            double sumActivations = 0.0;
            for (int source = 0; source < activations[n - 1].length; source++)    // dest is the third index of the weights,
            // either j or i
            {
               sumActivations += activations[n - 1][source] * weights[n - 1][source][dest];
            }
            activations[n][dest] = f(sumActivations);
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
    *
    * @param highValue the highest value that the random weights can go to.
    * @param lowValue the lowest value that the random weights can go to.
    */
   void randomizeWeights(double lowValue, double highValue)
   {
      double diff = highValue - lowValue;
      for (int i = 0; i < weights.length; i++)
      {
         for (int j = 0; j < weights[i].length; j++)
         {
            for (int k = 0; k < weights[i][j].length; k++)
            {
               weights[i][j][k] = (diff * Math.random()) - lowValue;
            }
         }
      }
      System.out.println(Arrays.deepToString(weights));
   }

   /**
    * Find the partial derivatives for the gradient descent, then add them to the trial set of weights. Currently capable of
    * determining partials for multiple output nodes.
    */
   public double[][][] findPartials(double[] truthValues)
   {

      double[] outputs = activations[activations.length - 1];                    // Output nodes

      for (int i = 0; i < outputNodes; i++)       // First calculate partials for the output layer (rightmost weights layer)  Recall,
      {                                           // weights[i][j][k] --> i is the layer, j is the source and k is the destination
         double sumHColumn = 0;
         for (int J = 0; J < hiddenLayerNodes[0]; J++)
         {
            sumHColumn += activations[1][J] * weights[1][J][i];
         }
         for (int j = 0; j < hiddenLayerNodes[0]; j++)
         {
            partials[1][j][i] = -(truthValues[i] - outputs[i]) * fPrime(sumHColumn) * activations[1][j];
         }
      } // for (int i = 0; i < outputNodes; i++)

      for (int j = 0; j < hiddenLayerNodes[0]; j++)                                     // Now calculate the previous/first layer
      {
         double sumI = 0;
         for (int I = 0; I < outputNodes; I++)
         {
            double sumJ = 0;
            for (int J = 0; J < hiddenLayerNodes[0]; J++)
            {
               sumJ += activations[1][J] * weights[1][J][I];                            // This is sigma(h_j*w_JI)
            }
            sumI += (truthValues[I] - outputs[I]) * fPrime(sumJ) * weights[1][j][I];    // Calculate sigma((T_I-F_I)*f'(sumJ)*w_jI
         } // for (int I = 0; I < outputNodes; I++)
         for (int k = 0; k < inputNodes; k++)
         {
            double sumK = 0;
            for (int K = 0; K < inputNodes; K++)
            {
               sumK += activations[0][K] * weights[0][K][j];
            }
            partials[0][k][j] = -activations[0][k] * fPrime(sumK) * sumI;
         } // for (int k = 0; k < inputNodes; k++)
      } // for (int j = 0; j < hiddenLayerNodes[0]; j++)
      return partials;
   }

   void printResult()
   {
      for (int i = 0; i < outputNodes; i++)
      {
         System.out.println("Perceptron's output [" + (i + 1) + "]: " + activations[activations.length - 1][i] + "; Expected " +
               "Output " + expectedOutputs[i]);
      }
      System.out.println();
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
         for (int j = 0; j < this.weights[i].length; j++)
         {
            for (int k = 0; k < this.weights[i][j].length; k++)
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
