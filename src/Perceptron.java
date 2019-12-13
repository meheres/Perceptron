package src;

/**
 * @author Mihir Sharma
 * Created on Friday, November 6th (9.6.19)
 *
 * The Perceptron class is written to follow the documentation as closely as possible.
 *
 * The Perceptron class is a Java implementation of a basic neural network. The network attempts to solve a basic logic problem,
 * either AND, OR, or XOR, using a two-connectivity network with three layers of activation. The neural network takes in a list of double inputs,
 * then uses the provided weights (or randomized ones) to caluculate vector dot products and provide a vector of double outputs.
 * The Perceptron should be constructed within the Trainer.java class, as a basic perceptron during the optimization of weights.
 *
 * Functions in this class include the following:
 * - runNetwork, which uses the inputs and weights to run the Simple Network for an AND logic table.
 * - backProp, which runs the back propagation algorithm to modify the weights of the Perceptron, based on a set of expected outputs.
 * - f, which follows the documentation's notation for the threshold function.
 * - fPrime, which is the derivative of the threshold function.
 * - randomizeWeights, which uses a lower bound and an upper bound to randomize the Perceptron's weights.
 * - printResult, which provides some information by printing the result and the expected output.
 * - calculateError, which calculates the root of the sum of the squares as shown in the documentation. (Ti - Fi)^2.
 */
public class Perceptron
{
   int inputNodes;                    // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;            // The number of nodes in each hidden activation layer.
   int outputNodes;                   // The number of nodes in the output activation layer.

   static final int NUM_COLUMNS = 2;  // One for the input layer, one for the output layer.

   int numberActivationLayers;        // The number of connectivity layers will be one less than the number of activation layers.
                                      // The number of activation layers will be 2 more than the number of hidden layers, one for
                                      // input activations and one for output activations.

   double[] expectedOutputs;          // An Array that holds the values for the expected outputs, for comparison with the actual outputs.
   double[][] activations;            // A 2D Array that represents the different activation layers. First index will be the number of activation
                                      // layers, and the second index will specify which node from the activation layer to use.

   // The following variables are used in back propagation. They match the documentation exactly.
   double[][] theta;
   double[][] omega;
   double [][] psi;
   double lambda;

   public double[][][] weights;        // A 3D Array that represents the connectivity layers.

   public double[][][] partials;       // Partials 3D Array, may not be necessary, but left in because "you touch it, you break it!"

   /**
    * BPerceptron is the constructor for the neural network. It creates the activation layers and connectivity layers for the
    * network, as well as the arrays for all of the variables used during back propagation.
    *
    * @param inputNodes The number of input activations.
    * @param hiddenLayerNodes An array containing the number of activations in each hidden layer.
    * @param outputNodes The number of output activations.
    */
   public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes)
   {
      this.inputNodes = inputNodes;
      this.hiddenLayerNodes = hiddenLayerNodes;
      this.outputNodes = outputNodes;
      this.numberActivationLayers = NUM_COLUMNS + hiddenLayerNodes.length;  // add 2 to the number of hidden layers for the
                                                                            // total number of layers (1 input + n hidden + 1 output)
      this.expectedOutputs = new double[outputNodes];
      activations = new double[numberActivationLayers][];   // The indices of the activations array matches the documentation exactly. The first
                                                            // index represents the current layer, and the second index represents the current node.

      theta = new double[numberActivationLayers - 1][];     // Theta, omega, and psi are not computed for the input layer,
      omega = new double[numberActivationLayers - 1][];     // so the lengths are one less than the length of activations.
      psi = new double[numberActivationLayers - 1][];

      activations[0] = new double[inputNodes];

      for (int i = 1; i < activations.length - 1; i++)
      {
         activations[i] = new double[hiddenLayerNodes[i - 1]];
         theta[i - 1] = new double[hiddenLayerNodes[i - 1]];
         omega[i - 1] = new double[hiddenLayerNodes[i - 1]];
         psi[i - 1] = new double[hiddenLayerNodes[i - 1]];
      }
      activations[activations.length - 1] = new double[outputNodes];
      theta[activations.length - 2] = new double[outputNodes];
      omega[activations.length - 2] = new double[outputNodes];
      psi[activations.length - 2] = new double[outputNodes];


      weights = new double[numberActivationLayers - 1][][];       // The first index of the connectivity layer represents the current layer,  the
                                                                  // second index represents the source node, and the third index represents the
                                                                  // destination node, as specified in the documentation.

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
      } // for (int i = 0; i < hiddenLayerNodes.length; i++)
      weights[hiddenLayerNodes.length] = new double[hiddenLayerNodes[hiddenLayerNodes.length - 1]][outputNodes];
      partials = new double[weights.length][][];
      for (int m = 0; m < weights.length; m++)
      {
         for (int i = 0; i < weights[m].length; i++)
         {
            partials[m] = new double[weights[m].length][weights[m][i].length];
         }
      }

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
      // Set inputs
      for (int source = 0; source < inputs.length; source++)
      {
         activations[0][source] = inputs[source];        // Read inputs & modify input activations, the 0 is hardcoded for the input activation layer.
      }
      // Forward prop
      for (int n = 1; n < activations.length; n++)
      {
         for (int dest = 0; dest < activations[n].length; dest++)
         {
            double sumActivations = 0.0;
            for (int source = 0; source < activations[n - 1].length; source++)
            {
               sumActivations += activations[n - 1][source] * weights[n - 1][source][dest];
            }                                            // We can save theta_i before we take the derivative to calculate h_j.
            theta[n - 1][dest] = sumActivations;         // Calculate theta_i and h_j during forward propagation.
            activations[n][dest] = f(sumActivations);
         }  // for (int dest = 0; dest < activations[n].length; dest++)
      }     // for (int n = 1; n < activations.length; n++)
   }


   /**
    * Method backProp follows the documentation to use back propagation to modify the perceptron's weights based upon a set of expected outputs.
    * There are three cases: the output layer, the hidden layers, and the input layer.
    * For the output layer, formulae are slightly different (no loops).
    * For the input layer, only partials and weights need to be calculated.
    *
    * @param truth The set of expected outputs.
    */
   public void backProp(double[] truth)
   {
      int n = activations.length - 1;
      for (int source = 0; source < activations[n].length; source++) // OUTPUT LAYER
      {
         omega[n - 1][source] = truth[source] - activations[n][source];
         psi[n - 1][source] = omega[n - 1][source] * fPrime(theta[n - 1][source]);
      }
      for (n = activations.length - 2; n > 0; n--) // HIDDEN LAYERS
      {
         for (int source = 0; source < activations[n].length; source++)
         {
            omega[n - 1][source] = 0;
            for (int dest = 0; dest < activations[n + 1].length; dest++) // index "I" in our 3-layer network
            {
               omega[n - 1][source] += psi[n][dest] * weights[n][source][dest];
               partials[n][source][dest] = lambda * activations[n][source] * psi[n][dest];
               weights[n][source][dest] += partials[n][source][dest];
            }
            psi[n - 1][source] = omega[n - 1][source] * fPrime(theta[n - 1][source]);
         } // for (int source = 0; source < activations[n].length; source++)
      }    // for (n = activations.length - 2; n > 0; n--)
      n = 0;
      for (int source = 0; source < activations[n].length; source++) // INPUT LAYER
      {
         for (int dest = 0; dest < activations[n + 1].length; dest++) // index "I" in our 3-layer network
         {
            partials[n][source][dest] = lambda * activations[n][source] * psi[n][dest];
            weights[n][source][dest] += partials[n][source][dest];
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
    * @param lowValue the lowest value that the random weights can go to.
    * @param highValue the highest value that the random weights can go to.
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
               weights[i][j][k] = (diff * Math.random()) + lowValue;
            }
         }
      }
   }

   /**
    * Method printResult prints all of the perceptron's outputs.
    */
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
    * Calculate error calculates the perceptron's error in relation to a provided truth value, using the formula written in the
    * design document.
    *
    * @param truthValue The truth value, or expected output.
    * @param networkOutput The network's calculated output.
    * @return A double value for the error.
    */
   public double calculateError (double truthValue, double networkOutput)
   {
      return (truthValue - networkOutput) * (truthValue - networkOutput);
   }

}
