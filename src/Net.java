import java.util.Arrays;
import java.util.Scanner;

/**
 *
 * @author Jai Bahri
 * @version 6 September 2019
 *
 * Net is a Neural Network that will create a model of a set amount of input nodes, output nodes, and nodes
 * in between representing connectivity layers in order to recognize patterns and be trained to pattern-match
 * data. 
 */
public class Net
{
   private double[][] activations;
   private double[][][] weights;
   private static final int NUM_COLUMNS = 2; //the number of columns of input and output nodes
   private int[] activeNodes;
   private double[][] inputs;
   private double[][] outputs;
   private int outputNodes;
   private static final double LAMBDA_FACTOR = 1.0;
   private static final double LAMBDA_VALUE = 1.0;
   private static final int ITERATION_THRESHOLD = 10000;

   /**
    * Creates an instance of the Net class.
    *
    * @param inputNodes          the amount of nodes in the first column of the Network
    * @param hiddenLayerNodes    the indirect nodes between the first and last column
    * @param outputNodes         the amount of nodes in the last column of the Network
    */
   public Net(int inputNodes, int[] hiddenLayerNodes, int outputNodes)
   {
      int max = hiddenLayerNodes[0];
      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         if (hiddenLayerNodes[i] > max)
            max = hiddenLayerNodes[i];

      } // s the largest column length
      this.outputNodes = outputNodes;

      activeNodes = new int[]{inputNodes, hiddenLayerNodes[0], outputNodes};
      activations = new double[hiddenLayerNodes.length + NUM_COLUMNS][max];
      activations = setActivations(activations);
      weights = setWeights(activations.length - 1, max, max);
   }

   /**
    * Instantiates a two dimensional array of active nodes.
    *
    * @param a    the 2D array that represents the activations
    * @return the activations array with all active nodes having bee1n set.
    */
   public double[][] setActivations(double[][] a)
   {
      for (int i = 0; i < a.length; i++)
      {
         for (int j = 0; j < a[i].length; j++)
            a[i][j] = 0.0;
      } // sets the active nodes in the activations array
      return a;
   }

   /**
    * Sets the weights to those in the Excel spreadsheet (AND) to test the Neural Net
    *
    * @param i    the length of the first dimension of the array
    * @param j    the length of the second dimension of the array
    * @param k    the length of the third dimension of the array
    * @return the modified weights array
    */
   public double[][][] setWeights(int i, int j, int k)
   {
      weights = new double[i][j][k];

      /**
       weights[0][0][0] = 1.44;
       weights[0][0][1] = 0.49;
       weights[0][1][0] = 1.25;
       weights[0][1][1] = 0.05;
       weights[1][0][0] = 0.22;
       weights[1][1][0] = 0.11;
       */

      weights[0][0][0] = 0.58;
      weights[0][0][1] = 0.59;
      weights[0][1][0] = 0.77;
      weights[0][1][1] = 0.90;
      weights[1][0][0] = 0.72;
      weights[1][1][0] = 0.82;
      return weights;
   }

   /**
    * Runs the Neural Net by setting the activation nodes.
    *
    * @param inputs  the activation nodes given by the user.
    */
   public double runNetwork(double[] inputs)
   {
      for(int inpInd = 0; inpInd < inputs.length; inpInd++) //n=0, put inputs into activations
      {
         activations[0][inpInd] = inputs[inpInd];
      }

      for (int layer = 1; layer < activations.length; layer++)
      {

         for(int node = 0; node < activeNodes[layer]; node++)
         {
            for(int i = 0; i < activeNodes[layer - 1]; i++)
            {
               activations[layer][node] += activations[layer-1][i] * weights[layer-1][i][node];
            }
            activations[layer][node] = sigmoid(activations[layer][node]);
         }

//         for (int row = 0; row < activations[col].length; row++)
//         {
//            for (int i = 0; i < activations.length; i++)
//            {
//               activations[row][col] += activations[i][col-1] * weights[col-1][i][row];
//            }
//            activations[row][col] = sigmoid(activations[row][col]);
//         }
      } // traverses through the activations array column-wise to set the nodes to the user's inputs
      double[] selectedOutputs = new double[activations[0].length];
//      for(int row = 0; row < activations.length; row++)
//      {
//         selectedOutputs[row] = activations[activations.length][0];
//      }

      return activations[activations.length - 1][0];

   }

   /**
    * Sigmoid wrapper function that will take in the dot product x and return f(x).
    *
    * @param x    the dot product of the weights
    * @return the result of f(x)
    */
   public double sigmoid(double x)
   {
      return 1.0/(1.0 + Math.exp(-x));
   }

   /**
    * Finds the derivative of a sigmoid function.
    *
    * @param x    the dot product of the weights
    * @return the derivative
    */
   public double differentiate(double x)
   {
      double result = sigmoid(x);
      return result*(1.0 - result);
   }
  
   /*
   public double findError()
   {
      double error = 0.0;
      for (int i = 0; i < outputNodes; i++)
         error += (outputs[i] + activations[NUM_COLUMNS-1][i]) * (outputs[i] + activations[NUM_COLUMNS-1][i]);
      return error*.5;
   }
   
   }
*/

   public double[][][] randomizeWeights()
   {
      for (int i = 0; i < weights.length; i++)
      {
         for (int j = 0; j < weights[i].length; j++)
         {
            for (int k = 0; k < weights[i][j].length; k++)
            {
               weights[i][j][k] = (.5-Math.random())*4;
            }
         }
      }
      return weights;
   }


   /**
    * Calculates the changes between the original weights and the current weights.
    *
    * @param weightst   a 3D array containing the original weights
    * @return the difference between the original and new weights arrays
    */
   public double[][][] findDeltaWeights(double[][][] weightst)
   {
      for (int i = 0; i < weights.length; i++)
      {
         for (int j = 0; j < weights[i].length; j++)
         {
            for (int k = 0; k < weights[i][j].length; k++)
               weightst[i][j][k] = Math.abs(weightst[i][j][k]) - Math.abs(weights[i][j][k]);
         }
      }
      return weightst;
   }

   /**
    * Saves the original weights into a new 3D array
    *
    * @return the original values of weights
    */
   public double[][][] copyWeights()
   {
      double[][][] weightst = new double[weights.length][weights[0].length][weights[0][0].length];
      for (int d1 = 0; d1 < weights.length; d1++) // dimension 1 of the 3D weights array
      {
         for (int d2 = 0; d2 < weights[d1].length; d2++) // dimension 2
         {
            for (int d3 = 0; d3 < weights[d1][d2].length; d3++) // dimension 3
            {
               weightst[d1][d2][d3] = weights[d1][d2][d3];
            }
         }
      } // creates a new array, weightst, with the same original values as weights
      return weightst;
   }


   /**
    * Trains the network to get the smallest errors and the optimal weights.
    * @param inputs     the desired input nodes
    * @param outputs    the desired outputs
    */
   public void optimize(double[][] inputs, double[][] outputs)
   {
      randomizeWeights();

      double[] errors = new double[outputs.length];
      double prevErr = Integer.MAX_VALUE;

      double lambda = LAMBDA_VALUE;

      boolean notZero = true;

      int iterations = 0;

      while (notZero || iterations < ITERATION_THRESHOLD)
      {
         for (int row = 0; row < inputs.length; row++)
         {

            double jSum = 0.0;
            double kSum = 0.0;

            double trained = runNetwork(inputs[row]);
            //System.out.println(Arrays.deepToString(activations));
            double error = (trained - outputs[row][0]);

            double[][][] weightst = copyWeights();    // 3D array to keep track of the changes in the weights.

            //System.out.println("error " + error);
           
           /*
           if (Math.abs(error) > Math.abs(trained[0]))
              lambda /= LAMBDA_FACTOR;
           else
              lambda *= LAMBDA_FACTOR;
           */

            for (int j = 0; j < activations[0].length; j++)
            {
               jSum = 0.0;
               for (int jSumInd = 0; jSumInd < activations[0].length; jSumInd++)
               {
                  jSum += activations[1][jSumInd]*weightst[1][jSumInd][0];
               }
               double deriv = error * differentiate(jSum) * activations[1][j];
               weights[1][j][0] += -deriv*lambda;
            } // finds the left hand side weights

            for (int k = 0; k < activations[0].length; k++)
            {
               for (int j = 0; j < activations[0].length; j++)
               {
                  kSum = 0.0;
                  for (int kSumInd = 0; kSumInd <= 1; kSumInd++)
                  {
                     kSum += activations[0][kSumInd]*weightst[0][kSumInd][j];
                  }
                  double deriv = activations[0][k]*differentiate(kSum)*error*differentiate(jSum)*weightst[1][j][0];
                  weights[0][k][j] += -deriv*lambda;
               }
            } // finds the right hand side weights

            // weightst = findDeltaWeights(weightst);

            /**

             if (error > prevErr)
             {
             lambda = lambda/prevErr;
             for (int d1 = 0; d1 < weights.length; d1++)
             {
             for (int d2 = 0; d2 < weights[d1].length; d2++)
             {
             for (int d3 = 0; d3 < weights[d1][d2].length; d3++)
             weights[d1][d2][d3] -= lambda;
             }
             }
             } // rolls back the weights

             */

            prevErr = error;
            errors[row] = Math.abs(error);
            //lambda %= 10;

         }
         double max = 0.0;

         for (int maxInd = 0; maxInd < errors.length; maxInd++)
         {
            if (Math.abs(errors[maxInd]) > Math.abs(max))
               max = errors[maxInd];
         }

         System.out.println("errors" + Arrays.toString(errors));
         System.out.println("max Error: " + max);

         double tot = 0.0;
         for(int i = 0; i < errors.length; i++)
         {
            tot += errors[i];
         }
         System.out.println("Total Error: " + tot);

         System.out.println("lambda: " + LAMBDA_VALUE);

         if(Math.abs(max) < .1)
            notZero = false;
         //System.out.println("Activations: " + Arrays.deepToString(activations));
         System.out.println("new weights: " + Arrays.deepToString(weights));
         iterations++;
      }
      if(iterations >= ITERATION_THRESHOLD)
         System.out.println("timed out");
      System.out.println(iterations + " iterations");
      System.out.println("inputs: " + Arrays.deepToString(inputs));
      System.out.println("desired outputs: " + Arrays.deepToString(outputs));

   }

   /**
    * Tests the Neural Net by creating a 2-2-1 connectivity layer.
    *
    * @param args    input arguments from the command line
    */
   public static void main(String[] args)
   {
      /**
       System.out.println("Please enter your first activation value.");
       Scanner in = new Scanner(System.in);

       double a1 = in.nextDouble();
       System.out.println("Please enter your next activation value.");

       double a2 = in.nextDouble();
       in.close();
       */

      Net finn = new Net(2, new int[] {4}, 1);
      System.out.println("activations: "+ Arrays.deepToString(finn.activations));
      System.out.println("original weights" + Arrays.deepToString(finn.weights));

      finn.optimize(new double[][] {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}}, new double[][] {{0.0},{0.0},{0.0},{1.0}});

      System.out.println(Arrays.deepToString(finn.activations));

      System.out.println(finn.runNetwork(new double[] {0.0, 0.0}));
      System.out.println(finn.runNetwork(new double[] {0.0, 1.0}));
      System.out.println(finn.runNetwork(new double[] {1.0, 0.0}));
      System.out.println(finn.runNetwork(new double[] {1.0, 1.0}));


   }

}