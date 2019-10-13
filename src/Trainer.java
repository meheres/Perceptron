import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;


public class Trainer
{
   Perceptron perceptron;
   String inputFile;
   static final double MINIMUM_ERROR = 0.01;
   static final int MAX_STEPS = 100;
   double[][] testCases;
   double currCase;
   int counter;                               // Counter checks the number of steps and provides the current test case (counter %
                                              // testCases.length)

   double[][] truths;                         // A truth is an expected output.

   double lambda;                             // Training function
   double currError;
   double previousError;
   double[][][] optimizedWeights;             // Optimized thus far.
   double[][][] experimentalWeights;          // Experimental, could be optimal, may not be.

   int inputNodes;                            // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;                    // The number of nodes in each hidden activation layer.
   int outputNodes;                           // The number of nodes in the output activation layer.

   /**
    * Creates a new trainer for a perceptron, with a list of input filenames.
    *
    * @param inputFile The list of input file names. Must be a full path.
    */
   public Trainer(String inputFile)
   {
      this.inputFile = inputFile;
      readInputFile();
      this.currError = 1.0;                                                            // Total and Previous Total errors start
      this.previousError = 0.0;                                                        // as 0.0.
      this.lambda = 0.5;
      this.counter = 0;

      perceptron = new Perceptron(2, new int[]{2}, 1, inputFile);
      perceptron.randomizeWeights();

      this.optimizedWeights = new double[perceptron.weights.length]                    // Create the currentWeights, copying the
            [perceptron.weights[0].length]                 // values from the perceptron's set of
            [perceptron.weights[0][0].length];             // weights the first time.

      for (int i = 0; i < perceptron.weights.length; i++)
      {
         for (int j = 0; j < perceptron.weights[0].length; j++)
         {
            for (int k = 0; k < perceptron.weights[0][0].length; k++)
            {
               this.optimizedWeights[i][j][k] = perceptron.weights[i][j][k];
            }
         }
      }

      this.experimentalWeights = new double[perceptron.weights.length]
            [perceptron.weights[0].length]
            [perceptron.weights[0][0].length];
      for (int i = 0; i < perceptron.weights.length; i++)
      {
         for (int j = 0; j < perceptron.weights[0].length; j++)
         {
            System.arraycopy(this.optimizedWeights[i][j], 0, this.experimentalWeights[i][j], 0, perceptron.weights[0][0].length);
         }
      }
   }

   /**
    * Method readInputFile reads all the input as specified in the README file and updates the class level variables.
    */
   public void readInputFile()
   {
      BufferedReader bufferedReader;
      StringTokenizer stringTokenizer;

      try
      {
         bufferedReader = new BufferedReader(new FileReader(inputFile));

         inputNodes = Integer.parseInt(bufferedReader.readLine());  // Find number of input nodes

         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
         hiddenLayerNodes = new int[Integer.parseInt(stringTokenizer.nextToken())];
         for (int i = 0; i < hiddenLayerNodes.length; i++)
         {
            hiddenLayerNodes[i] = Integer.parseInt(stringTokenizer.nextToken());  // Populate number of nodes per hidden layer
         }

         outputNodes = Integer.parseInt(bufferedReader.readLine());  // Find number of output nodes

         int numberCases = Integer.parseInt(bufferedReader.readLine());  // Find the number of test cases
         testCases = new double[numberCases][];
         truths = new double[numberCases][];
         for (int i = 0; i < numberCases; i++)
         {
            stringTokenizer = new StringTokenizer(bufferedReader.readLine());
            testCases[i] = new double[inputNodes];
            for (int j = 0; j < inputNodes; j++)
            {
               testCases[i][j] = Double.parseDouble(stringTokenizer.nextToken());
            }
            truths[i] = new double[outputNodes];
            for (int j = 0; j < outputNodes; j++)
            {
               truths[i][j] = Double.parseDouble(stringTokenizer.nextToken());
            }
         }     // Iterating over the number of test cases
      }        // Reads the input file.
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " not accepted, terminating.");
      }
   }



   /**
    * Function copyOldToNew is a helper function to copy the old weights to new ones. Will be used in each step.
    */
   public void copyWeights(double[][][] oldWeights, double[][][] newWeights)
   {
      for (int i = 0; i < perceptron.weights.length; i++)
      {
         for (int j = 0; j < perceptron.weights[0].length; j++)
         {
            System.arraycopy(oldWeights[i][j], 0, newWeights[i][j], 0, perceptron.weights[0][0].length);
         }
      }
   }

   /**
    * Main method for the Trainer reads the input files and then trains the network.
    *
    * @param args an array of String arguments for the main method
    */
   public static void main(String[] args)
   {
      Trainer trainer = new Trainer("/Users/mihir/IdeaProjects/Neural Networks/Java XOR Implementation/src/old/inputsFile.txt");
      trainer.train();

   }

   /**
    * Function train makes steps and modifies the perceptron's weights using gradient descent until one of three conditions are met:
    * 1. The error is the exact same as the error from the previous trial,
    * 2. The error is less than a minimum error, or
    * 3. The training is capped by a predetermined counter.
    * Only randomizes the perceptron's weights the first time, after that it does not.
    */
   void train()
   {
      while (currError > MINIMUM_ERROR && currError != previousError && counter < MAX_STEPS)
      {
         currCase = truths[counter % testCases.length][0];
         step();
         counter++;
      }
   }

   /**
    * Function step runs an individual step in the training process. A step is defined as finding the partials, using the error
    * to update the weights of the perceptron, setting weights/running the network, then deciding whether or not to use the
    * updated weights.
    */
   public void step()
   {
      double[][][] currPartialWeights = perceptron.findPartials(currCase);
      copyWeights(experimentalWeights, optimizedWeights);
      for (int i = 0; i < experimentalWeights.length; i++)
      {
         for (int j = 0; j < experimentalWeights[0].length; j++)
         {
            for (int k = 0; k < experimentalWeights[0][0].length; k++)
            {
               experimentalWeights[i][j][k] += -1.0 * lambda * currPartialWeights[i][j][k]; // Updates currWeights to new weights
            }
         }
      }

      // double[][][] lambdasChoice = adaptLambda();  TODO:FUTURE WORK

      perceptron.setWeights(experimentalWeights);
      perceptron.runNetwork(testCases[counter % testCases.length]);
      perceptron.printResult();

      double newError = perceptron.calculateError(truths[counter % testCases.length][0]);
      System.out.println("Error: " + newError);

      /*if (newError > currError)
      {
         perceptron.setWeights(optimizedWeights);
         copyWeights(optimizedWeights, experimentalWeights);
      }
      else
      {
         perceptron.setWeights(experimentalWeights);
      }*/
   }


   /**
    * Function adaptLambda
    */
   public void adaptLambda()
   {
      return;
      /*if(totalError < previousTotalError)
      {
         lambda *= 2.0;
      }
      else
      {
         lambda /= 2.0;
      }*/
   }

}
