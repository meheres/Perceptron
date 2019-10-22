import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.StringTokenizer;


public class Trainer
{
   Perceptron perceptron;
   String inputFile;
   static final double MINIMUM_ERROR = 0.01;
   static final int MAX_STEPS = 10000;
   double[][] testCases;
   double currCase;
   int counter;                               // Counter checks the number of steps and provides the current test case (counter %
                                              // testCases.length)

   double[][] truths;                         // A truth is an expected output.

   double lambda;                             // Training function
   double currError;
   double previousError;

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
      this.currError = Double.MAX_VALUE - 1.0;                                // Current error starts as larger as possible
      this.previousError = Double.MIN_VALUE;                                  // Previous error needs to be different from current
      this.lambda = 0.5;
      this.counter = 0;
      perceptron = new Perceptron(2, new int[]{2}, 1, inputFile);
      perceptron.randomizeWeights();                                          // Randomize weights before first use

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
         testCases = new double[numberCases][inputNodes]; //JM we already know the dimension. added inputNodes
         truths = new double[numberCases][outputNodes]; //JM we already know the dimension. added outputNodes
         for (int i = 0; i < numberCases; i++)
         {
            stringTokenizer = new StringTokenizer(bufferedReader.readLine());
            // testCases[i] = new double[inputNodes][]; //JM added []
            for (int j = 0; j < inputNodes; j++)
            {
               testCases[i][j] = Double.parseDouble(stringTokenizer.nextToken());
            }
            // truths[i] = new double[outputNodes][]; //JM added []
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
      perceptron.randomizeWeights();

      while (currError > MINIMUM_ERROR && counter < MAX_STEPS)
      {
         currCase = truths[counter % testCases.length][0];
         System.out.println("Current test case: " + currCase); //TODO: REMOVE DEBUG STATEMENT
         step();
         counter++;
         System.out.println("Counter: "+ counter + "\nCounter % #testcases: " + counter % testCases.length); //TODO: REMOVE DEBUG STATEMENT
      }
      System.out.println("End condition met, Terminating");
   }

   /**
    * Function step runs an individual step in the training process. A step is defined as finding the partials, using the error
    * to update the weights of the perceptron, setting weights/running the network, then deciding whether or not to use the
    * updated weights.
    */
   public void step()
   {
      perceptron.runNetwork(testCases[counter % testCases.length]); // Train current perceptron
      System.out.print("Pre-updated result: ");
      perceptron.printResult();

      double[][][] currPartialWeights = perceptron.findPartials(currCase);
      System.out.println("Partials : " + Arrays.deepToString(currPartialWeights)); //TODO: REMOVE DEBUG STATEMENT

      for (int i = 0; i < currPartialWeights.length; i++)
      {
         for (int j = 0; j < currPartialWeights[i].length; j++)
         {
            for (int k = 0; k < currPartialWeights[i][j].length; k++)
            {
               perceptron.weights[i][j][k] += -lambda * currPartialWeights[i][j][k]; // Updates currWeights to new weights
            }
         }
      }

      // double[][][] lambdasChoice = adaptLambda();  TODO: FUTURE WORK ON ADAPTIVE LAMBDA
      // perceptron.setWeights(experimentalWeights);
      perceptron.runNetwork(testCases[counter % testCases.length]);
      System.out.print("Post-updated result: ");
      perceptron.printResult();
      double trainedResult = perceptron.activations[perceptron.activations.length - 1][0];

      double newError = perceptron.calculateError(truths[counter % testCases.length][0], trainedResult);
      previousError = currError;
      currError = newError;
      System.out.println("Previous Error: " + previousError); //TODO: REMOVE DEBUG STATEMENT
      System.out.println("New Error: " + currError); //TODO: REMOVE DEBUG STATEMENT
      System.out.println(); //TODO: REMOVE DEBUG STATEMENT

      if(previousError < currError)
      {
         for (int i = 0; i < currPartialWeights.length; i++)
         {
            for (int j = 0; j < currPartialWeights[i].length; j++)
            {
               for (int k = 0; k < currPartialWeights[i][j].length; k++)
               {
                  perceptron.weights[i][j][k] -= -lambda * currPartialWeights[i][j][k]; // Roll back the last change so that the weights are still optimized
               }
            }
         }
        currError = previousError;
	counter = MAX_STEPS;
      }
      else 
      {
         previousError = currError;
      }
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
