import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * @author Mihir Sharma
 * Created on Friday, October 4th (10.4.19).
 * <p>
 * The Trainer class is written to match the documentation as closely as possible.
 * <p>
 * The Trainer class uses a partial derivative-based method to minimize the error function for any number of input nodes and any number of output
 * nodes, making the assumption that the previously written Perceptron class is a functioning perceptron model.
 * The Trainer class also requires a correctly formatted input file, as specified in the README.md file.
 * <p>
 * Functions in this class include the following:
 * - main, which asks the user for an input file through the System.in and then runs the Trainer.
 * - readInputFile, which reads the input from the constructor's filename then populates the class-level variables.
 * - train, which begins the training and also provides diagnostic information after completion.
 * - step, which takes the individual steps during training. Matches documentation as closely as possible.
 */
public class Trainer
{
   Perceptron perceptron;
   String inputFile;
   static double MINIMUM_ERROR;
   static int MAX_STEPS;
   double[][] trialCases;
   int counter;                               // Counter checks the number of steps.
   double[][] truths;                         // A truth is an expected output.
   double lowValue;                           // Low threshold for randomized weights.
   double highValue;                          // High threshold for randomized weights.

   double lambda;                             // Training function
   double currError;

   int inputNodes;                            // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;                    // The number of nodes in each hidden activation layer.
   int outputNodes;                           // The number of nodes in the output activation layer.

   /**
    * Creates a new trainer for a perceptron, using the user-provided input filename.
    *
    * @param inputFile The list of input file names. Must be a full path.
    */
   public Trainer(String inputFile)
   {
      this.inputFile = inputFile;
      readInputFile();
      this.currError = Double.MAX_VALUE - 1.0;                                // Current error starts as larger as possible
      this.counter = 0;
      perceptron = new Perceptron(this.inputNodes, this.hiddenLayerNodes, this.outputNodes, inputFile);
      perceptron.randomizeWeights(lowValue, highValue);                       // Randomize weights before first use
   }

   /**
    * Main method for the Trainer reads the input files and then trains the network.
    * Uses System input to determine filename, with full file path. If inaccurate file path is provided, will throw FileNotFoundException and
    * terminate.
    *
    * @param args an array of String arguments for the main method
    */
   public static void main(String[] args)
   {
      Scanner sc = new Scanner(System.in);
      System.out.println("Please enter the desired input file, with the full file format.");
      String filename = sc.nextLine();
      Trainer trainer = new Trainer(filename);
      trainer.train();

   }

   /**
    * Method readInputFile reads all the input as specified in the README file and updates the class level variables.
    * Uses a try-catch to deal with I/O exceptions.
    */
   public void readInputFile()
   {
      BufferedReader bufferedReader;
      StringTokenizer stringTokenizer;

      try
      {
         bufferedReader = new BufferedReader(new FileReader(inputFile));                     // Opens previously specified input file.

         inputNodes = Integer.parseInt(bufferedReader.readLine());                           // Find number of input nodes

         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
         hiddenLayerNodes = new int[Integer.parseInt(stringTokenizer.nextToken())];
         for (int i = 0; i < hiddenLayerNodes.length; i++)
         {
            hiddenLayerNodes[i] = Integer.parseInt(stringTokenizer.nextToken());             // Populate number of nodes per hidden layer
         }

         outputNodes = Integer.parseInt(bufferedReader.readLine());                          // Find number of output nodes

         int numberCases = Integer.parseInt(bufferedReader.readLine());                      // Find the number of trial cases
         trialCases = new double[numberCases][inputNodes];
         truths = new double[numberCases][outputNodes];
         for (int i = 0; i < numberCases; i++)
         {
            stringTokenizer = new StringTokenizer(bufferedReader.readLine());
            trialCases[i] = new double[inputNodes];
            for (int j = 0; j < inputNodes; j++)
            {
               trialCases[i][j] = Double.parseDouble(stringTokenizer.nextToken());
            }
            truths[i] = new double[outputNodes];
            for (int j = 0; j < outputNodes; j++)
            {
               truths[i][j] = Double.parseDouble(stringTokenizer.nextToken());
            }
         }     // Iterating over the number of test cases
         lambda = Double.parseDouble(bufferedReader.readLine());                             // Lambda
         MINIMUM_ERROR = Double.parseDouble(bufferedReader.readLine());                      // Error
         MAX_STEPS = Integer.parseInt(bufferedReader.readLine());                            // Steps
         lowValue = Double.parseDouble(bufferedReader.readLine());                           // Low Value
         highValue = Double.parseDouble(bufferedReader.readLine());                          // High Value
         bufferedReader.close();
      }        // Reads the input file.
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " not accepted, terminating.");
      }

   }

   /**
    * Function train makes steps and modifies the perceptron's weights using gradient descent until one of three conditions are met:
    * 1. The error is less than a minimum error, or
    * 2. The training is capped by a predetermined counter.
    * Only randomizes the perceptron's weights the first time, after that it does not.
    * Also provides statistics at the end for readability after program terminates.
    */
   void train()
   {
      counter = 0;
      while ((currError > MINIMUM_ERROR && counter < MAX_STEPS))
      {
         step();
         counter++;
      }
      if (currError <= MINIMUM_ERROR)
      {
         System.out.println("Terminated because total error is less than the pre-determined threshold of " + MINIMUM_ERROR);
      }
      else if (counter >= MAX_STEPS)
      {
         System.out.println("\nTerminated because number of iterations exceeded the pre-determined threshold of " + MAX_STEPS);
      }
      System.out.println("Lambda: " + lambda);                                                        // Currently not adaptive.
      System.out.println("Minimum Error: " + MINIMUM_ERROR + "\nMax Number of Steps: " + MAX_STEPS);
      System.out.println("For random weights: Low Value " + lowValue + ", High Value " + highValue);
      System.out.println("Counter: " + counter);
      System.out.println("Error: " + currError);
      System.out.println("Final weights: " + Arrays.deepToString(perceptron.weights));

   }

   /**
    * Function step runs an individual step in the training process. A step is defined as finding the partials, using the error
    * to update the weights of the perceptron, setting weights/running the network, then deciding whether or not to use the
    * updated weights.
    */
   public void step()
   {
      double errors = 0.0;
      for (int tc = 0; tc < trialCases.length; tc++)
      {
         perceptron.expectedOutputs = truths[tc];
         perceptron.runNetwork(trialCases[tc]);                                              // Train current perceptron
         double[][][] currPartialWeights = perceptron.findPartials(perceptron.expectedOutputs);

         for (int i = 0; i < perceptron.numberActivationLayers - 1; i++)
         {
            for (int j = 0; j < perceptron.activations[i].length; j++)                       // Loop over Source array.
            {
               for (int k = 0; k < perceptron.activations[i + 1].length; k++)                // Loop over Destination array.
               {
                  perceptron.weights[i][j][k] += -lambda * currPartialWeights[i][j][k];      // Updates currWeights to new weights
               }
            }
         }

         // double[][][] lambdasChoice = adaptLambda();  FUTURE WORK ON ADAPTIVE LAMBDA, for now "you touch it, you break it."

         perceptron.runNetwork(trialCases[tc]);
         double[] trainedResult = perceptron.activations[perceptron.activations.length - 1]; // Find error after weight update
         perceptron.printResult();

         double newError = 0;                                                                // We could sum directly over errors,
                                                                                             // but two step for debug/printing
         for (int i = 0; i < perceptron.outputNodes; i++)
         {
            newError += 0.5 * perceptron.calculateError(truths[tc][i], trainedResult[i]);
         }
         errors += newError;
      } // For loop over the trial cases
      currError = errors;
   }


   /**
    * Function adaptLambda never gets used. In the future, it will be used for the adaptive lambda, but currently provides an added layer of
    * unnecessary problems to the code.
    */
   public void adaptLambda()
   {
      if(totalError < previousTotalError)
      {
         lambda *= 2.0;
      }
      else
      {
         lambda /= 2.0;
      }
   }

}
