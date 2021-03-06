package src.old;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.StringTokenizer;

/**
 * @author Mihir Sharma
 * Created on Friday, October 4th (10.4.19).
 * 
 * The Trainer class is written to match the documentation as closely as possible.
 * 
 * The Trainer class uses a partial derivative-based method to minimize the error function for any number of input nodes and any number of output
 * nodes, making the assumption that the previously written Perceptron class is a functioning perceptron model.
 * The Trainer class also requires a correctly formatted input file, as specified in the README.md file.
 * 
 * Functions in this class include the following:
 * - main, which asks the user for an input file through the System.in and then runs the Trainer.
 * - readInputFile, which reads the input from the constructor's filename then populates the class-level variables.
 * - readInputActivations, which reads the inputs and then populates the input activatiions.
 * - readTruths, which reads the inputs and then populates the expected outputs.
 * - writeOutputsToFile, 
 * - train, which begins the training and also provides diagnostic information after completion.
 * - step, which takes the individual steps during training. Matches documentation as closely as possible.
 */
class TrainerOld
{
   PerceptronOld perceptronOld;
   String inputFile;
   String activationsFile;
   String truthsFile;
   String outputsFile;
   static double MINIMUM_ERROR;
   static int MAX_STEPS;
   double[][] trialCases;
   int counter;                               // Counter checks the number of steps.
   double[][] truths;                         // A truth is an expected output.
   double lowValue;                           // Low threshold for randomized weights.
   double highValue;                          // High threshold for randomized weights.
   int numberCases;                           // Number of trial cases
   double[][][] currPartialWeights;

   double lambda;                             // Training function
   double currError;

   int inputNodes;                            // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;                    // The number of nodes in each hidden activation layer.
   int outputNodes;                           // The number of nodes in the output activation layer.

   /**
    * Creates a new trainer for a perceptron, using the user-provided input filename.
    *
    * @param inputFile The input file name.
    * @param activationsFile The name of the file containing activations.
    * @param truthsFile The name of the file containing expected outputs.
    * @param outputsFile The name of the file to which the perceptron's final outputs will be printed.
    */
   public TrainerOld(String inputFile, String activationsFile, String truthsFile, String outputsFile)
   {
      this.inputFile = inputFile;
      this.activationsFile = activationsFile;
      this.truthsFile = truthsFile;
      this.outputsFile = outputsFile;
      readInputFile();
      this.currError = Double.MAX_VALUE - 1.0;                                // Current error starts as larger as possible
      this.counter = 0;
      perceptronOld = new PerceptronOld(this.inputNodes, this.hiddenLayerNodes, this.outputNodes, inputFile);
      perceptronOld.randomizeWeights(lowValue, highValue);                       // Randomize weights before first use
      readInputActivations();
      readTruths();
      double[][][] currPartialWeights = new double[perceptronOld.numberActivationLayers - 1][][];
      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         if (i == 0)
         {
            currPartialWeights[i] = new double[inputNodes][hiddenLayerNodes[i]];
         }
         else
         {
            currPartialWeights[i] = new double[hiddenLayerNodes[i - 1]][hiddenLayerNodes[i]];
         }
      }
      currPartialWeights[hiddenLayerNodes.length] = new double[hiddenLayerNodes[hiddenLayerNodes.length - 1]][outputNodes];
   }

   /**
    * Main method for the Trainer reads the input files and then trains the network.
    * Uses System input to determine filename, with full file path. If inaccurate file path is provided, will throw FileNotFoundException and
    * terminate.
    *
    * @param args an array of String arguments for the main method.
    *             The first argument must be the name of the input file.
    *             The second argument must be the name of the activations file.
    *             The third argument must be the name of the truths file.
    *             The fourth argument must be the name of the outputs file.
    */
   public static void main(String[] args)
   {
      long startTime = System.nanoTime();
      String filename = args[0];
      String activationsFilename = args[1];
      String truthsFilename = args[2];
      String outputFilename = args[3];
      TrainerOld trainer = new TrainerOld(filename, activationsFilename, truthsFilename, outputFilename);
      trainer.train();
      long endTime = System.nanoTime();
      double time = (endTime - startTime)/1E6;
      System.out.println(time);
      System.out.println("Trainer");
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

         numberCases = Integer.parseInt(bufferedReader.readLine());                          // Find the number of trial cases
         trialCases = new double[numberCases][inputNodes];
         truths = new double[numberCases][outputNodes];
         lambda = Double.parseDouble(bufferedReader.readLine());                             // Lambda
         MINIMUM_ERROR = Double.parseDouble(bufferedReader.readLine());                      // Error
         MAX_STEPS = Integer.parseInt(bufferedReader.readLine());                            // Steps
         lowValue = Double.parseDouble(bufferedReader.readLine());                           // Low Value
         highValue = Double.parseDouble(bufferedReader.readLine());                          // High Value
         bufferedReader.close();
      }        // Reads the input file.
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input File " + e.toString() + " not accepted, terminating.");
      }

   }

   /**
    * Method readInputActivations uses the input file specified to read and load the input activations as specified.
    * Uses a try-catch to handle I/O exceptions.
    */
   public void readInputActivations()
   {
      BufferedReader br;
      StringTokenizer s;
      try
      {
         br = new BufferedReader(new FileReader(activationsFile));
         for (int i = 0; i < numberCases; i++)
         {
            String[] split = br.readLine().split(" ");
            for (int j = 0; j < perceptronOld.inputNodes; j++)
            {
               trialCases[i][j] = Double.parseDouble(split[j]);
            }
         }


         br.close();
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Activations File " + e.toString() + " not accepted, terminating.");
      }
   }

   /**
    * Method readTruths() uses the input file specified to read and load the expected outputs, or truths, as specified.
    * Uses a try-catch to handle I/O exceptions.
    */
   public void readTruths()
   {
      BufferedReader br;
      try
      {
         br = new BufferedReader(new FileReader(truthsFile));

         for (int i = 0; i < numberCases; i++)
         {
            String[] split = br.readLine().split(" ");

            for (int j = 0; j < perceptronOld.outputNodes; j++)
            {
               truths[i][j] = Double.parseDouble(split[j]);
            }
         }
         br.close();
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Truths File " + e.toString() + " not accepted, terminating.");
      }
   }

   /**
    * Method printOutputsToFile takes the perceptron's current outputs and writes them to a specified file.
    * Uses a try-catch to handle I/O exceptions.
    */
   public void printOutputsToFile()
   {
      PrintWriter pw;
      try
      {
         pw = new PrintWriter(outputsFile);
         for (int i = 0; i < perceptronOld.outputNodes; i++)
         {
            pw.write(perceptronOld.activations[perceptronOld.activations.length - 1][i] + " "); // Writes all values in last layer.
         }
         pw.close();
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Output File " + e.toString() + " not accepted, terminating.");
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
      System.out.println("Minimum Error: " + MINIMUM_ERROR + "\nMax Number of Steps: " + MAX_STEPS);  // Print a bunch of debug info.
      System.out.println("For random weights: Low Value " + lowValue + ", High Value " + highValue);
      System.out.println("Counter: " + counter);
      System.out.println("Error: " + currError);                                                      
      // System.out.println("Final weights: " + Arrays.deepToString(perceptron.weights));
      printOutputsToFile();                                                                           // Writes final outputs to file.


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
         perceptronOld.expectedOutputs = truths[tc];
         perceptronOld.runNetwork(trialCases[tc]);                                              // Train current perceptron
         currPartialWeights = perceptronOld.findPartials(perceptronOld.expectedOutputs);
         for (int i = 0; i < perceptronOld.numberActivationLayers - 1; i++)
         {
            for (int j = 0; j < perceptronOld.activations[i].length; j++)                       // Loop over Source array.
            {
               for (int k = 0; k < perceptronOld.activations[i + 1].length; k++)                // Loop over Destination array.
               {
                  perceptronOld.weights[i][j][k] += -lambda * currPartialWeights[i][j][k];      // Updates currWeights to new weights
               }
            }
         }

         // double[][][] lambdasChoice = adaptLambda();  FUTURE WORK ON ADAPTIVE LAMBDA, for now "you touch it, you break it."

         perceptronOld.runNetwork(trialCases[tc]);
         double[] trainedResult = perceptronOld.activations[perceptronOld.activations.length - 1]; // Find error after weight update
         // perceptron.printResult();                                                        // For small numbers of output nodes
         double newError = 0;                                                                // We could sum directly over errors,
         // but two step for debug/printing
         for (int i = 0; i < perceptronOld.outputNodes; i++)
         {
            newError += 0.5 * perceptronOld.calculateError(truths[tc][i], trainedResult[i]);
         }
         errors += newError;
      } // For loop over the trial cases
      currError = errors;
   }


   /**
    * Function adaptLambda never gets used. In the future, it will be used for the adaptive lambda, but currently provides an added layer of
    * unnecessary problems to the code.
    *
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
    }*/

}
