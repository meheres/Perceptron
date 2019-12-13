package src;
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
 * The Trainer class uses back propagation to minimize the error function for any number of input nodes and any number of output
 * nodes, making the assumption that the previously written Perceptron class is a functioning perceptron model. The backpropagation
 * algorithm is generalized to handle any number of hidden activation layers, and any number of activations in each layer.
 * The Trainer class also requires correctly formatted input files, as specified in the README.md file.
 * 
 * Functions in this class include the following:
 * - main, which asks the user for an input file through the System.in and then runs the Trainer.
 * - readInputFile, which reads the input from the constructor's filename then populates the class-level variables.
 * - readInputActivations, which reads the inputs and then populates the input activatiions.
 * - readTruths, which reads the inputs and then populates the expected outputs.
 * - printOutputsToFile, which writes the final outputs to a file, each line in the output file representing the trained result for each case.
 * - train, which begins the training and also provides diagnostic information after completion.
 * - step, which takes the individual steps during training. Matches documentation as closely as possible.
 */
public class Trainer
{
   Perceptron perceptron;
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

   double lambda;                             // Training function
   double currError;

   int inputNodes;                            // The number of nodes in the input activation layer.
   int[] hiddenLayerNodes;                    // The number of nodes in each hidden activation layer.
   int outputNodes;                           // The number of nodes in the output activation layer.

   /**
    * Creates a new trainer for a perceptron, using the user-provided input filename. All files must follow the structure provided in the
    * README.md file.
    *
    * @param inputFile The input file name.
    * @param activationsFile The name of the file containing activations.
    * @param truthsFile The name of the file containing expected outputs.
    * @param outputsFile The name of the file to which the perceptron's final outputs will be printed.
    */
   public Trainer(String inputFile, String activationsFile, String truthsFile, String outputsFile)
   {
      this.inputFile = inputFile;
      this.activationsFile = activationsFile;
      this.truthsFile = truthsFile;
      this.outputsFile = outputsFile;
      readInputFile();
      this.currError = Double.MAX_VALUE - 1.0;                                // The current error begins as large as possible.
      this.counter = 0;
      perceptron = new Perceptron(this.inputNodes, this.hiddenLayerNodes, this.outputNodes);
      perceptron.randomizeWeights(lowValue, highValue);                       // Randomize the weights before the first use.
      readInputActivations();
      readTruths();
      perceptron.lambda = lambda;
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
      Trainer trainer = new Trainer(filename, activationsFilename, truthsFilename, outputFilename);
      trainer.train();
      long endTime = System.nanoTime();
      double time = (endTime - startTime)/1E6;
      System.out.println("Time: " + time);
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
      }  // Reads the input file.
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
            for (int j = 0; j < perceptron.inputNodes; j++)
            {

               trialCases[i][j] = Double.parseDouble(split[j]);
            }
         }
         br.close();
      } // Reads the input file.
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

            for (int j = 0; j < perceptron.outputNodes; j++)
            {
               truths[i][j] = Double.parseDouble(split[j]);
            }
         }
         br.close();
      } // Reads the input file.
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
         for (int j = 0; j < numberCases; j++)
         {
            perceptron.runNetwork(trialCases[j]);
            for (int i = 0; i < perceptron.outputNodes; i++)
            {
               pw.write(perceptron.activations[perceptron.activations.length - 1][i] + " "); // Writes all values in last layer.
            }
            pw.println();
         }
         pw.close();
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Output File " + e.toString() + " not accepted, terminating.");
      }
   }

   /**
    * Function train makes steps and modifies the perceptron's weights using gradient descent until one of two conditions are met:
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

      if (currError <= MINIMUM_ERROR) // The first end condition.
      {
         System.out.println("Terminated because total error is less than the pre-determined threshold of " + MINIMUM_ERROR);
      }

      else if (counter >= MAX_STEPS) // The second end condition.
      {
         System.out.println("\nTerminated because number of iterations exceeded the pre-determined threshold of " + MAX_STEPS);
      }

      System.out.print("Perceptron config: " + inputNodes + "-");
      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         System.out.print(hiddenLayerNodes[i] + "-");
      }
      System.out.print(outputNodes + "\n");

      System.out.println("Lambda: " + lambda);                                                        // Currently not adaptive.
      System.out.println("Minimum Error: " + MINIMUM_ERROR + "\nMax Number of Steps: " + MAX_STEPS);  // Print a bunch of debug info.
      System.out.println("For random weights: Low Value " + lowValue + ", High Value " + highValue);
      System.out.println("Number of iterations: " + counter);
      System.out.println("Error: " + currError);
      // System.out.println("Final weights: " + Arrays.deepToString(perceptron.weights));             // Print final weights for debugging purposes.
      printOutputsToFile();                                                                           // Writes final outputs to file.


   }


   /**
    * Function step runs an individual step in the training process. A step is defined as defining the expected outputs of the current case, then
    * running the network using the current trial case, then modifies the weights of the perceptron using back propagation.
    */
   public void step()
   {
      double errors = 0.0;
      for (int tc = 0; tc < trialCases.length; tc++)
      {
         perceptron.expectedOutputs = truths[tc];
         perceptron.runNetwork(trialCases[tc]);
         perceptron.backProp(truths[tc]);

         double[] trainedResult = perceptron.activations[perceptron.activations.length - 1]; // Find error after weight updates

         double newError = 0;
         for (int i = 0; i < perceptron.outputNodes; i++)
         {
            newError += 0.5 * perceptron.calculateError(truths[tc][i], trainedResult[i]);    // We could sum directly over errors, but this is better
         }                                                                                   // for debugging.
         errors += newError;
      } // for (int tc = 0; tc < trialCases.length; tc++)
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
