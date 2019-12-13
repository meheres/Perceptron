package src;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
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
public class BTrainer
{
   BPerceptron perceptron;
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
    * Creates a new trainer for a perceptron, using the user-provided input filename.
    *
    * @param inputFile The input file name.
    * @param activationsFile The name of the file containing activations.
    * @param truthsFile The name of the file containing expected outputs.
    * @param outputsFile The name of the file to which the perceptron's final outputs will be printed.
    */
   public BTrainer(String inputFile, String activationsFile, String truthsFile, String outputsFile)
   {
      this.inputFile = inputFile;
      this.activationsFile = activationsFile;
      this.truthsFile = truthsFile;
      this.outputsFile = outputsFile;
      readInputFile();
      this.currError = Double.MAX_VALUE - 1.0;                                // Current error starts as larger as possible
      this.counter = 0;
      perceptron = new BPerceptron(this.inputNodes, this.hiddenLayerNodes, this.outputNodes, inputFile);
      perceptron.randomizeWeights(lowValue, highValue);                       // Randomize weights before first use
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
      BTrainer trainer = new BTrainer(filename, activationsFilename, truthsFilename, outputFilename);
      trainer.train();
      long endTime = System.nanoTime();
      double time = (endTime - startTime)/1E6;
      System.out.println("Time: " + time);
      System.out.println("BTrainer");
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
            for (int j = 0; j < perceptron.inputNodes; j++)
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

            for (int j = 0; j < perceptron.outputNodes; j++)
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
         for (int i = 0; i < perceptron.outputNodes; i++)
         {
            pw.write(perceptron.activations[perceptron.activations.length - 1][i] + " "); // Writes all values in last layer.
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
      // Final weights: [[[-4.316252797058683, 3.9804910380455656, 4.104050501517506, -4.191499997560866, 3.663871622334993, 3.1052220770515966, 4.051284158628942, 8.075292536163476], [-7.798094954971835, 7.288882201832737, 7.390729163402655, 9.968988659509892, 6.852845789775467, 6.239198162614919, 7.30517388688126, -3.398760972493725]], [[-13.83970725785841, -11.805754487378614, -11.688158236061561], [1.006083243497097, -3.085485550073578, 3.238713071059456], [1.4614660144399974, -3.1685798225778816, 3.6300233116033302], [-1.2544889554869232, 8.563587112223395, -9.111047572445525], [1.0701066356328952, -2.196883191787846, 2.4493795260933253], [1.0276455223012981, -1.8245332161018673, 0.9685618012592963], [1.7041194176439167, -2.9300346611998775, 3.4791822557815917], [0.08243710245304843, 8.56093460780411, -8.829818031084338]]]
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
      System.out.println("Number of iterations: " + counter);
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
         perceptron.expectedOutputs = truths[tc];
         perceptron.runNetwork(trialCases[tc]);                                              // Train current perceptron
         perceptron.backProp(truths[tc]);

         double[] trainedResult = perceptron.activations[perceptron.activations.length - 1]; // Find error after weight updates

         perceptron.printResult();                                                        // Uncomment for small numbers of output nodes
         double newError = 0;                                                                // We could sum directly over errors,
         for (int i = 0; i < perceptron.outputNodes; i++)                                    // but two step for debug/printing
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
