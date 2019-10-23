package org.neuralnet;

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
   static final int MAX_STEPS = 100;
   double[][] testCases;
   int counter;                               // Counter checks the number of steps and provides the current test case (counter %
                                              // testCases.length)

   double[][] truths;                         // A truth is an expected output.

   double lambda;                             // Training function
   double currError;

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
      this.lambda = 1.5;
      this.counter = 0;
      perceptron = new Perceptron(this.inputNodes, this.hiddenLayerNodes, this.outputNodes, inputFile);
      perceptron.randomizeWeights();                                          // Randomize weights before first use
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
      perceptron.setWeights(testWeights);
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
         testCases = new double[numberCases][inputNodes];
         truths = new double[numberCases][outputNodes];
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
    * Main method for the Trainer reads the input files and then trains the network.
    *
    * @param args an array of String arguments for the main method
    */
   public static void main(String[] args)
   {
      Trainer trainer = new Trainer("/Users/ssamant/personal/school/src/org/neuralnet/inputsFile.txt");
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
      counter = 0; // TODO: REMOVE
      while ((currError > MINIMUM_ERROR && counter < MAX_STEPS))
      {
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
       double errors = 0.0;
       for (int tc = 0; tc < testCases.length; tc++) {
           perceptron.expectedOutputs = truths[tc];
           perceptron.runNetwork(testCases[tc]); // Train current perceptron
           /*System.out.print("Pre-updated result: ");
           perceptron.printResult();*/
           double trainedResult = perceptron.activations[perceptron.activations.length - 1][0];
           double[][][] currPartialWeights = perceptron.findPartials(perceptron.expectedOutputs[0]);


           //currPartialWeights[0][0][0] = 0.037396443;
           //currPartialWeights[0][1][0] = 0.04866787;
           for (int i = 0; i < perceptron.numberActivationLayers - 1; i++) // SS: SHould this be -1?
           {
               for (int j = 0; j < perceptron.activations[i].length; j++) {
                   for (int k = 0; k < perceptron.activations[i + 1].length; k++) // SS: Is this right?
                   {
                       perceptron.weights[i][j][k] += -lambda * currPartialWeights[i][j][k]; // Updates currWeights to new weights
                   }
               }
           }

           // double[][][] lambdasChoice = adaptLambda();  TODO: FUTURE WORK ON ADAPTIVE LAMBDA
           perceptron.runNetwork(testCases[tc]);
           System.out.print("Post-updated result: ");
           perceptron.printResult();
           System.out.println("Partials: " + Arrays.deepToString(currPartialWeights)); //TODO: REMOVE DEBUG STATEMENT
           System.out.println("Weights: " + Arrays.deepToString(perceptron.weights)); //TODO: REMOVE DEBUG STATEMENT

           double newError = 0.5 * perceptron.calculateError(truths[tc][0], trainedResult);
           System.out.println("New Error: " + newError);

           errors += newError;

       }

       System.out.println("Current Error: " + currError);
       System.out.println("Total Error: " + errors + "\n"); //TODO: REMOVE DEBUG STATEMENT

      currError = errors;
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
