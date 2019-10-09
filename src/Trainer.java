public class Trainer
{
   Perceptron perceptron;
   String[] trialInputFiles;
   static double MINIMUM_ERROR;

   double lambda;
   double currError;
   double previousError;
   double[][][] oldWeights;
   double[][][] currWeights;


   /**
    * Creates a new trainer for a perceptron, with a list of input filenames.
    *
    * @param trialInputFiles The list of input file names. Must be a full path.
    */
   public Trainer(String[] trialInputFiles)
   {
      this.trialInputFiles = trialInputFiles;
      this.currError = 1.0;                                                            // Total and Previous Total errors start
      this.previousError = 0.0;                                                    // as 0.0.
      this.lambda = 0.5;

      perceptron = new Perceptron(2, new int[]{2}, 1, trialInputFiles[0]);

      this.oldWeights = new double[perceptron.weights.length]            // Create the currentWeights, copying the
            [perceptron.weights[0].length]         // values from the perceptron's set of
            [perceptron.weights[0][0].length];     // weights the first time.

      for (int i = 0; i < perceptron.weights.length; i++)
      {
         for (int j = 0; j < perceptron.weights[0].length; j++)
         {
            for (int k = 0; k < perceptron.weights[0][0].length; k++)
            {
               this.oldWeights[i][j][k] = 1.0 * (Math.random());
            }
         }
      }

      this.currWeights = new double[perceptron.weights.length]
            [perceptron.weights[0].length]
            [perceptron.weights[0][0].length];
      for (int i = 0; i < perceptron.weights.length; i++)
      {
         for (int j = 0; j < perceptron.weights[0].length; j++)
         {
            System.arraycopy(this.oldWeights[i][j], 0, this.currWeights[i][j], 0, perceptron.weights[0][0].length);
         }
      }
      System.out.println("debug");           //TODO: REMOVE
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

  /*public void determineInputs(String inputFile)
   {
      BufferedReader bufferedReader;
      StringTokenizer stringTokenizer;

      try
      {
         bufferedReader = new BufferedReader(new FileReader(inputFileName));
         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " not accepted, terminating.");
      }

      // Iterate over the inputs in the first line then populate the input activation array
      for (int i = 0; i < inputs.length; i++)
      {
         inputs[i] = Double.parseDouble(stringTokenizer.nextToken());
      }

      try
      {
         // Advance the Buffered Reader by one line to begin reading the weights
         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " incorrectly shows weight, terminating.");
      }

      // Iterate over the 3D array and populate it with the weights as written in the weights file
      for (int i = 0; i < weights.length; i++)
      {
         for (int j = 0; j < weights[i].length; j++)
         {
            for (int k = 0; k < weights[i][j].length; k++)
            {
               weights[i][j][k] = Double.parseDouble(stringTokenizer.nextToken());
            }
         }
      }

      try
      {
         // Advance the Buffered Reader by one line to begin reading the expected outputs
         stringTokenizer = new StringTokenizer(bufferedReader.readLine());
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException("Input " + e.toString() + " incorrectly shows expected output, terminating.");
      }
      for (int i = 0; i < outputNodes; i++)
      {
         expectedOutputs[i] = Double.parseDouble(stringTokenizer.nextToken());
      }
   }*/

   /**
    * Main method for the Trainer reads the input files and then trains the network.
    *
    * @param args an array of String arguments for the main method
    */
   public static void main(String[] args)
   {
      Trainer trainer = new Trainer(new String[]{
            "/Users/mihir/IdeaProjects/Neural Networks/Java XOR Implementation/src/inputs/inputFile00.txt",
            "/Users/mihir/IdeaProjects/Neural Networks/Java XOR Implementation/src/inputs/inputFile01.txt",
            "/Users/mihir/IdeaProjects/Neural Networks/Java XOR Implementation/src/inputs/inputFile10.txt",
            "/Users/mihir/IdeaProjects/Neural Networks/Java XOR Implementation/src/inputs/inputFile11.txt"});

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
      int counter = 0;
      while (currError > 0.01 && currError != previousError && counter < 100)
      {
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
      double[][][] totalPartialWeights = this.getTotalPartialWeights();
      copyWeights(currWeights, oldWeights);
      for (int i = 0; i < currWeights.length; i++)
      {
         for (int j = 0; j < currWeights[0].length; j++)
         {
            for (int k = 0; k < currWeights[0][0].length; k++)
            {
               currWeights[i][j][k] += (-lambda * totalPartialWeights[i][j][k]); // Updates currWeights to new weights
            }
         }
      }

      perceptron.setWeights(currWeights);
      for (int d = 0; d < trialInputFiles.length; d++)
      {
         double[] inputTestCase;
         if (d == 0) inputTestCase = new double[]{1.0, 1.0};
         else if (d == 1) inputTestCase = new double[]{0.0, 1.0};
         else if (d == 2) inputTestCase = new double[]{1.0, 0.0};
         else inputTestCase = new double[]{0.0, 0.0};
         perceptron.runNetwork(inputTestCase);
      }

      double newError = findTotalError();
      if (newError > currError)
      {
         perceptron.setWeights(oldWeights);
      }
   }

   public double[][][] getTotalPartialWeights()
   {
      double[][][] totalPartialWeights = new double[currWeights.length][currWeights[0].length][currWeights[0][0].length];

      // using magic numbers
      for (int i = 0; i < 4; i++)
      {
         double[] inputTestCase;
         if (i == 0) inputTestCase = new double[]{1.0, 1.0};
         else if (i == 1) inputTestCase = new double[]{0.0, 1.0};
         else if (i == 2) inputTestCase = new double[]{1.0, 0.0};
         else inputTestCase = new double[]{0.0, 0.0};
         perceptron.runNetwork(inputTestCase);
         double[][][] partialsOfNetworkWeights = perceptron.findPartials();

         for (int j = 0; j < partialsOfNetworkWeights.length; j++)
         {
            for (int k = 0; k < partialsOfNetworkWeights[0].length; k++)
            {
               for (int l = 0; l < partialsOfNetworkWeights[0][0].length; l++)
               {
                  totalPartialWeights[j][k][l] += partialsOfNetworkWeights[j][k][l];
               }
            }
         }
      }

      /*
      for (String inputFile : inputFiles)
      {

      }

      How you should be doing it
       */
      return totalPartialWeights;
   }


   /**
    * Finds the total error of the function and updates the previous error.
    */
   public double findTotalError()
   {
      for (int d = 0; d < trialInputFiles.length; d++)
      {
         double[] inputTestCase;
         if (d == 0) inputTestCase = new double[]{1.0, 1.0};
         else if (d == 1) inputTestCase = new double[]{0.0, 1.0};
         else if (d == 2) inputTestCase = new double[]{1.0, 0.0};
         else inputTestCase = new double[]{0.0, 0.0};
         perceptron.runNetwork(inputTestCase);
      }
      double error = 0.0;
      for (int i = 0; i < perceptron.outputNodes; i++)
      {
         error += 0.5 * ((perceptron.expectedOutputs[i] - perceptron.activations[i][perceptron.maxNumberNodes - 1]) *
               (perceptron.expectedOutputs[i] - perceptron.activations[i][perceptron.maxNumberNodes - 1]));
         System.out.println("perceptron expected output: " + perceptron.expectedOutputs[i]);
         System.out.println("perceptron actual output: " + perceptron.activations[i][perceptron.maxNumberNodes - 1]);
      }
      System.out.println("Error: " + error);                   // TODO: REMOVE DEBUG STATEMENT
      return Math.sqrt(error);
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

   public void updateWeights()
   {
      for (int i = 0; i < currWeights.length; i++)
      {
         for (int j = 0; j < currWeights[0].length; j++)
         {
            for (int k = 0; k < currWeights[0][0].length; k++)
            {
               currWeights[i][j][k] = oldWeights[i][j][k];
            }
         }
      }
   }
}
