package src;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Class BmpToActivations takes in an input file of Bitmap activations as specified by DibDump.txt and then converts it to the activations of a
 * Perceptron.
 *
 */
public class BmpToActivations
{
   static String inputName;  // Input comes from the bitmap (from DibDump.txt)
   static String outputName; // Output should be formatted to go into a perceptron.
   static String readingName;// Reading file is the inputsFile.txt

   public BmpToActivations(String inputName, String outputName, String readingName)
   {
      this.inputName   = inputName;
      this.outputName  = outputName;
      this.readingName = readingName;
   }

   /**
    *
    * @param args Must be formatted as follows:
    *             args[0] inputFileName
    *             args[1] outputFileName
    *             args[2] readingFileName
    */
   public static void main(String[] args) throws IOException
   {
      String inputFileName, outputFileName, readingFileName;

      if(args.length > 0) inputFileName   = args[0];
      else inputFileName                  = "trialCases.txt";

      if(args.length > 1) outputFileName  = args[1];
      else outputFileName                 = "truths.txt";

      if(args.length > 2) readingFileName = args[2];
      else readingFileName                = "inputsFile.txt";

      BmpToActivations writer = new BmpToActivations(inputFileName, outputFileName, readingFileName);

      BufferedReader inputReader = new BufferedReader(new FileReader(inputFileName));
      PrintWriter outputWriter   = new PrintWriter(outputFileName);
      BufferedReader preReader   = new BufferedReader(new FileReader(readingFileName));

      for (int i = 0; i < 3; i++)
      {
         outputWriter.println(preReader.readLine()); // Write inputs, numhidden, and outputs
      }

      int numTrials = Integer.parseInt(preReader.readLine());
      outputWriter.println(numTrials);

      String input = inputReader.readLine();

      for (int i = 0; i < numTrials; i++)
      {
         outputWriter.println(input + input);
      }

      for (int i = 0; i < 3; i++)
      {
         outputWriter.println(preReader.readLine());
      }
      
      inputReader.close();
      outputWriter.close();
      preReader.close();
   }
}
