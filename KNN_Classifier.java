/**
 * This class creates a KNN classifier for Medical Data
 * Each Data-point contains 13 features and a single label
 * The Training File Path, testing file path and output file path can be specified in the global variables
 *
 * @author Anupam Gupta
 * No specific flags are needed to run this code.
 * Each java class has a main() function and in this case as well, we just need to run the main() function.
 */

import java.io.*;
import java.util.*;

public class KNN_Classifier {

    public static final String TESTING_FILE_PATH = <testing_file_path>;
    public static final String TRAINING_FILE_PATH = <training_file_path>;
    public static final String OUTPUT_FILE = <output_file_path>;
    public static final int K = 5;

    /**
     * Private Class to store individual data points
     */
    private static class Feature {
        private int age;
        private int sex;
        private int cp;
        private int trestbps;
        private int chol;
        private int fbs;
        private int restecg;
        private int thalach;
        private int exang;
        private double oldpeak;
        private int slope;
        private int ca;
        private int thal;

        public Feature(int age,
                       int sex,
                       int cp,
                       int trestbps,
                       int chol,
                       int fbs,
                       int restecg,
                       int thalach,
                       int exang,
                       double oldpeak,
                       int slope,
                       int ca,
                       int thal) {
            this.age = age;
            this.sex = sex;
            this.cp = cp;
            this.trestbps = trestbps;
            this.chol = chol;
            this.fbs = fbs;
            this.restecg = restecg;
            this.thalach = thalach;
            this.exang = exang;
            this.oldpeak = oldpeak;
            this.slope = slope;
            this.ca = ca;
            this.thal = thal;
        }

        /**
         * Function to return Euclidean distance between current feature and input feature
         * @param aux input feature
         * @return Euclidean distance between current feature and input feature
         */
        private double getDistance(Feature aux) {
            return Math.sqrt(
                    Math.pow(aux.age - this.age, 2.0) +
                    Math.pow(aux.sex - this.sex, 2.0) +
                    Math.pow(aux.cp - this.cp, 2.0) +
                    Math.pow(aux.trestbps - this.trestbps, 2.0) +
                    Math.pow(aux.chol - this.chol, 2.0) +
                    Math.pow(aux.fbs - this.fbs, 2.0) +
                    Math.pow(aux.restecg - this.restecg, 2.0) +
                    Math.pow(aux.thalach - this.thalach, 2.0) +
                    Math.pow(aux.exang - this.exang, 2.0) +
                    Math.pow(aux.oldpeak - this.oldpeak, 2.0) +
                    Math.pow(aux.slope - this.slope, 2.0) +
                    Math.pow(aux.ca - this.ca, 2.0) +
                    Math.pow(aux.thal - this.thal, 2.0)
                );
        }

        @Override
        public String toString() {
            return "age = " + this.age + ", " +
                    "sex = " + this.sex + ", " +
                    "cp = " + this.cp + ", " +
                    "trestbps = " + this.trestbps + ", " +
                    "chol = " + this.chol + ", " +
                    "fbs = " + this.fbs + ", " +
                    "restecg = " + this.restecg + ", " +
                    "thalach = " + this.thalach + ", " +
                    "exang = " + this.exang + ", " +
                    "oldpeak = " + this.oldpeak + ", " +
                    "slope = " + this.slope + ", " +
                    "ca = " + this.ca + ", " +
                    "thal = " + this.thal + ".";
        }
    }

    public static void main(String[] args) throws IOException {
        Map<Feature, Integer> featureToLabelMap = importData(TRAINING_FILE_PATH);
        List<Feature> testingFeatures = importTestData(TESTING_FILE_PATH);
        String result = "";
        for(int i = 0; i < testingFeatures.size(); i++) {
            List<Feature> closestNeighbors = findKClosestNeighbors(testingFeatures.get(i), featureToLabelMap.keySet(), K);
            int label = findMajorityTarget(closestNeighbors, featureToLabelMap);
            result += label + " ";
        }
        result = result.trim();
        BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE));
        writer.write(result);
        writer.close();
    }

    /**
     * Finds the Majority label from a set of closest datapoints entered as input.
     * @param closestFeatures List of Closest data points(features)
     * @param featureToLabelMap Map containing data-point (feature) to corresponding label mapping
     * @return Majority label from the list of closest closest data points (features). In case of ties, either of the labels
     * are selected with equal probability.
     */
    private static Integer findMajorityTarget(List<Feature> closestFeatures, Map<Feature, Integer> featureToLabelMap) {
        Map<Integer, Integer> countKeeper = new HashMap<Integer, Integer>();
        for(Feature feature: closestFeatures) {
            int target = featureToLabelMap.get(feature);
            if(!countKeeper.containsKey(target)) {
                countKeeper.put(target, 0);
            }
            countKeeper.put(target, countKeeper.get(target) + 1);
        }

        int majorityValue = 0;
        int max = 0;
        for(Map.Entry<Integer, Integer> entry: countKeeper.entrySet()) {
            if(entry.getValue() > max) {
                majorityValue = entry.getKey();
                max = entry.getValue();
            }
        }
        return majorityValue;
    }

    /**
     * Method to return the K closest neighbors to a given feature (data-point)
     * @param feature Input data-point
     * @param setOfFeatures Set of all available data-points (as features sub-class)
     * @param k value of K
     * @return returns the K closest unique neighbors to the input data-point as a List
     */
    private static List<Feature> findKClosestNeighbors(Feature feature, Set<Feature> setOfFeatures, int k) {
        List<Feature> closestFeatures = new ArrayList<Feature>();
        List<Double> distancesOfFeatures = new ArrayList<Double>();
        for(Feature individualFeature: setOfFeatures) {
            double distance = feature.getDistance(individualFeature);
            if(closestFeatures.size() < k) {
                closestFeatures.add(individualFeature);
                distancesOfFeatures.add(distance);
            } else {
                for (int i = 0; i < k; i++) {
                    if (distancesOfFeatures.get(i) > distance) {
                        distancesOfFeatures.remove(i);
                        closestFeatures.remove(i);
                        distancesOfFeatures.add(distance);
                        closestFeatures.add(individualFeature);
                        break;
                    }
                }
            }
        }
        return closestFeatures;
    }

    /**
     * Function to import Data from the input file-name
     * The function assumes the file is arranged in a csv format with 14 numbers in each line - 13 features and 1 data label
     * Each line is separated by a new line
     * The first line of the csv file is skipped since it is assumed to hold feature headers
     * @param fileName File-name to import data from
     * @return Map containing Mapping between individual features and their data label
     * @throws FileNotFoundException
     */
    private static Map<Feature, Integer> importData(String fileName) throws FileNotFoundException{
        boolean flag = false;
        Scanner sc = new Scanner(new File(fileName));
        sc.useDelimiter(",|\\n");
        Map<Feature, Integer> featureToLabelMap = new HashMap<Feature, Integer>();
        while(sc.hasNext()) {
            if(!flag) {
                sc.nextLine();
                flag = true;
            }
            Feature feature = extractFeature(sc);
            int label = Integer.parseInt(sc.next().trim());
            featureToLabelMap.put(feature, label);
        }
        return featureToLabelMap;
    }

    /**
     * Function to import test Data from the input file-name
     * The function assumes the file is arranged in a csv format with 13 numbers in each line - 13 features
     * Each line is separated by a new line
     * The first line of the csv file is skipped since it is assumed to hold feature headers
     * @param fileName File-name to import data from
     * @return Map containing Mapping between individual features and their data label
     *
     * @throws FileNotFoundException
     */
    private static List<Feature> importTestData(String fileName) throws FileNotFoundException {
        boolean flag = false;
        Scanner sc = new Scanner(new File(fileName));
        sc.useDelimiter(",|\\n");
        List<Feature> testFeatures = new ArrayList<Feature>();
        while (sc.hasNext()) {
            if (!flag) {
                sc.nextLine();
                flag = true;
            }
            testFeatures.add(extractFeature(sc));
        }
        return testFeatures;
    }

    /**
     * Helper function to take in a scanner object, read 13 data-points as specified in the input file
     * and return a Feature object
     * @param sc Scanner object attached to a specific file
     * @return A Feature object created with reading 13 data-points successively using the scanner object
     */
    private static Feature extractFeature(Scanner sc) {
        return new Feature(
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Double.parseDouble(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next()),
                Integer.parseInt(sc.next().trim())
        );
    }
}
