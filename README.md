# K_Nearest_Neighbors
This project showcases a Java Implementation of K Nearest Neighbors algorithm on Medical Data aimed at determining the possibility of heart disease in an individual based on previously seen data. The code, is currently hardcoded to use the value of K as 5 and takes in a training data file, a test data file (both in csv) and an output file as inputs, which can be specified as hardcoded variables in the code. This specific implementation is built on a data set that contains 13 data markers (attributes) to describe the health status of an individual: 

- age - Integer
- sex - Integer
- cp(Chest pain) - Integer
- trestbps (resting bps) - Integer
- chol(cholestorol) - Integer
- fbs(fasting blood sugar) - Integer
- restecg(rest ecgabnormality) - Integer
- thalach(thalium  stress test maximum heart rate) - Integer
- exang(exercise  induced  angina) - Integer
- oldpeak(ecg  STpeak) - Double
- slope(ecg ST slope)- Integer
- ca(colored fluoroscopy) - Integer

The first line of the csv is ignored since it's considered to contain the labels for each column. An example data entry is: 

```
  63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
```

The last column determines if the person has a heart disease or not in the training data. It is represented as a binary variable and is absent in the test data. 

The distance measure used for this project is the Eucalidean distance between points. The file data is output to the file path presented in the output file path variable and is separated by spaces and is encoded in the same binary scheme utilized by the training data set to denote heart patients. 

----
