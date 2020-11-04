package tweeter;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class GenderClassification {
	
	//Method to clean text in the description and text columns
		private static String cleanText(Object obj) {
			if (obj == null){
				return "na";
			}
			else{
		        String cleanedText = obj.toString().toLowerCase()
		        		.replaceAll("http.*?\\s", " ")
		        		.replaceAll("[^a-zA-Z\\s+]", " ")
		        		.replaceAll("\\s\\W"," ")
		        		.replaceAll("\\W\\s"," ")
		        		.replaceAll("\\s+"," ");
		        return cleanedText;
		    }
	    }
		

	//entry point
	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		// Create spark session
		SparkSession sparkSession = SparkSession.builder()  
				.appName("GenderPrediction") 
				.master("local[*]") 
				.getOrCreate();
		
		// Data loading
		String pathTrain = "data/gender-classifier-DFE-791531.csv";	
		Dataset<Row> dataset = sparkSession.read()
				.option("header","true")
				.option("encoding", "latin1")
				.option("quote", "\"") //quoted columns
				.option("escape", "\"") //for quotes within quotes
				.option("mode", "DROPMALFORMED") // if erroneous record
				.csv(pathTrain)
				.select("gender", "description", "fav_number", "link_color", "sidebar_color", "text", "tweet_count");
		
		//filter only rows where gender column has valid values
		Dataset<Row> filteredData = dataset.filter("gender=\"male\" OR gender=\"female\" OR gender=\"brand\"");
		
		
		StructType customStructType1 = new StructType();
		customStructType1 = customStructType1.add("gender", DataTypes.StringType, true);
		customStructType1 = customStructType1.add("text", DataTypes.StringType, true);
		
		// cleaning data removing space, html links, #!@, fillers and making lowercase and merge all text data into one column
		Dataset<Row> cleanedData = filteredData.map(row->{
	          return RowFactory.create(row.get(0),cleanText(row.get(1)) + " "+ row.get(2).toString()+ " "+ row.get(3).toString() + " "+ cleanText(row.get(4)));
	            }, RowEncoder.apply(customStructType1));
	    
		// Data split for test and training
		Dataset<Row>[] splitData = cleanedData.randomSplit(new double[] {0.8,0.2}, 10L);
		Dataset<Row> trainData = splitData[0];
		Dataset<Row> testData = splitData[1];
			
		//change gender column into integers values (0,1,2)
		StringIndexerModel labelindexerModel = new StringIndexer()
				.setInputCol("gender")
				.setOutputCol("label").fit(trainData);
		
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("text")
				.setOutputCol("tweet_text");
		
		//remove English stopwords
		String[] stopWords = StopWordsRemover.loadDefaultStopWords("english");
		StopWordsRemover remover = new StopWordsRemover()
				.setStopWords(stopWords)
				.setInputCol(tokenizer.getOutputCol())
				.setOutputCol("filtered");		
		
		//CountVector
		CountVectorizer countvectorizer = new CountVectorizer()
				  .setInputCol(remover.getOutputCol())
				  .setOutputCol("features")
				  .setVocabSize(3000)
				  .setMinDF(10);
		
		//Models
		RandomForestClassifier rf = new RandomForestClassifier();

		DecisionTreeClassifier dt = new DecisionTreeClassifier()
				.setImpurity("entropy")
				.setMaxBins(32)
				.setMaxDepth(29);
		
		LogisticRegression lr = new LogisticRegression();

		//Convert indexes back to gender label {0,1,2 ->> brand, female, male}
		IndexToString labelConverter = new IndexToString()
				.setInputCol("prediction")
				.setOutputCol("predictedLabel").setLabels(labelindexerModel.labels());
		
		IndexToString inputlabelConverter = new IndexToString()
				.setInputCol("label")
				.setOutputCol("inputLabel").setLabels(labelindexerModel.labels());

		//Pipeline for the Random Forest
		Pipeline pipelineRF = new Pipeline()
				.setStages(new PipelineStage[] {labelindexerModel, tokenizer, remover, countvectorizer, rf,labelConverter, inputlabelConverter});	
		PipelineModel modelRF = pipelineRF.fit(trainData);		
		Dataset<Row> predictionsRF = modelRF.transform(testData);
		Dataset<Row> predictionsRFTrain = modelRF.transform(trainData);

		//Pipeline for the Decision Tree
		Pipeline pipelineDT = new Pipeline()
				.setStages(new PipelineStage[] {labelindexerModel, tokenizer, remover, countvectorizer, dt,labelConverter, inputlabelConverter});	
		PipelineModel modelDT = pipelineDT.fit(trainData);		
		Dataset<Row> predictionsDT = modelDT.transform(testData);
		Dataset<Row> predictionsDTTrain = modelDT.transform(trainData);
		
		//Pipeline for logistic regression
		Pipeline pipelineLR = new Pipeline()
				.setStages(new PipelineStage[] {labelindexerModel, tokenizer, remover, countvectorizer, lr ,labelConverter, inputlabelConverter});	
		PipelineModel modelLR = pipelineLR.fit(trainData);		
		Dataset<Row> predictionsLR = modelLR.transform(testData);
		Dataset<Row> predictionsLRTrain = modelLR.transform(trainData);
		
		// Evaluation 
		MulticlassClassificationEvaluator evaluatorPrecision = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("weightedPrecision");
		
		MulticlassClassificationEvaluator evaluatorRecall = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("weightedRecall");
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("accuracy");	

		//Evaluation Metrics
		double accuracyRF = evaluator.evaluate(predictionsRF);
		System.out.println("Accuracy for Random Forest = " + (accuracyRF));
		
		double accuracyRFTrain = evaluator.evaluate(predictionsRFTrain);
		System.out.println("Accuracy for Random Forest(TrainData) = " + (accuracyRFTrain));
		
		double recallRF = evaluatorRecall.evaluate(predictionsRF);
		System.out.println("Recall for Random Forest = " + (recallRF));
		
		double precisionRF = evaluatorPrecision.evaluate(predictionsRF);
		System.out.println("Precision for Random Forest = " + (precisionRF));
		
		System.out.println("Confusion Matrix for Random Forest ======");
		predictionsRF.groupBy("inputLabel","predictedLabel").count().show();
		
		double accuracyDT = evaluator.evaluate(predictionsDT);
		System.out.println("Accuracy for Decision Tree = " + (accuracyDT));
		
		double accuracyDTTrain = evaluator.evaluate(predictionsDTTrain);
		System.out.println("Accuracy for Decision Tree(TrainData) = " + (accuracyDTTrain));
		
		double recallDT = evaluatorRecall.evaluate(predictionsDT);
		System.out.println("Recall for Decision Tree = " + (recallDT));
		
		double precisionDT = evaluatorPrecision.evaluate(predictionsDT);
		System.out.println("Precision for Decision Tree = " + (precisionDT));
		
		System.out.println("Confusion Matrix for Decision Tree ======");
		predictionsDT.groupBy("inputLabel","predictedLabel").count().show();
		
		double accuracyLR = evaluator.evaluate(predictionsLR);
		System.out.println("Accuracy for Logistic Regression = " + (accuracyLR));
		
		double accuracyLRTrain = evaluator.evaluate(predictionsLRTrain);
		System.out.println("Accuracy for Logistic Regression(TrainData) = " + (accuracyLRTrain));
		
		double recallLR = evaluatorRecall.evaluate(predictionsLR);
		System.out.println("Recall for Logistic Regression = " + (recallLR));
		
		double precisionLR = evaluatorPrecision.evaluate(predictionsLR);
		System.out.println("Precision for Logistic Regression = " + (precisionLR));
		
		System.out.println("Confusion Matrix for Logistic Regression ======");
		predictionsLR.groupBy("inputLabel","predictedLabel").count().show();
	}

}
