package imagenet;

import org.apache.hadoop.io.BytesWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.canova.spark.functions.data.RecordReaderBytesFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.CanovaDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.apache.hadoop.io.Text;

import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;


/**
 * Created by nyghtowl on 11/24/15.
 */
public class CNNImageNetSparkExample extends CNNImageNetMain{
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetSparkExample.class);


    public void initialize() throws Exception{
        // Spark context
        JavaSparkContext sc = setupSpark();

        // Load data
        JavaRDD<DataSet> trainData = loadData(sc, trainPath, totalTrainNumExamples);
        JavaRDD<DataSet> testData = loadData(sc, testPath, totalTestNumExamples);

        // Build
        buildModel();
        setListeners();

        // Train
        SparkDl4jMultiLayer sparkModelCopy = new SparkDl4jMultiLayer(sc,
                trainModel(new SparkDl4jMultiLayer(sc, model), trainData));

        // Eval
        evaluatePerformance(sparkModelCopy, testData);

        // Save
        saveAndPrintResults();

        // Close
//        cleanUp(trainData);
//        cleanUp(testData);
    }

    private JavaSparkContext setupSpark(){
        SparkConf conf = new SparkConf()
                .setMaster("local[*]");
        conf.setAppName("imageNet");
//        conf.set("spak.executor.memory", "4g");
//        conf.set("spak.driver.memory", "4g");
//        conf.set("spark.driver.maxResultSize", "1g");
//        conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        return new JavaSparkContext(conf);
    }

    private JavaRDD<DataSet> loadData(JavaSparkContext sc, String path, int totalNumExamples) {
        String regexSplit = Pattern.quote("_");
        boolean appendLabel = true;
        // TODO setup pre process to group by # pics, temp save and reload
        System.out.println("Load data...");
        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(path);
        JavaPairRDD<Text, BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        RecordReader recordReader = new ImageNetRecordReader(HEIGHT, WIDTH, CHANNELS, labelPath, appendLabel, regexSplit);
        RecordReaderBytesFunction recordReaderFunc = new RecordReaderBytesFunction(recordReader);
        JavaRDD<Collection<Writable>> rdd = filesAsBytes.map(recordReaderFunc);
//        JavaRDD<DataSet> data = rdd.map(new CanovaDataSetFunction(-1, outputNum, false));

        JavaRDD<DataSet> dataRdd = rdd.map(new CanovaDataSetFunction(-1, outputNum, false));
        List<DataSet> listData = dataRdd.take(batchSize * numBatches); // should have features and labels (1*1860) filled out
        JavaRDD<DataSet> data = sc.parallelize(listData);

        // TODO check data
//        List<Tuple2<String, PortableDataStream>> listPortable = sparkData.collect();
//        List<Collection<Writable>> listRDD = rdd.collect();

        data.cache();
        return data;
    }

    private MultiLayerNetwork trainModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data){
        System.out.println("Train model...");
        startTime = System.currentTimeMillis();
        model.fitDataSet(data);
        endTime = System.currentTimeMillis();
        trainTime = (int) (endTime - startTime) / 60000;
        return model.getNetwork().clone();

    }

    private void evaluatePerformance(SparkDl4jMultiLayer model, JavaRDD<DataSet> testData) {
        System.out.println("Eval model...");
        startTime = System.currentTimeMillis();
        Evaluation evalActual = model.evaluate(testData, testBatchSize, labels, false);
        System.out.println(evalActual.stats());
        endTime = System.currentTimeMillis();
        testTime = (int) (endTime - startTime) / 60000;
    }


    public void cleanUp(JavaRDD<DataSet> data) {
        data.unpersist();
    }

}
