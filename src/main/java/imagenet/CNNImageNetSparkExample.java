package imagenet;

import imagenet.Utils.PreProcessDataSpark;
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
 * Spark configuration to run ImageNet. The version argument from CNNImageNetMain sets whether it will run
 * SparkStandalone on just a local machine or SparkCluster on a cluster with master and workers.
 */
public class CNNImageNetSparkExample extends CNNImageNetMain{
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetSparkExample.class);


    public void initialize() throws Exception{
        // Spark context
        JavaSparkContext sc = (version == "SparkStandAlone")? setupLocalSpark(): setupClusterSpark();

        // Load data
        String seqOutputPath = null;
        JavaRDD<DataSet> trainData = loadData(sc, trainPath, seqOutputPath, totalTrainNumExamples, false);
        JavaRDD<DataSet> testData = loadData(sc, testPath, seqOutputPath, totalTestNumExamples, false);

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
        cleanUp(trainData);
        cleanUp(testData);
    }

    private JavaSparkContext setupLocalSpark(){
        SparkConf conf = new SparkConf()
                .setMaster("local[*]");
        conf.setAppName("ImageNet Local");
//        conf.set("spak.executor.memory", "4g");
//        conf.set("spak.driver.memory", "4g");
//        conf.set("spark.driver.maxResultSize", "1g");
        conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
//        conf.set(SparkDl4jMultiLayer.ACCUM_GRADIENT, String.valueOf(true));
        return new JavaSparkContext(conf);
    }


    private JavaSparkContext setupClusterSpark(){
        SparkConf conf = new SparkConf();
        conf.setAppName("ImageNet Cluster");
        return new JavaSparkContext(conf);
    }

    private JavaRDD<DataSet> loadData(JavaSparkContext sc, String inputPath, String seqOutputPath, int numberExamples, boolean save) {
        System.out.println("Load data...");

        JavaPairRDD<Text, BytesWritable> filesAsBytes;
        JavaRDD<DataSet> data;
        String regexSplit = Pattern.quote("_");
        boolean appendLabel = true;

        if(inputPath==null && seqOutputPath != null){
            filesAsBytes = sc.sequenceFile(seqOutputPath, Text.class,BytesWritable.class);
        } else if(version == "SparkStandAlone"){
            filesAsBytes = new PreProcessDataSpark(sc, inputPath, seqOutputPath, save).getFile();
        } else {
            throw new IllegalArgumentException("Data can not be loaded running on a cluaster without an outputPath.");
        }

        RecordReaderBytesFunction recordReaderFunc = new RecordReaderBytesFunction(
                new ImageNetRecordReader(HEIGHT, WIDTH, CHANNELS, labelPath, appendLabel, regexSplit));
        JavaRDD<Collection<Writable>> rdd = filesAsBytes.map(recordReaderFunc);

        // Load all files in path
        if(numberExamples==-1)
            data = rdd.map(new CanovaDataSetFunction(-1, outputNum, false));
        else {
            // Limit number examples loaded
            JavaRDD<DataSet> dataRdd = rdd.map(new CanovaDataSetFunction(-1, outputNum, false));
            List<DataSet> listData = dataRdd.take(numberExamples); // should have features and labels (1*1860) filled out
            data = sc.parallelize(listData);
        }

        data.cache();
        filesAsBytes.unpersist();
        return data;
    }

    private MultiLayerNetwork trainModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data){
        System.out.println("Train model...");
        startTime = System.currentTimeMillis();
        model.fitDataSet(data, batchSize, totalTrainNumExamples, numBatches);
        endTime = System.currentTimeMillis();
        trainTime = (int) (endTime - startTime) / 60000;
        return model.getNetwork().clone();

    }

    private void evaluatePerformance(SparkDl4jMultiLayer model, JavaRDD<DataSet> testData) {
        System.out.println("Eval model...");
        startTime = System.currentTimeMillis();
        Evaluation evalActual = model.evaluate(testData, labels);
        System.out.println(evalActual.stats());
        endTime = System.currentTimeMillis();
        testTime = (int) (endTime - startTime) / 60000;
    }


    public void cleanUp(JavaRDD<DataSet> data) {
        data.unpersist();
    }

}
