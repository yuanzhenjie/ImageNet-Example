package imagenet;

import imagenet.Utils.DataMode;
import imagenet.Utils.ImageNetRecordReader;
import imagenet.Utils.PreProcessData;
import org.apache.hadoop.io.BytesWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.writable.Writable;
import org.canova.image.transform.FlipImageTransform;
import org.canova.image.transform.ImageTransform;
import org.canova.image.transform.WarpImageTransform;
import org.canova.spark.functions.data.RecordReaderBytesFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.spark.canova.CanovaDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.apache.hadoop.io.Text;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;


/**
 * Spark configuration to run ImageNet. The version argument from CNNImageNetMain sets whether it will run
 * SparkStandalone on just a local machine or SparkCluster on a cluster with master and workers.
 */
public class ImageNetSparkExample extends ImageNetMain {
    private static final Logger log = LoggerFactory.getLogger(ImageNetSparkExample.class);


    public void initialize() throws Exception{
        // Spark context
        JavaSparkContext sc = (version == "SparkStandAlone")? setupLocalSpark(): setupClusterSpark();

        // Load data and train
        String seqOutputPath = null;
        SparkDl4jMultiLayer sparkNetwork = null;
        JavaRDD<DataSet> trainData = null;
        ImageTransform flipTransform = new FlipImageTransform(new Random(42));
        ImageTransform warpTransform = new WarpImageTransform(new Random(42), 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});
        for(ImageTransform transform: transforms) {
            trainData = loadData(sc, trainPath, seqOutputPath, numTrainExamples, false, transform, DataMode.CLS_TRAIN);
            sparkNetwork = trainModel(new SparkDl4jMultiLayer(sc, model, new ParameterAveragingTrainingMaster(true, Runtime.getRuntime().availableProcessors(), 5, 1, 0)), trainData);
        }
        // Eval
        JavaRDD<DataSet> testData = loadData(sc, testPath, seqOutputPath, numTestExamples, false, null, DataMode.CLS_TEST);
        evaluatePerformance(sparkNetwork, testData);

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
//        conf.set(SparkDl4jMultiLayer.ACCUM_GRADIENT, String.valueOf(true));
        return new JavaSparkContext(conf);
    }


    private JavaSparkContext setupClusterSpark(){
        SparkConf conf = new SparkConf();
        conf.setAppName("ImageNet Cluster");
        return new JavaSparkContext(conf);
    }

    private JavaRDD<DataSet> loadData(JavaSparkContext sc, String inputPath, String seqOutputPath, int numExamples, boolean save, ImageTransform transform, DataMode dataMode) {
        System.out.println("Load data...");

        JavaPairRDD<Text, BytesWritable> filesAsBytes;
        JavaRDD<DataSet> data;

        if(inputPath==null && seqOutputPath != null){
            filesAsBytes = sc.sequenceFile(seqOutputPath, Text.class, BytesWritable.class);
        } else if(version == "SparkStandAlone"){
            PreProcessData pData = new PreProcessData(sc, save);
            pData.setupSequnceFile(inputPath, seqOutputPath);
            filesAsBytes = pData.getFile();
        } else {
            throw new IllegalArgumentException("Data can not be loaded running on a cluaster without an outputPath.");
        }

        RecordReaderBytesFunction recordReaderFunc = new RecordReaderBytesFunction(
                    new ImageNetRecordReader(HEIGHT, WIDTH, CHANNELS, null, transform, normalizeValue, dataMode));

        JavaRDD<Collection<Writable>> rdd = filesAsBytes.map(recordReaderFunc);
        // Load all files in path
        if(numExamples==-1)
            data = rdd.map(new CanovaDataSetFunction(-1, numLabels, false));
        else {
            // Limit number examples loaded
            JavaRDD<DataSet> dataRdd = rdd.map(new CanovaDataSetFunction(-1, numLabels, false));
            List<DataSet> listData = dataRdd.take(numExamples); // should have features and labels (1*1860) filled out
            data = sc.parallelize(listData);
        }

        data.cache();
        filesAsBytes.unpersist();
        return data;
    }

    private SparkDl4jMultiLayer trainModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data){
        System.out.println("Train model...");
        startTime = System.currentTimeMillis();
        model.fit(data);
        endTime = System.currentTimeMillis();
        trainTime = (int) (endTime - startTime) / 60000;
        return model;

    }

    private void evaluatePerformance(SparkDl4jMultiLayer model, JavaRDD<DataSet> testData) {
        System.out.println("Eval model...");
        startTime = System.currentTimeMillis();
        Evaluation evalActual = model.evaluate(testData);
        System.out.println(evalActual.stats());
        endTime = System.currentTimeMillis();
        testTime = (int) (endTime - startTime) / 60000;
    }


    public void cleanUp(JavaRDD<DataSet> data) {
        data.unpersist();
    }

}
