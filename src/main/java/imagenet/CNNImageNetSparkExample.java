package imagenet;

import imagenet.Utils.ImageNetLoader;
import imagenet.Utils.ModelUtils;
import imagenet.sampleModels.AlexNet;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.io.BytesWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.canova.spark.functions.data.RecordReaderBytesFunction;
import org.canova.spark.functions.pairdata.BytesPairWritable;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.canova.spark.functions.RecordReaderFunction;
import org.deeplearning4j.spark.canova.CanovaDataSetFunction;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
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
public class CNNImageNetSparkExample {
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetSparkExample.class);

    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    private String modelType = "LeNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    private int batchSize = 2;
    @Option(name="--numBatches",usage="Number of batches",aliases="-nB")
    private int numBatches = 1;
    @Option(name="--numTestBatches",usage="Number of test batches",aliases="-nTB")
    private int numTestBatches = 1;
    @Option(name="--numEpochs",usage="Number of epochs",aliases="-nE")
    private int numEpochs = 1;
    @Option(name="--iterations",usage="Number of iterations",aliases="-i")
    private int iterations = 1;
    @Option(name="--numCategories",usage="Number of categories",aliases="-nC")
    private int numCategories = 2;
    @Option(name="--trainFolder",usage="Train folder",aliases="-taF")
    private String trainFolder = "train";
    @Option(name="--testFolder",usage="Test folder",aliases="-teF")
    private String testFolder = "val/val-sample";
    @Option(name="--saveModel",usage="Save model",aliases="-sM")
    private boolean saveModel = true;

    public void doMain(String[] args) throws Exception{
        String outputPath = ModelUtils.defineOutputDir(modelType.toString());
        String confPath = this.toString() + "conf.yaml";
        String paramPath = this.toString() + "param.bin";
        String labelPath = FilenameUtils.concat(ImageNetLoader.BASE_DIR, ImageNetLoader.LABEL_FILENAME);
        String trainPath = FilenameUtils.concat(new ClassPathResource("train").getFile().getAbsolutePath(), "*");
        String testPath = FilenameUtils.concat(new ClassPathResource("test").getFile().getAbsolutePath(), "*");

        final int numRows = 100;
        final int numColumns = 100;
        int nChannels = 3;
        int outputNum = 1860;
        int seed = 123;

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);

        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        // Spark context
        SparkConf conf = new SparkConf()
                .setMaster("local");
//                .setMaster("local[6]");
        conf.setAppName("imageNet");
//        conf.set("spar.executor.memory", "2g");
//        conf.set("spark.driver.maxResultSize", "4g");
        conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        final JavaSparkContext sc = new JavaSparkContext(conf);

        // TODO setup pre process to group by # pics, temp save and reload
        System.out.println("Load data...");
        // train data
//        JavaPairRDD<String,PortableDataStream> sparkDataTrain = sc.binaryFiles(trainPath);
//        RecordReaderFunction trainRrf = new RecordReaderFunction(new ImageNetRecordReader(numRows, numColumns, nChannels, labelPath, true, Pattern.quote("_")));
//        JavaRDD<Collection<Writable>> trainRdd = sparkDataTrain.map(trainRrf);
//        JavaRDD<DataSet> trainData = trainRdd.map(new CanovaDataSetFunction(-1, outputNum, false));

        JavaPairRDD<String,PortableDataStream> sparkDataTrain = sc.binaryFiles(trainPath);
        JavaPairRDD<Text, BytesWritable> filesTrainAsBytes = sparkDataTrain.mapToPair(new FilesAsBytesFunction());
        RecordReaderBytesFunction trainRrf = new RecordReaderBytesFunction(new ImageNetRecordReader(numRows, numColumns, nChannels, labelPath, true, Pattern.quote("_")));
        JavaRDD<Collection<Writable>> trainRdd = filesTrainAsBytes.map(trainRrf);
        JavaRDD<DataSet> trainData = trainRdd.map(new CanovaDataSetFunction(-1, outputNum, false));


        // TODO check data
//        List<Tuple2<String, PortableDataStream>> listPortable = sparkDataTrain.collect();
//        List<Collection<Writable>> listRDD = rdd.take(2);
//        List<DataSet>  listData = data.collect(); // should have features and labels (1*1860) filled out

        // test data
        // TODO fix ImageNet iterator to load this data
        JavaPairRDD<String,PortableDataStream> sparkDataTest = sc.binaryFiles(testPath);
        JavaPairRDD<Text, BytesWritable> filesTestAsBytes = sparkDataTest.mapToPair(new FilesAsBytesFunction());
        RecordReaderBytesFunction testRrf = new RecordReaderBytesFunction(new ImageNetRecordReader(numRows, numColumns, nChannels, labelPath, true, Pattern.quote("_")));
        JavaRDD<Collection<Writable>> testRdd = filesTestAsBytes.map(testRrf);
        JavaRDD<DataSet> testData = testRdd.map(new CanovaDataSetFunction(-1, outputNum, false));

        trainData.cache();
        testData.cache();

        System.out.println("Build model...");
        MultiLayerNetwork model = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
//        net.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1))); // doesn't seem needed

        System.out.println("Train model...");
        SparkDl4jMultiLayer sparkModel = new SparkDl4jMultiLayer(sc, model);
        model = sparkModel.fitDataSet(trainData); // TODO hangs and doesn't run correctly

        System.out.println("Eval model...");
        MultiLayerNetwork netCopy = sparkModel.getNetwork().clone();
        Evaluation evalActual = sparkModel.evaluate(testData,batchSize);
        log.info(evalActual.stats());

        System.out.println("Save model and parameters...");
        if (saveModel) ModelUtils.saveModelAndParameters(model, outputPath.toString());

    }

    public static void main(String[] args) throws Exception {
        new CNNImageNetSparkExample().doMain(args);
    }

}
