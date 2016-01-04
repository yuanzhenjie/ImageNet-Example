package imagenet;

import imagenet.sampleModels.AlexNet;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;
import java.util.regex.Pattern;


/**
 * Created by nyghtowl on 11/24/15.
 */
public class CNNImageNetSparkExample {
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetSparkExample.class);

    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    private String modelType = "AlexNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    private int batchSize = 1;
    @Option(name="--numBatches",usage="Number of batches",aliases="-nB")
    private int numBatches = 1;
    @Option(name="--numTestBatches",usage="Number of test batches",aliases="-nTB")
    private int numTestBatches = 1;
    @Option(name="--numEpochs",usage="Number of epochs",aliases="-nE")
    private int numEpochs = 1;
    @Option(name="--iterations",usage="Number of iterations",aliases="-i")
    private int iterations = 1;
    @Option(name="--numCategories",usage="Number of categories",aliases="-nC")
    private int numCategories = 1;
    @Option(name="--trainFolder",usage="Train folder",aliases="-taF")
    private String trainFolder = "train";
    @Option(name="--testFolder",usage="Test folder",aliases="-teF")
    private String testFolder = "val/val-sample";
    @Option(name="--saveParams",usage="Save parameters",aliases="-sP")
    private boolean saveParams = true;


    public void doMain(String[] args) throws Exception{
        String basePath = System.getProperty("user.home") + File.separator + "Documents" + File.separator + "skymind" + File.separator + "imagenet" + File.separator;
        String trainData = basePath + trainFolder + File.separator;
        String testData = basePath + testFolder + File.separator;
        String labelPath = basePath + "cls-loc-labels.txt";
        String confPath = this.toString() + "conf.yaml";
        String paramPath = this.toString() + "param.bin";

        final int numRows = 224;
        final int numColumns = 224;
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

        System.out.println("Load data...");
        // TODO finish applying how to load data - especially limitSplit

        //load the images from the bucket setting the size to 28 x 28
        int totalTrainNumExamples = batchSize * numBatches;

        String[] allForms = {"jpg", "jpeg", "JPG", "JPEG"};
        RecordReader recordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, labelPath, true);
        recordReader.initialize(new LimitFileSplit(new File(trainData), allForms, totalTrainNumExamples, numCategories, Pattern.quote("_"), 0, new Random(123)));
//        JavaRDD<LabeledPoint> data = MLLibUtil.fromDataSet();

        System.out.println("Build model..."
        );
        MultiLayerConfiguration netConf = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).conf();

        // Spark context
        SparkConf conf = new SparkConf().setMaster("local");
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        conf.set("spark.executor.memory", "1024m");
        conf.setAppName("imageNet");
        conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        final JavaSparkContext sc = new JavaSparkContext(conf);

        String s = recordReader.next().toString();
        sc.textFile(s);

        SparkDl4jMultiLayer sparkModel = new SparkDl4jMultiLayer(sc, netConf);

        System.out.println("Train model...");
        //Train network
//        MultiLayerNetwork net = sparkModel.fit(sc, recordReader, batchSize/iterations);

        System.out.println("Eval model...");
        //TODO

        System.out.println("Save model...");
//        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(paramPath));
//        Nd4j.write(model.params(), bos);
//        bos.flush();
//        bos.close();

//        Nd4j.write(network.params(), new DataOutputStream(new FileOutputStream(paramPath)));
//        FileUtils.write(new File(confPath), model.conf().toYaml()); // Yaml or Json?
//        log.info("Saved configuration and parameters to: {}, {}", confPath, paramPath);



    }

    public static void main(String[] args) throws Exception {
        new CNNImageNetSparkExample().doMain(args);
    }

}
