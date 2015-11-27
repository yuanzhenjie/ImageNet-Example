package imagenet;

import imagenet.sampleModels.AlexNet;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;


/**
 * Created by nyghtowl on 11/24/15.
 */
public class CNNImageNetSparkExample {
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetSparkExample.class);

    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    private String modelType = "AlexNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    private int batchSize = 8;
    @Option(name="--numBatches",usage="Number of batches",aliases="-nB")
    private int numBatches = 1;
    @Option(name="--numTestBatches",usage="Number of test batches",aliases="-nTB")
    private int numTestBatches = 1;
    @Option(name="--numEpochs",usage="Number of epochs",aliases="-nE")
    private int numEpochs = 1;
    @Option(name="--iterations",usage="Number of iterations",aliases="-i")
    private int iterations = 1;
    @Option(name="--numCategories",usage="Number of categories",aliases="-nC")
    private int numCategories = 4;
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
        String labelPath = basePath + "cls-loc-labels.csv";
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

        if (args == null || args.length != 3) {
            String err = "Invalid input. Expect 3 arg passed. Usage: args[0] = input path, args[1] = output folder, " +
                    "args[2] = summary stats output file (local)";
            throw new RuntimeException(err);
        }

        for (int i = 0; i < args.length; i++) {
            System.out.println("args[" + i + "] = \"" + args[i] + "\"");
        }

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        conf.set("spark.executor.memory", "528m");
        conf.setAppName("imageNet");
        final JavaSparkContext sc = new JavaSparkContext(conf);


        System.out.println("Load data...");
        // TODO finish applying how to load data - especially limitSplit

        //load the images from the bucket setting the size to 28 x 28
        final String s3Bucket = "file:///home/ec2-user..."; // from S3
        JavaRDD<LabeledPoint> data = MLLibUtil.fromBinary(sc.binaryFiles(s3Bucket + "/*")
                , new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath));

        //OR
//        JavaRDD<String> input = sc.textFile(args[0]); // from command line
//        StandardScaler scaler = new StandardScaler();
//        final StandardScalerModel scalarModel = scaler.fit(data.map(new Function<LabeledPoint, Vector>() {
//            @Override
//            public Vector call(LabeledPoint v1) throws Exception {
//                return v1.features();
//            }
//        }).rdd());
//        //get the trained data for the train/test split
//        JavaRDD<LabeledPoint> normalizedData = data.map(new Function<LabeledPoint, LabeledPoint>() {
//            @Override
//            public LabeledPoint call(LabeledPoint v1) throws Exception {
//                Vector features = v1.features();
//                Vector normalized = scalarModel.transform(features);
//                return new LabeledPoint(v1.label(), normalized);
//            }
//        }).cache()
//        JavaRDD<LabeledPoint>[] trainTestSplit = normalizedData.randomSplit(new double[]{80, 20});

        System.out.println("Build model...");
        MultiLayerConfiguration netConf = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).conf();
        SparkDl4jMultiLayer model = new SparkDl4jMultiLayer(sc.sc(),netConf);

        System.out.println("Train model...");
//        MultiLayerNetwork trainedNetwork = model.fit(trainTestSplit[0],100);
        model.fit(sc,data);

        System.out.println("Eval model...");
        //TODO

        System.out.println("Save model...");
//        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(paramPath));
//        Nd4j.write(model.params(), bos);
//        bos.flush();
//        bos.close();
// TODO Setup spark instance to output params publicly - lombok
//        Nd4j.write(model.params(), new DataOutputStream(new FileOutputStream(paramPath)));
//        FileUtils.write(new File(confPath), model.conf().toYaml()); // Yaml or Json?
//        log.info("Saved configuration and parameters to: {}, {}", confPath, paramPath);



    }

    public static void main(String[] args) throws Exception {
        new CNNImageNetSparkExample().doMain(args);
    }

}
