package imagenet;

import imagenet.Utils.ImageNetLoader;
import imagenet.sampleModels.AlexNet;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.canova.spark.functions.RecordReaderFunction;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.canova.CanovaDataSetFunction;
import org.deeplearning4j.spark.canova.CanovaSequenceDataSetFunction;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;


/**
 * Created by nyghtowl on 11/24/15.
 */
public class CNNImageNetSparkExample {
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetSparkExample.class);

    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    private String modelType = "AlexNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    private int batchSize = 10;
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

        ClassPathResource trainPath = new ClassPathResource("train");
        String path = FilenameUtils.concat(trainPath.getFile().getAbsolutePath(), "*");

        System.out.println("Build model...");
        MultiLayerNetwork net = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
        net.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));

        // Spark context
        SparkConf conf = new SparkConf().setMaster("local");
//        conf.set("spark.executor.memory", "1024m");
        conf.setAppName("imageNet");
        conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        final JavaSparkContext sc = new JavaSparkContext(conf);

        SparkDl4jMultiLayer sparkModel = new SparkDl4jMultiLayer(sc, net);

        JavaPairRDD<String,PortableDataStream> sparkDataTrain = sc.binaryFiles(path);
        List<Tuple2<String, PortableDataStream>> t = sparkDataTrain.collect();


        System.out.println("Train model...");
        //Train network
        String labelPath = FilenameUtils.concat(ImageNetLoader.BASE_DIR, ImageNetLoader.LABEL_FILENAME);
        RecordReaderFunction rrf = new RecordReaderFunction(new ImageNetRecordReader(numRows, numColumns, nChannels, labelPath, true, Pattern.quote("_")));
        JavaRDD<Collection<Writable>> rdd = sparkDataTrain.map(rrf);
//        List<Collection<Writable>>  listSpark = rdd.take(2); // TODO check data

        JavaRDD<DataSet> data = rdd.map(new CanovaDataSetFunction(numRows * numColumns * nChannels, numCategories, true));
//        List<DataSet>  listSpark = data.collect(); // TODO check data

//        net = sparkModel.fitDataSet(data); // TODO hangs and doesn't run correctly

        System.out.println("Eval model...");
//        MultiLayerNetwork netCopy = sparkModel.getNetwork().clone();
// TODO check eval with below
//        Evaluation evalExpected = new Evaluation();
//        INDArray outLocal = netCopy.output( input??? , Layer.TrainingMode.TEST);
//        evalExpected.eval(labels???, outLocal);

        // TODO fix ImageNet iterator to load this data
        ClassPathResource testPath = new ClassPathResource("test");
        path = FilenameUtils.concat(testPath.getFile().getAbsolutePath(), "*");

        List<DataSet> test = new ArrayList<>(78);

        JavaRDD<DataSet> testDS = sc.parallelize(test);
        Evaluation evalActual = sparkModel.evaluate(testDS,batchSize);
        log.info(evalActual.stats());

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
