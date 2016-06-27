package imagenet;

import imagenet.Utils.ImageNetLoader;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.AlexNet;
import org.deeplearning4j.LeNet;
import org.deeplearning4j.VGGNetA;
import org.deeplearning4j.VGGNetD;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.util.NetSaverLoaderUtils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.lang.management.MemoryUsage;
import java.util.*;

/**
 * ImageNet is a large scale visual recognition challenge run by Stanford and Princeton. The competition covers
 * standard object classification as well as identifying object location in the image.
 *
 * This file is the main class that is called when running the program. Pass in arguments to help
 * adjust how and where the program will run. Note ImageNet is typically structured with 224 x 224
 * pixel size images but this can be adjusted by change WIDTH & HEIGHT.
 *
 * References: ImageNet
 * Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
 * Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.
 * (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
 *

 * Created by nyghtowl on 1/12/16.
 */
public class CNNImageNetMain {
    private static final Logger log = LoggerFactory.getLogger(CNNImageNetMain.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--version",usage="Version to run (Standard, SparkStandAlone, SparkCluster)",aliases = "-v")
    protected String version = "Standard";
    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    protected String modelType = "AlexNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 40;
    @Option(name="--testBatchSize",usage="Test Batch size",aliases="-tB")
    protected int testBatchSize = batchSize;
    @Option(name="--numBatches",usage="Number of batches",aliases="-nB")
    protected int numBatches = 5;
    @Option(name="--numTestBatches",usage="Number of test batches",aliases="-nTB")
    protected int numTestBatches = numBatches;
    @Option(name="--numEpochs",usage="Number of epochs",aliases="-nE")
    protected int numEpochs = 5;
    @Option(name="--iterations",usage="Number of iterations",aliases="-i")
    protected int iterations = 1;
    @Option(name="--numCategories",usage="Number of categories",aliases="-nC")
    protected int numCategories = 4;
    @Option(name="--trainFolder",usage="Train folder",aliases="-taF")
    protected String trainFolder = "train";
    @Option(name="--testFolder",usage="Test folder",aliases="-teF")
    protected String testFolder = "test";
    @Option(name="--saveModel",usage="Save model",aliases="-sM")
    protected boolean saveModel = false;
    @Option(name="--saveParams",usage="Save parameters",aliases="-sP")
    protected boolean saveParams = false;

    @Option(name="--confName",usage="Model configuration file name",aliases="-conf")
    protected String confName = null;
    @Option(name="--paramName",usage="Parameter file name",aliases="-param")
    protected String paramName = null;

    protected long startTime = 0;
    protected long endTime = 0;
    protected int trainTime = 0;
    protected int testTime = 0;

    protected static final int HEIGHT = 224;
    protected static final int WIDTH = 224;
    protected static final int CHANNELS = 3;
    protected static final int outputNum = 1860;
    protected int seed = 42;
    protected int listenerFreq = 1;
    protected int totalTrainNumExamples = batchSize * numBatches;
    protected int totalTestNumExamples = testBatchSize * numTestBatches;

    // Paths for data
    protected String basePath = ImageNetLoader.BASE_DIR;
    protected String trainPath = FilenameUtils.concat(basePath, trainFolder);
    protected String testPath = FilenameUtils.concat(basePath, testFolder);

//        String trainPath = FilenameUtils.concat(new ClassPathResource("train").getFile().getAbsolutePath(), "*");
//        String testPath = FilenameUtils.concat(new ClassPathResource("test").getFile().getAbsolutePath(), "*");
//    protected String trainPath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/train/*");
//    protected String testPath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/" + testFolder + "/*");

    protected String labelPath = FilenameUtils.concat(basePath, ImageNetLoader.LABEL_FILENAME);
    protected String valLabelMap = FilenameUtils.concat(basePath, ImageNetLoader.VAL_MAP_FILENAME);
    protected String outputPath = NetSaverLoaderUtils.defineOutputDir(modelType.toString());
    protected String confPath = this.toString() + "conf.yaml";
    protected String paramPath = this.toString() + "param.bin";
    protected Map<String, String> paramPaths = new HashMap<>();
    protected String[] layerNames; // Names of layers to store parameters
    protected String rootParamPath;

    protected MultiLayerNetwork model = null;

    public void run(String[] args) throws Exception {
        Nd4j.dtype = DataBuffer.Type.FLOAT;
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        switch (version) {
            case "Standard":
                new CNNImageNetExample().initialize();
                break;
            case "SparkStandAlone":
                new CNNImageNetSparkExample().initialize();
                break;
            case "SparkCluster":
                new CNNImageNetSparkExample().initialize();
                break;
            default:
                break;
        }
        System.out.println("****************Example finished********************");
    }

    protected void buildModel() {
        System.out.println("Build model....");
        if (confName != null && paramName != null) {
            String confPath = FilenameUtils.concat(outputPath, confName + "conf.yaml");
            String paramPath = FilenameUtils.concat(outputPath, paramName + "param.bin");
            model = NetSaverLoaderUtils.loadNetworkAndParameters(confPath, paramPath);
        } else {
            switch (modelType) {
                case "LeNet":
                    model = new LeNet(HEIGHT, WIDTH, CHANNELS, outputNum, seed, iterations).init();
                    break;
                case "AlexNet":
                    model = new AlexNet(HEIGHT, WIDTH, CHANNELS, outputNum, seed, iterations).init();
                    break;
                case "VGGNetA":
                    model = new VGGNetA(HEIGHT, WIDTH, CHANNELS, outputNum, seed, iterations).init();
                    break;
                case "VGGNetD":
                    model = new VGGNetD(HEIGHT, WIDTH, CHANNELS, outputNum, seed, iterations, rootParamPath).init();
                    break;
                default:
                    break;
            }
        }
    }

    protected void setListeners(){
        // Listeners
        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(1).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();

        model.setListeners(new ScoreIterationListener(listenerFreq)); // not needed for spark?
//        model.setListeners(new HistogramIterationListener(1));
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), paramListener));

    }

    protected void saveAndPrintResults(){
        System.out.println("****************************************************");
        System.out.println("Total training runtime: " + trainTime + " minutes");
        System.out.println("Total evaluation runtime: " + testTime + " minutes");
        System.out.println("****************************************************");
        if (saveModel) NetSaverLoaderUtils.saveNetworkAndParameters(model, outputPath.toString());
        if (saveParams) NetSaverLoaderUtils.saveParameters(model, layerNames, paramPaths);


    }

    public static void main(String[] args) throws Exception {
        new CNNImageNetExample().run(args);
    }


}
