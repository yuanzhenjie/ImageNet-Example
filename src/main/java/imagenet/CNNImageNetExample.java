package imagenet;


import imagenet.Utils.ImageNetDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

import static org.junit.Assert.assertTrue;

/**
 *
 * Created by nyghtowl on 9/24/15.
 */
public class CNNImageNetExample extends CNNImageNetMain{

    private static final Logger log = LoggerFactory.getLogger(CNNImageNetExample.class);

    public CNNImageNetExample() {
    }

    public void initialize() throws Exception {
        boolean gradientCheck = false;

        Map<String, String> paramPaths = null;

        // Load data
        MultipleEpochsIterator trainIter = loadData(batchSize, totalTrainNumExamples, "CLS_TRAIN");
        MultipleEpochsIterator testIter = loadData(testBatchSize, totalTestNumExamples, "CLS_VAL");

        // Build
        buildModel();
        setListeners();

        // Gradient Check
        if (gradientCheck) gradientCheck(trainIter, model);

        // Train
        trainModel(trainIter);

        // Evaluation
        evaluatePerformance(testIter);

        // Save
        saveAndPrintResults();

    }

    private MultipleEpochsIterator loadData(int batchSize, int totalNumExamples, String mode){
        log.info("Load data....");
        //// asyncIter = new AsyncDataSetIterator(dataIter, 1); TODO doesn't have next(num)

        // TODO incorporate some formate of below code when using full validation set to pass valLabelMap through iterator
//                RecordReader testRecordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath, valLabelMap); // use when pulling from main val for all labels
//                testRecordReader.initialize(new LimitFileSplit(new File(testData), allForms, totalNumExamples, numCategories, Pattern.quote("_"), 0, new Random(123)));

        //TODO need dataIter that loops through set number of examples like SamplingIter but takes iter vs dataset
        return new MultipleEpochsIterator(numEpochs,
                new ImageNetDataSetIterator(batchSize, totalNumExamples,
                        new int[] {HEIGHT, WIDTH, CHANNELS}, numCategories, outputNum, mode));
    }

    private void trainModel(MultipleEpochsIterator data){
        log.info("Train model....");
        startTime = System.currentTimeMillis();
        model.fit(data);
        endTime = System.currentTimeMillis();
        trainTime = (int) (endTime - startTime) / 60000;
    }

    private void evaluatePerformance(MultipleEpochsIterator iter){
        log.info("Evaluate model....");
        DataSet imgNet;
        INDArray output;

        Evaluation eval = new Evaluation(labels);
        startTime = System.currentTimeMillis();
        //TODO setup iterator to randomize and pass in iterator vs doing a loop here
        for(int i=0; i < numTestBatches; i++){
            imgNet = iter.next(testBatchSize);
            output = model.output(imgNet.getFeatureMatrix());
            eval.eval(imgNet.getLabels(), output);
        }
        endTime = System.currentTimeMillis();
        log.info(eval.stats());
        log.info("****************************************************");
        testTime = (int) (endTime - startTime) / 60000;

    }

    private void gradientCheck(DataSetIterator dataIter, MultiLayerNetwork model){
        DataSet imgNet;
        log.info("Gradient Check....");

        imgNet = dataIter.next();
        String name = new Object() {
        }.getClass().getEnclosingMethod().getName();

        model.setInput(imgNet.getFeatures());
        model.setLabels(imgNet.getLabels());
        model.computeGradientAndScore();
        double scoreBefore = model.score();
        for (int j = 0; j < 1; j++)
            model.fit(imgNet);
        model.computeGradientAndScore();
        double scoreAfter = model.score();
//            String msg = name + " - score did not (sufficiently) decrease during learning (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
//            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
        for (int j = 0; j < model.getnLayers(); j++)
            System.out.println("Layer " + j + " # params: " + model.getLayer(j).numParams());

        double default_eps = 1e-6;
        double default_max_rel_error = 0.25;
        boolean print_results = true;
        boolean return_on_first_failure = false;
        boolean useUpdater = true;

        boolean gradOK = GradientCheckUtil.checkGradients(model, default_eps, default_max_rel_error,
                print_results, return_on_first_failure, imgNet.getFeatures(), imgNet.getLabels(), useUpdater);

        assertTrue(gradOK);

    }



}
