package imagenet.Utils;

import org.canova.api.io.labels.PathLabelGenerator;
import org.canova.image.transform.ImageTransform;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import java.io.File;
import java.util.Random;


/**
 * DL4J DataSetIterator specific to this project.
 */

public class ImageNetDataSetIterator extends RecordReaderDataSetIterator {

    /** Loads images with given  batchSize, numExamples returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples) {
        this(batchSize, numExamples, new int[] {ImageNetLoader.HEIGHT, ImageNetLoader.WIDTH, ImageNetLoader.CHANNELS }, ImageNetLoader.NUM_CLS_LABELS,ImageNetLoader.LABEL_PATTERN,  DataModeEnum.CLS_TRAIN, 1, null, 255, new Random(System.currentTimeMillis()), null);
    }

    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, dataMode returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, DataModeEnum dataModeEnum) {
        this(batchSize, numExamples, imgDim, numLabels, ImageNetLoader.LABEL_PATTERN, dataModeEnum, 1, null, 0, new Random(System.currentTimeMillis()), null);
    }

    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, dataMode, train, splitTrainTest, imageTransform, normalizeValue, rng returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, DataModeEnum dataModeEnum, double splitTrainTest, ImageTransform imageTransform, int normalizeValue, Random rng) {
        this(batchSize, numExamples, imgDim, numLabels, ImageNetLoader.LABEL_PATTERN, dataModeEnum, splitTrainTest, imageTransform, normalizeValue, rng, null);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param imgDim an array of width, height and channels
     * @param numLabels the overall number of examples
     * @param dataModeEnum which type of data to load CLS_TRAIN, CLS_VAL, DET_TRAIN, DET_VAL
     * @param labelGenerator path label generator to use
     * @param splitTrainTest the percentage to split data for train and remainder goes to test
     * @param imageTransform how to transform the image
     * @param normalizeValue value to divide pixels by to normalize
     * @param localDir File path to an explicit directory
     * @param rng random number to lock in batch shuffling

     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, PathLabelGenerator labelGenerator, DataModeEnum dataModeEnum, double splitTrainTest, ImageTransform imageTransform, int normalizeValue, Random rng, File localDir) {
        super(new ImageNetLoader(batchSize, numExamples, numLabels, labelGenerator, dataModeEnum, splitTrainTest, rng, localDir).getRecordReader(imgDim, imageTransform, normalizeValue), batchSize, 1, numLabels);
    }

}
