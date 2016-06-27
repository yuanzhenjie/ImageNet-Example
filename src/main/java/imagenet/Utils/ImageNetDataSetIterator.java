package imagenet.Utils;

import org.canova.api.io.labels.PathLabelGenerator;
import org.canova.image.loader.CifarLoader;
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
        this(batchSize, numExamples, new int[] {ImageNetLoader.HEIGHT, ImageNetLoader.WIDTH, ImageNetLoader.CHANNELS },ImageNetLoader.NUM_CLS_LABELS, "CLS_TRAIN", null, true, 1, null, 255, null, new Random(42));
    }

    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, mode returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, String mode) {
        this(batchSize, numExamples, imgDim, numLabels, mode,ImageNetLoader.LABEL_PATTERN, true, 1, null, 0, null, new Random(System.currentTimeMillis()));
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param imgDim an array of width, height and channels
     * @param numLabels the overall number of examples
     * @param mode which type of data to load CLS_TRAIN, CLS_VAL, DET_TRAIN, DET_VAL
     * @param labelGenerator path label generator to use
     * @param train true if use train value
     * @param splitTrainTest the percentage to split data for train and remainder goes to test
     * @param imageTransform how to transform the image
     * @param normalizeValue value to divide pixels by to normalize
     * @param localDir File path to an explicit directory
     * @param rng random number to lock in batch shuffling

     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, String mode, PathLabelGenerator labelGenerator, boolean train, double splitTrainTest, ImageTransform imageTransform, int normalizeValue, File localDir, Random rng) {
        super(new ImageNetLoader(batchSize, numExamples, numLabels, labelGenerator, splitTrainTest, rng, mode, localDir).getRecordReader(imgDim, imageTransform, normalizeValue, train), batchSize, 1, numLabels);
    }

}
