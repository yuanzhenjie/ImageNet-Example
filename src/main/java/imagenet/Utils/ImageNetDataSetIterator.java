package imagenet.Utils;

import org.canova.api.records.reader.RecordReader;
import org.canova.image.loader.CifarLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import java.util.List;

/**
 * Created by nyghtowl on 12/18/15.
 */
public class ImageNetDataSetIterator extends RecordReaderDataSetIterator {


    protected static int width = 224;
    protected static int height = 224;
    protected static int channels = 3;

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples) {
        super(new ImageNetLoader().getRecordReader(numExamples), batchSize, width * height * channels, CifarLoader.NUM_LABELS);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param numCategories the overall number of labels
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int numCategories) {
        super(new ImageNetLoader().getRecordReader(numExamples, numCategories), batchSize, width * height * channels, numCategories);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     */
    public ImageNetDataSetIterator(int batchSize, int[] imgDim)  {
        super(new ImageNetLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2]), batchSize, imgDim[0] * imgDim[1] * imgDim[2], ImageNetLoader.NUM_CLS_LABELS);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
        super(new ImageNetLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples), batchSize, imgDim[0] * imgDim[1] * imgDim[2], ImageNetLoader.NUM_CLS_LABELS);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories) {
        super(new ImageNetLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples, numCategories), batchSize, imgDim[0] * imgDim[1] * imgDim[2], numCategories);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, int totalNumCategories) {
        super(new ImageNetLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples, numCategories), batchSize, imgDim[0] * imgDim[1] * imgDim[2], totalNumCategories);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, String version) {
        super(new ImageNetLoader(version).getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples, numCategories), batchSize, imgDim[0] * imgDim[1] * imgDim[2], numCategories);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, int totalNumCategories, String version) {
        super(new ImageNetLoader(version).getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples, numCategories), batchSize, imgDim[0] * imgDim[1] * imgDim[2], totalNumCategories);
    }

}
