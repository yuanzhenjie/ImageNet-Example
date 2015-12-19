package imagenet.Utils;

import org.canova.api.records.reader.RecordReader;
import org.junit.Test;

import java.io.File;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by nyghtowl on 12/18/15.
 */
public class ImageNetLoaderTest {

    @Test
    public void testLoader() {
        File dir = new File(ImageNetLoader.BASE_DIR, ImageNetLoader.LOCAL_TRAIN_DIR);
        new ImageNetLoader(dir);
        assertTrue(dir.exists());
    }

    @Test
    public void testReader() throws Exception {
        final int numRows = 224;
        final int numColumns = 224;
        int nChannels = 3;
        int batchSize = 2;
        int numBatches = 1;
        int numCategories = ImageNetLoader.NUM_CLS_LABELS;
        int totalTrainNumExamples = batchSize * numBatches;


        RecordReader record = new ImageNetLoader("CLS_VAL").getRecordReader(totalTrainNumExamples, numCategories);
        List<String> numLabels = record.getLabels();
        assertEquals(numCategories, numLabels.size());

    }
}
