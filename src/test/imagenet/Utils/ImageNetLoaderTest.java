package imagenet.Utils;

import org.canova.api.records.reader.RecordReader;
import org.junit.Test;

import java.io.File;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by nyghtowl on 12/18/15.
 */
public class ImageNetLoaderTest {

    @Test
    public void testLoader() {
        File dir = new File(ImageNetLoader.BASE_DIR, ImageNetLoader.LOCAL_TRAIN_DIR);
        new ImageNetLoader(1, 1, 1, null, DataModeEnum.CLS_TRAIN, 1, new Random(42), dir);
        assertTrue(dir.exists());
    }

    @Test
    public void testReader() throws Exception {
        final int numRows = 224;
        final int numColumns = 224;
        int nChannels = 3;
        int numCategories = ImageNetLoader.NUM_CLS_LABELS;

        RecordReader record = new ImageNetLoader(1, 1, numCategories, null, DataModeEnum.CLS_TRAIN, 1, new Random(42), null).getRecordReader(new int[]{numRows, numColumns, nChannels}, null, 255);
        List<String> numLabels = record.getLabels();
        assertEquals(numCategories, numLabels.size());

    }
}
