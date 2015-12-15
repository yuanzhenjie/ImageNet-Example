import imagenet.Utils.ImageTransformUtils;
import org.apache.commons.io.FilenameUtils;
import org.junit.Test;

import java.io.IOException;

/**
 * Created by nyghtowl on 12/14/15.
 */
public class TestImageTransformUtils {

    @Test
    public void testConverImage() throws IOException {
        String path = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/train/");
        ImageTransformUtils util = new ImageTransformUtils();
        try {
            util.init(path, 200, 200);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
