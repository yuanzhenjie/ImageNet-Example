package imagenet;

import imagenet.Utils.DataUtils;
import imagenet.Utils.ImageTransformUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;

/**
 * @deprecated
 * Use this script to run ImageTransformUtils to transform images by cropping, resizing and setting channel.
 */

@Deprecated
public class TransformImages {

    public static void main(String[] args) {

        // TODO pass in dimensions 224 x 224 and do the
        DataUtils util = new DataUtils();
        String pathTrain = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/train/");
        String pathTest = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/test/");

        util.init(new File(pathTrain));
        util.init(new File(pathTest));
    }

}
