package imagenet.Utils;

import imagenet.Utils.DataUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;

/**
 * Use this script to run ImageTransformUtils to remove images that are empty.
 */

public class TransformImages {

    public static void main(String[] args) {

        DataUtils util = new DataUtils();
        String pathTrain = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/train/");
        String pathTest = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/test/");

        util.init(new File(pathTrain));
        util.init(new File(pathTest));
    }

}
