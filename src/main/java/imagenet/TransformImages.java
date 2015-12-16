package imagenet;

import imagenet.Utils.ImageTransformUtils;
import org.apache.commons.io.FilenameUtils;

/**
 * Created by nyghtowl on 12/15/15.
 * Use this script to run ImageTransformUtils to transform images by cropping, resizing and setting channel.
 */
public class TransformImages {

    public static void main(String[] args) {

        ImageTransformUtils util = new ImageTransformUtils(224, 224);
        String pathTrain = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/train/");
        String pathTest = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/test/");

        util.init(pathTrain);
        util.init(pathTest);
    }

}
