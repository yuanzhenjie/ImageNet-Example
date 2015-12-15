package imagenet.Utils;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;

/**
 * Created by nyghtowl on 12/14/15.
 */
public class ImageTransformUtils {
    private static final Logger log = LoggerFactory.getLogger(ImageTransformUtils.class);
    protected String[] allowFormat = {"jpg", "jpeg", "JPG", "JPEG"};

    public ImageTransformUtils(){}

    public BufferedImage centerCrop(File file) throws IOException {
        BufferedImage bimg = ImageIO.read(file);
        int x= 0;
        int y = 0;
        int width = bimg.getWidth();
        int height = bimg.getHeight();
        int diff = Math.abs(width - height)/2;

        if (width > height) {
            x = diff;
            width = width - diff;
        } else if (height > width) {
            y = diff;
            height = height - diff;
        }

        return bimg.getSubimage(x, y, width, height);
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }

    public void init(String name, int newW, int newH) throws IOException {
        File path = new File(name);
        boolean recursive = true;
        Iterator iter = null;

        if(path.isDirectory()) {
            iter = FileUtils.iterateFiles(path, allowFormat, recursive);
        } else {
            log.warn("Submit a directory of image files to convert");
        }


        while(iter.hasNext()) {
            File fileName = (File) iter.next();
            double len = fileName.length();
            if (fileName.length() > 8000) {
                BufferedImage img = resize(centerCrop(fileName), newW, newH);
                ImageIO.write(img, "jpg", fileName);
            } else {
                log.warn("This file is empty and has been deleted", fileName);
                fileName.delete();
            }
        }
    }
}
