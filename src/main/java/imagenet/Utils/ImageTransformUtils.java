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
 * Project class to transform images
 *
 * Includes cropping and resize. Note the functionality here has been transfered and covered in ImageReader.
 * Thus this is left more for reference.
 */

public class ImageTransformUtils {
    private static final Logger log = LoggerFactory.getLogger(ImageTransformUtils.class);
    protected String[] allowFormat = {"jpg", "jpeg", "JPG", "JPEG"};
    protected int newW = 28;
    protected int newH = 28;
    protected int channelType = BufferedImage.TYPE_INT_RGB;

    public ImageTransformUtils(int newW, int newH){
        this.newW = newW;
        this.newH = newH;
    }

    public ImageTransformUtils(int newW, int newH, int channelType){
        this.newW = newW;
        this.newH = newH;
        this.channelType = channelType;
    }

    public ImageTransformUtils(int newW, int newH, String[] allowedFormat){
        this.newW = newW;
        this.newH = newH;
        this.allowFormat = allowFormat;
    }

    public ImageTransformUtils(){}

    public int[] imgStats(BufferedImage img){
        int width = img.getWidth();
        int height = img.getHeight();
        return new int[] {width, height};
    }

    public BufferedImage centerCrop(BufferedImage img, int[] imgStats) {
        int x = 0;
        int y = 0;
        int width = imgStats[0];
        int height = imgStats[1];
        int diff = Math.abs(width - height) / 2;

        if (width > height) {
            x = diff;
            width = width - diff;
        } else if (height > width) {
            y = diff;
            height = height - diff;
        }
        return img.getSubimage(x, y, width, height);
    }

    public BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, channelType);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }

    public int transformImage(File fileName, int numDel) {
        try {
            BufferedImage img = ImageIO.read(fileName);
            if (img != null) {
                int[] imgStats = imgStats(img);
                int orig_w = imgStats[0];
                int orig_h = imgStats[1];

                // center and crop TODO allow to create non square shapes
                if ( orig_w != orig_h) {
                    img = centerCrop(img, imgStats);
                }

                // resize based on set size entered - defaults 28 x 28 and set rgb channel
                if (newW != orig_w && newH != orig_h)
                    ImageIO.write(resize(img, newW, newH), "jpg", fileName);
            } else {
                log.warn("This file is empty and has been deleted", fileName);
                fileName.delete();
                numDel += 1;
            }
        } catch (IOException e) {
            log.warn("Caught an IOException: " + e);
        }
        return numDel;
    }

    public void init(String name) {
        File path = new File(name);
        boolean recursive = true;
        int numDel = 0;

        if(path.isDirectory()) {
            Iterator iter = FileUtils.iterateFiles(path, allowFormat, recursive);
            while(iter.hasNext()) {
                File fileName = (File) iter.next();
                numDel = transformImage(fileName, numDel);
            }
        } else {
            numDel = transformImage(new File(name), numDel);
        }
        log.info("Number of files deleted: " + numDel);
    }
}
