package imagenet.Utils;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;

/**
 * Project class to transform images
 *
 * Includes cropping and resize. Note the functionality here has been transfered and covered in ImageReader.
 * Thus this is left more for reference.
 */

@Deprecated
public class ImageTransformUtils {
    private static final Logger log = LoggerFactory.getLogger(ImageTransformUtils.class);
    protected int channelType = BufferedImage.TYPE_INT_RGB;
    protected BufferedImage bImg;

    public ImageTransformUtils(){}

    public void centerCrop() {
        int x = 0;
        int y = 0;
        int width = bImg.getWidth();
        int height = bImg.getHeight();
        int diff = Math.abs(width - height) / 2;

        if (width > height) {
            x = diff;
            width = width - diff;
        } else if (height > width) {
            y = diff;
            height = height - diff;
        }
        bImg = bImg.getSubimage(x, y, width, height);
    }

    public void resize(int newW, int newH) {
        Image tmp = bImg.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, channelType);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        bImg = dimg;
    }

    public void flipImage(double rotationAngle) {
        AffineTransform transform = new AffineTransform();
        transform.rotate(rotationAngle, bImg.getWidth()/2, bImg.getHeight()/2);

        AffineTransformOp op = new AffineTransformOp(transform, AffineTransformOp.TYPE_BILINEAR);
        bImg = op.filter(bImg, null);
    }

    public void changeChannel(int type){
        // TODO alpha only works with png so need to save as png if that is the output
        // Options: BufferedImage.TYPE_BYTE_GRAY, TYPE_3BYTE_BGR, BufferedImage.TYPE_4BYTE_ABGR
        bImg = new BufferedImage(bImg.getWidth(), bImg.getHeight(), type);
        Graphics2D bGr = bImg.createGraphics();
        bGr.drawImage(bImg, 0, 0, null);
        bGr.dispose();
    }

    public void subtractConstantFromPixels(int constant){
        // pull the color and subtract constant
        // pull color and subtract other colors
        // TODO finish building
        byte[] pixels = ((DataBufferByte)bImg.getRaster().getDataBuffer()).getData();
    }

    public void centerResize(int newW, int newH) {
        // center and crop image
        int orig_w = bImg.getWidth();
        int orig_h = bImg.getHeight();

        // center and crop
        if ( orig_w != orig_h) {
            centerCrop();
        }
        // resize based on set size entered - defaults 28 x 28 and set rgb channel
        if (newW != orig_w && newH != orig_h)
            resize(newW, newH);

    }

    public void saveImage(File fileName){
        try {
            ImageIO.write(bImg, "jpg", fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void convertToBuffer(File fileName){
        BufferedImage img = null;
        try {
            img = ImageIO.read(fileName);
            if (img == null) {
                log.warn("This file is empty and has been deleted", fileName);
                fileName.delete();
                return;
            }
            bImg = img;
        } catch (IOException e) {
            log.warn("Caught an IOException: " + e);
        }
    }

    public BufferedImage getImage(){
        return bImg;
    }

}
