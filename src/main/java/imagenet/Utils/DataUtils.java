package imagenet.Utils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

/**
 * Utilities to load, parse, relabel data.
 * Target to pull in mixed batches of images and store as sequence files
 * Leverage PreProcess data to store
 */
public class DataUtils {
    private static final Logger log = LoggerFactory.getLogger(DataUtils.class);

    // TODO pass in the main paths
    protected static final String TEMP_DIR = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "ImageNetSeqFiles");
    protected static String basePath = ImageNetLoader.BASE_DIR;
    protected static String inputFolder = "train";
    protected static String outputFolder = "seqTrain";
    protected static String inputPath = FilenameUtils.concat(basePath, inputFolder);
    protected static String outputPath = FilenameUtils.concat(basePath, outputFolder);

    public static void checkDirExists(File path){
        if (!path.exists()) {
            try {
                path.mkdir();
                return;
            } catch (SecurityException se) {
                se.printStackTrace();
            }
        }
    }

    // flip images
    public static void flipFiles(ImageTransformUtils imgTransform, int rotationAngle){
        imgTransform.flipImage(rotationAngle); // 180 and other values
    }

    // center and resize images
    public void initCenterResize(ImageTransformUtils imgTransform, int newW, int newH){
        imgTransform.centerResize(newW, newH);
    }

    // change pixels...

    // change file contents
    public void init(File inputPath) {
        boolean recursive = true;

        ImageTransformUtils imgTransform = new ImageTransformUtils();

        if(inputPath.isDirectory()) {
            Iterator iter = FileUtils.iterateFiles(inputPath, ImageNetLoader.ALLOWED_FORMATS, recursive);

            int numSaved = 0;
            int numDel = 0;
            while(iter.hasNext()) {
                File fileName = (File) iter.next();
                imgTransform.convertToBuffer(fileName);
                // ******** add other transforms here ********
                if(imgTransform.getImage()==null)
                    numDel++;
                else {
                    imgTransform.saveImage(new File(outputPath+numSaved));
                    numSaved++;
                }
            }
            log.info("Number of files deleted: " + numDel);
        } else {
            imgTransform.convertToBuffer(inputPath);
        }

    }

    // rename files
    public void fixFileName(Path path) {
        try {
            Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path filePath, BasicFileAttributes attrs) throws IOException {
                    Path fileName = filePath.getFileName();
                    Path dirPath = filePath.getParent();
                    String newFileName = fileName.toString().trim().toLowerCase().replace(" ", "_");
                    Path newFilePath = new File(dirPath.toString(), newFileName).toPath();
                    Files.move(filePath, newFilePath);
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // track size of files and group them into 128MB size
    public static void segmentFileSplit(File inputPath, PreProcessData ppd){
        long length = 0;
        long limitSize = 128 * 1000000;
        Iterator<File> subFiles = FileUtils.iterateFiles(inputPath, ImageNetLoader.ALLOWED_FORMATS, true);
        String storePath = FilenameUtils.concat(TEMP_DIR, "storePath");
        File storeFiles = new File(storePath);
        checkDirExists(storeFiles);

        // TODO add below to FileSplit
        while(subFiles.hasNext()) {
            int count = 0;
            while (length < limitSize) {
                if (subFiles.hasNext()) {
                    File f = subFiles.next();
                    try {
                        Files.copy(f.toPath(), new File(storePath, f.getName()).toPath(), StandardCopyOption.REPLACE_EXISTING);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    length += f.length();
                } else
                    break;
            }

            // TODO process images before grouping into sequence files
            ppd.setupSequnceFile(storePath + "/*", outputPath + "/" + count);
            ppd.checkFile(outputPath + "/" + count);
            count++;
        }
        try {
            FileUtils.deleteDirectory(storeFiles);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // TODO diversify what files are pulled out by walking path?
//        Path path = Paths.get(inputPath.toString());
//        FileVisitor<Path> fv = new SimpleFileVisitor<Path>() {
//            @Override
//            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
//                    throws IOException {
//                ;
//                return FileVisitResult.CONTINUE;
//            }
//        };
//
//        try {
//            Files.walkFileTree(path, fv);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

    }

    public static void main(String[] args){
        boolean save = true;
        // TODO change what is called - group images & make diverse, convert images, save as sequence
        PreProcessData ppd = new PreProcessData(save);
        checkDirExists(new File(inputPath));
        checkDirExists(new File(outputPath));
        segmentFileSplit(new File(inputPath), ppd);

    }

}
