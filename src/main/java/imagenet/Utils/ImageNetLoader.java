package imagenet.Utils;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang.NotImplementedException;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.LimitFileSplit;
import org.canova.api.writable.Writable;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageNetRecordReader;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Canova Loader specific to this project.
 */

public class ImageNetLoader extends BaseImageLoader implements Serializable{

    public final static int NUM_CLS_TRAIN_IMAGES = 1281167;
    public final static int NUM_CLS_VAL_IMAGES = 50000;
    public final static int NUM_CLS_TEST_IMAGES = 100000;
    public final static int NUM_CLS_LABELS = 1860; // 1000 main with 860 ancestors

    public final static int NUM_DET_TRAIN_IMAGES = 395918;
    public final static int NUM_DET_VAL_IMAGES = 20121;
    public final static int NUM_DET_TEST_IMAGES = 40152;

    public final static int WIDTH = 224;
    public final static int HEIGHT = 224;
    public final static int CHANNELS = 3;

    public final static String BASE_DIR = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
    public final static String LABEL_FILENAME = "cls-loc-labels.txt";
    public final static String VAL_MAP_FILENAME = "cls-loc-val-map.txt";
    public String urlTrainFile = "image_train_urls.txt";
    public String urlValFile = "image_test_urls.txt";
    protected String regexPattern = Pattern.quote("_");

    protected List<String> labels = new ArrayList<>();
    protected Map<String,String> labelIdMap = new LinkedHashMap<>();

    public final static String LOCAL_TRAIN_DIR = "train";
    protected File fullTrainDir = new File(BASE_DIR, LOCAL_TRAIN_DIR);
    public final static String LOCAL_VAL_DIR = "test";
    protected File fullTestDir = new File(BASE_DIR, LOCAL_VAL_DIR);

    protected File fullDir;

    protected int numExamples = NUM_CLS_TRAIN_IMAGES;
    protected int numLabels = NUM_CLS_LABELS;

    protected String mode = "CLS_TRAIN"; // CLS_Train, CLS_VAL, CLS_TEST, DET_TRAIN, DET_VAL, DET_TEST

    public ImageNetLoader(File localDir){
        this.fullDir = localDir;
        switch (mode) {
            case "CLS_TRAIN":
                load(fullDir, new File(BASE_DIR, urlTrainFile));
                break;
            case "CLS_VAL":
                load(fullDir, new File(BASE_DIR, urlValFile));
                break;
            case "DET_TRAIN":
                throw new NotImplementedException("Detection has not been setup yet");
            case "DET_VAL":
                throw new NotImplementedException("Detection has not been setup yet");
        }
    }

    public ImageNetLoader() {
        this.fullDir = fullTrainDir;
        load(fullDir, new File(BASE_DIR, urlTrainFile));
    }

    public ImageNetLoader(String mode) {
        this.mode = mode;
        switch (mode) {
            case "CLS_TRAIN":
                this.fullDir = fullTrainDir;
                load(fullDir, new File(BASE_DIR, urlTrainFile));
                break;
            case "CLS_VAL":
                this.fullDir = fullTestDir;
                load(fullDir, new File(BASE_DIR, urlValFile));
                break;
            case "DET_TRAIN":
                throw new NotImplementedException("Detection has not been setup yet");
            case "DET_VAL":
                throw new NotImplementedException("Detection has not been setup yet");
        }
    }

    public  Map<String, String> generateMaps(String filesFilename, String url) {
        Map<String, String> imgNetData = new HashMap<>();
        imgNetData.put("filesFilename", filesFilename);
        imgNetData.put("filesURL", url);
        return imgNetData;
    }

    // TODO finish setting up the following and passing into the record reader...
    private void defineLabels() {
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(BASE_DIR, LABEL_FILENAME)));
            String line;

            while ((line = br.readLine()) != null) {
                String row[] = line.split(",");
                labelIdMap.put(row[0], row[1]);
                labels.add(row[1]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(File dir, File data)  {
        defineLabels();
        if (!dir.exists()) {
            dir.mkdir();
            log.info("Downloading {}...", FilenameUtils.getBaseName(dir.toString()));
            CSVRecordReader reader = new CSVRecordReader(7, ",");
            try {
                reader.initialize(new FileSplit(data));
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            int count = 0;
            while(reader.hasNext()) {
                Collection<Writable> val = reader.next();
                Object url =  val.toArray()[1];
                String fileName = val.toArray()[0] + "_" + count++ + ".jpg";
                downloadAndUntar(generateMaps(fileName, url.toString()), dir);
                try{
                    downloadAndUntar(generateMaps(fileName, url.toString()), dir);
                }
                catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
    }

    public RecordReader getRecordReader() {
        return getRecordReader(WIDTH, HEIGHT, CHANNELS, true, regexPattern);
    }

    public RecordReader getRecordReader(int width, int height, int channels) {
        return getRecordReader(width, height, channels, true, regexPattern);
    }

    public RecordReader getRecordReader(int numExamples) {
        this.numExamples = numExamples;
        return getRecordReader(WIDTH, HEIGHT, CHANNELS, true, regexPattern);
    }

    public RecordReader getRecordReader(int numExamples, int numCategories) {
        this.numExamples = numExamples;
        this.numLabels = numCategories;
        return getRecordReader(WIDTH, HEIGHT, CHANNELS, true, regexPattern);
    }

    public RecordReader getRecordReader(int width, int height, int channels, int numExamples) {
        this.numExamples = numExamples;
        return getRecordReader(width, height, channels, true, regexPattern);
    }

    public RecordReader getRecordReader(int width, int height, int channels, int numExamples, int numCategories) {
        this.numExamples = numExamples;
        this.numLabels = numCategories;
        return getRecordReader(width, height, channels, true, regexPattern);
    }

    public RecordReader getRecordReader(int width, int height, int channels, boolean appendLabel, String regexPattern) {
        RecordReader recordReader = new ImageNetRecordReader(width, height, channels, FilenameUtils.concat(BASE_DIR, LABEL_FILENAME), appendLabel, regexPattern);
        try {
            recordReader.initialize(new LimitFileSplit(fullDir, ALLOWED_FORMATS, numExamples, numLabels, regexPattern, 0, rng));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return recordReader;
    }

    public List<String> getLabels(){
        return labels;
    }

}
