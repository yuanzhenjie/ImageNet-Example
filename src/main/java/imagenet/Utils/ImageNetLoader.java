package imagenet.Utils;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang.NotImplementedException;
import org.canova.api.io.filters.BalancedPathFilter;
import org.canova.api.io.labels.PathLabelGenerator;
import org.canova.api.io.labels.PatternPathLabelGenerator;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.validation.constraints.Null;
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
    public final static int NUM_CLS_LABELS = 1861; // 1000 main with 860 ancestors

    public final static int NUM_DET_TRAIN_IMAGES = 395918;
    public final static int NUM_DET_VAL_IMAGES = 20121;
    public final static int NUM_DET_TEST_IMAGES = 40152;

    public final static int WIDTH = 224;
    public final static int HEIGHT = 224;
    public final static int CHANNELS = 3;

    public final static String BASE_DIR = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
    public final static String LOCAL_TRAIN_DIR = "train";
    public final static String LOCAL_VAL_DIR = "test";
    public final static String CLS_TRAIN_ID_TO_LABELS = "cls-loc-labels.txt";
    public final static String CLS_VAL_ID_TO_LABELS = "cls-loc-val-map.txt";
    public String urlTrainFile = "image_train_urls.txt";
    public String urlValFile = "image_test_urls.txt";
    protected String labelFilePath;

    protected List<String> labels = new ArrayList<>();
    protected Map<String,String> labelIdMap = new LinkedHashMap<>();

    protected File fullTrainDir = new File(BASE_DIR, LOCAL_TRAIN_DIR);
    protected File fullTestDir = new File(BASE_DIR, LOCAL_VAL_DIR);
    protected File sampleURLTrainList = new File(BASE_DIR, urlTrainFile);
    protected File sampleURLTestList = new File(BASE_DIR, urlValFile);

    protected File fullDir;
    protected File urlList;
    protected InputSplit[] inputSplit;
    protected int batchSize;
    protected int numExamples;
    protected int numLabels;
    protected PathLabelGenerator labelGenerator;
    protected double splitTrainTest;
    protected Random rng;

    protected DataModeEnum dataModeEnum; // CLS_Train, CLS_VAL, CLS_TEST, DET_TRAIN, DET_VAL, DET_TEST
    protected final static String REGEX_PATTERN = Pattern.quote("_");
    public final static PathLabelGenerator LABEL_PATTERN = new PatternPathLabelGenerator(REGEX_PATTERN);
    protected RecordReader recordReader;

    public ImageNetLoader(int batchSize, int numExamples, int numLabels, @Null PathLabelGenerator labelGenerator, DataModeEnum dataModeEnum, @Null double splitTrainTest, @Null Random rng, @Null File localDir){
        this.batchSize = batchSize;
        this.numExamples = numExamples;
        this.numLabels = numLabels;
        this.labelGenerator = labelGenerator == null? LABEL_PATTERN: labelGenerator;
        this.labelFilePath = (dataModeEnum == DataModeEnum.CLS_VAL || dataModeEnum == DataModeEnum.DET_VAL)? CLS_VAL_ID_TO_LABELS: CLS_TRAIN_ID_TO_LABELS;
        this.splitTrainTest = Double.isNaN(splitTrainTest)? 1: splitTrainTest;
        this.rng = rng == null? new Random(System.currentTimeMillis()): rng;
        this.dataModeEnum = dataModeEnum;
        switch (dataModeEnum) {
            case CLS_TRAIN:
                this.fullDir = localDir == null? fullTrainDir: localDir;
                this.urlList = sampleURLTrainList;
                load();
                break;
            case CLS_TEST:
                this.fullDir =  localDir == null? fullTestDir: localDir;
                this.urlList = sampleURLTestList;
                load();
                break;
            case CLS_VAL:
            case DET_TRAIN:
            case DET_VAL:
            case DET_TEST:
                throw new NotImplementedException("Detection has not been setup yet");
            default:
                break;
        }
    }

    @Override
    public INDArray asRowVector(File f) throws IOException {
        return null;
    }

    @Override
    public INDArray asRowVector(InputStream inputStream) throws IOException {
        return null;
    }

    @Override
    public INDArray asMatrix(File f) throws IOException {
        return null;
    }

    @Override
    public INDArray asMatrix(InputStream inputStream) throws IOException {
        return null;
    }

    public  Map<String, String> generateMaps(String filesFilename, String url) {
        Map<String, String> imgNetData = new HashMap<>();
        imgNetData.put("filesFilename", filesFilename);
        imgNetData.put("filesURL", url);
        return imgNetData;
    }

    // TODO finish setting up the following and passing into the record reader...
    private void defineLabels(File labelFilePath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(labelFilePath));
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

    public void load()  {
        defineLabels(new File(BASE_DIR, labelFilePath));
        // Downloading a sample set of data if not available
        if (!fullDir.exists()) {
            fullDir.mkdir();
            log.info("Downloading {}...", FilenameUtils.getBaseName(fullDir.toString()));
            CSVRecordReader reader = new CSVRecordReader(7, ",");
            int count = 0;
            try {
                reader.initialize(new FileSplit(urlList));
            } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

        while(reader.hasNext()) {
                Collection<Writable> val = reader.next();
                Object url =  val.toArray()[1];
                String fileName = val.toArray()[0] + "_" + count++ + ".jpg";
                downloadAndUntar(generateMaps(fileName, url.toString()), fullDir);
                try{
                    downloadAndUntar(generateMaps(fileName, url.toString()), fullDir);
                }
                catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
        FileSplit fileSplit = new FileSplit(fullDir, ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, ALLOWED_FORMATS,labelGenerator, numExamples, numLabels, 0, 0, null);
        inputSplit = fileSplit.sample(pathFilter, numExamples*splitTrainTest, numExamples*(1-splitTrainTest));
    }

    public RecordReader getRecordReader() {
        return getRecordReader(new int[]{ HEIGHT, WIDTH, CHANNELS}, null, 255);
    }

    public RecordReader getRecordReader(int[] imgDim) {
        return getRecordReader(imgDim, null, 255);
    }

    public RecordReader getRecordReader(int[]imgDim, ImageTransform imageTransform, int normalizeValue) {
        load();
        recordReader = new ImageNetRecordReader(imgDim[0], imgDim[1], imgDim[2], labelGenerator, imageTransform, normalizeValue, dataModeEnum);

        try {
            InputSplit data = (dataModeEnum == DataModeEnum.CLS_TRAIN || dataModeEnum == DataModeEnum.DET_TRAIN)? inputSplit[0]: inputSplit[1];
            recordReader.initialize(data);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return recordReader;
    }

    public List<String> getLabels(){
        return labels;
    }

    public RecordReader getTrain() throws Exception{
        recordReader.initialize(inputSplit[0]);
        return recordReader;
    }

    public RecordReader getTest() throws Exception{
        recordReader.initialize(inputSplit[1]);
        return recordReader;
    }

    public RecordReader getCrossVal() throws Exception{
        recordReader.initialize(inputSplit[2]);
        return recordReader;
    }

}
