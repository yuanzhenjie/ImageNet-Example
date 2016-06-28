package imagenet.Utils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.io.labels.PathLabelGenerator;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.image.loader.NativeImageLoader;
import org.canova.image.recordreader.BaseImageRecordReader;
import org.canova.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 *
 */
public class ImageNetRecordReader extends BaseImageRecordReader {

    protected static Logger log = LoggerFactory.getLogger(ImageNetRecordReader.class);
    protected Map<String,String> labelFileIdMap = new LinkedHashMap<>();
    protected String fileNameMapPath; // use when the WNID is not in the filename (e.g. val labels)
    protected DataMode dataMode = DataMode.CLS_TRAIN; // use to load label ids for validation data set

    public ImageNetRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator, ImageTransform imgTransform, double normalizeValue, DataMode dataMode) {
        super(height, width, channels, labelGenerator, imgTransform, normalizeValue);
        this.dataMode = dataMode;
        this.imgNetLabelSetup();
    }

    private Map<String, String> defineLabels(String path) throws IOException {
        Map<String,String> tmpMap = new LinkedHashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;

        while ((line = br.readLine()) != null) {
            String row[] = line.split(",");
            tmpMap.put(row[0], row[1]);
            labels.add(row[1]);
        }
        return tmpMap;
    }

    private void imgNetLabelSetup() {
        // creates hashmap with WNID (synset id) as key and first descriptive word in list as the string name
        if (labelFileIdMap.isEmpty()) {
            try {
                labelFileIdMap = defineLabels(ImageNetLoader.BASE_DIR + ImageNetLoader.CLS_TRAIN_ID_TO_LABELS);
            } catch (IOException e){
                e.printStackTrace();
            }
            labels = new ArrayList<>(labelFileIdMap.values());
        }
        // creates hasmap with filename as key and WNID(synset id) as value when using val files
        if((dataMode == DataMode.CLS_VAL || dataMode == DataMode.DET_VAL) && fileNameMap.isEmpty()) {
            try {
                fileNameMap = defineLabels(ImageNetLoader.BASE_DIR + ImageNetLoader.CLS_VAL_ID_TO_LABELS);
            }  catch (IOException e){
                e.printStackTrace();
            }
        }
    }

    @Override
    public Collection<Writable> next() {
        if(iter != null) {
            File image = iter.next();

            if(image.isDirectory())
                return next();

            try {
                invokeListeners(image);
                return load(imageLoader.asRowVector(image), image.getName());
            } catch (Exception e) {
                e.printStackTrace();
            }

            Collection<Writable> ret = new ArrayList<>();
            if(iter.hasNext()) {
                return ret;
            }
            else {
                if(iter.hasNext()) {
                    try {
                        image = iter.next();
                        invokeListeners(image);
                        ret.add(new Text(FileUtils.readFileToString(image)));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            return ret;
        }
        else if(record != null) {
            hitImage = true;
            invokeListeners(record);
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    private Collection<Writable> load(INDArray image, String filename) throws IOException {
        int labelId = -1;
        Collection<Writable> ret = RecordConverter.toRecord(image);
        if (dataMode != DataMode.CLS_VAL || dataMode != DataMode.DET_VAL) {
//            String WNID = FilenameUtils.getBaseName(filename).split(pattern)[patternPosition];
            Writable WNID = labelGenerator.getLabelForPath(filename);
            labelId = labels.indexOf(labelFileIdMap.get(WNID.toString()));
        } else {
            String fileName = FilenameUtils.getName(filename); // currently expects file extension
            labelId = labels.indexOf(labelFileIdMap.get(fileNameMap.get(fileName)));
        }
        if (labelId >= 0)
            ret.add(new IntWritable(labelId));
        else
            throw new IllegalStateException("Illegal label " + labelId);
        return ret;
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream ) throws IOException {
        invokeListeners(uri);
        imgNetLabelSetup();
        return load(imageLoader.asRowVector(dataInputStream), FilenameUtils.getName(uri.getPath()));
    }
}
