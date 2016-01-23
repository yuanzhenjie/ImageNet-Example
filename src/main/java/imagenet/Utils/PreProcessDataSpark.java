package imagenet.Utils;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.spark.functions.data.FilesAsBytesFunction;

/**
 * Prep data as sequence files to group small files into larger batches to enable
 * optimized processing with Spark and Hadoop.
 *
 */
public class PreProcessDataSpark {

    protected static final String TEMP_DIR = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "ImageNetSeqFiles");
    protected static String SEQUENCE_DIR;
    protected String inputPath;
    protected String outputPath;
    protected JavaPairRDD<Text, BytesWritable> sequenceFile;
    protected boolean save = false;

    public PreProcessDataSpark(String inputPath, String outputPath) {
        this(null, inputPath, outputPath, false);
    }

    public PreProcessDataSpark(JavaSparkContext sc, String inputPath, String outputPath, boolean save){
        // pass in file path with *.[file type] (e.g. jpg) to limit type of files loaded or use main dir and ensure only image files in subDir
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.save = save;
        setupSequnceFile(sc);
    }

    public void setupSequnceFile(JavaSparkContext sc){

        if(sc == null) {
            // Spark configuration
            SparkConf sparkConf = new SparkConf();
            sparkConf.setAppName("DataSequence");
            sparkConf.setMaster("local[*]");
            sc = new JavaSparkContext(sparkConf);
        }

        // Load and convert data
        JavaPairRDD<String, PortableDataStream> stream = sc.binaryFiles(inputPath);
        sequenceFile = stream.mapToPair(new FilesAsBytesFunction());
        sequenceFile.cache();
        System.out.println("\n************Sequence file loaded************************");
        // Save SequenceFiles:
        if(save) {
            String parentPath = outputPath != null? outputPath: "tmp";
            SEQUENCE_DIR = FilenameUtils.concat(TEMP_DIR, parentPath);
            sequenceFile.saveAsNewAPIHadoopFile(SEQUENCE_DIR, Text.class, BytesWritable.class, SequenceFileOutputFormat.class);
            System.out.println("\n****************Sequence files saved at \" + outputPath.toString() + \" ********************");
        }


    }

    public JavaPairRDD<Text, BytesWritable> getFile(){
        return sequenceFile;
    }

    public void unpersist(){
        sequenceFile.unpersist();
    }

}
