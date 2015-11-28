package imagenet.Utils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Util to save and load models and parameters.
 *
 * Created by nyghtowl on 11/26/15.
 */
public class ModelUtils {

    private static final Logger log = LoggerFactory.getLogger(ModelUtils.class);

    private ModelUtils(){}

    public static void saveModelAndParameters(MultiLayerNetwork net, String basePath) throws IOException {
        String confPath = FilenameUtils.concat(basePath, net.toString()+"-conf.json");
        String paramPath = FilenameUtils.concat(basePath, net.toString() + ".bin");
        log.info("Saving model and parameters to {} and {} ...",  confPath, paramPath);

        // save parameters
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
        Nd4j.write(net.params(), dos);
        dos.flush();
        dos.close();

        // save model configuration
        FileUtils.write(new File(confPath), net.conf().toJson());
    }

    public static MultiLayerNetwork loadModelAndParameters(File confPath, String paramPath) throws IOException {
        log.info("Loading saved model and parameters...");

        // load parameters
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(confPath));
        DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
        INDArray newParams = Nd4j.read(dis);
        dis.close();

        // load model configuration
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);

        return savedNetwork;
    }

    public static void saveLayerParameters(INDArray param, String paramPath) throws IOException {
        // save parameters for each layer
        log.info("Saving parameters to {} ...", paramPath);

        DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
        Nd4j.write(param, dos);
        dos.flush();
        dos.close();
    }

    public static Layer loadLayerParameters(Layer layer, String paramPath) throws IOException{
        // load parameters for each layer
        String name = layer.conf().getLayer().getLayerName();
        log.info("Loading saved parameters for layer {} ...", name);

        DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
        INDArray param = Nd4j.read(dis);
        dis.close();
        layer.setParams(param);
        return layer;
    }

    public static void saveParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) throws IOException {
        Layer layer;
        for(int layerId: layerIds) {
            layer = model.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }

    public static void saveParameters(MultiLayerNetwork model, String[] layerIds, Map<String, String> paramPaths) throws IOException {
        Layer layer;
        for(String layerId: layerIds) {
            layer = model.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }
    public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) throws IOException {
        Layer layer;
        for(int layerId: layerIds) {
            layer = model.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return model;
    }

    public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, String[] layerIds, Map<String, String> paramPaths) throws IOException {
        Layer layer;
        for(String layerId: layerIds) {
            layer = model.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return model;
    }

    public static  Map<Integer, String>  getIdParamPaths(MultiLayerNetwork model, String basePath, int[] layerIds){
        Map<Integer, String> paramPaths = new HashMap<>();
        for (int id : layerIds) {
            paramPaths.put(id, FilenameUtils.concat(basePath, id + ".bin"));
        }

        return paramPaths;
    }

    public static Map<String, String> getStringParamPaths(MultiLayerNetwork model, String basePath, String[] layerIds){
        Map<String, String> paramPaths = new HashMap<>();

        for (String name : layerIds) {
            paramPaths.put(name, FilenameUtils.concat(basePath, name + ".bin"));
        }

        return paramPaths;
    }


}
