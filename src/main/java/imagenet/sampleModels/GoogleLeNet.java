package imagenet.sampleModels;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Reference: http://arxiv.org/pdf/1409.4842v1.pdf
 * Created by nyghtowl on 9/11/15.
 */

public class GoogleLeNet {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;

    public GoogleLeNet(int height, int width, int channels, int outputNum, long seed, int iterations) {
        // TODO consider ways to make this adaptable to other problems not just imagenet
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //TODO this could be optimized with line but the paper calls for SGD
                .learningRate(1e-4) // TODO reduce by 4% every 8 epochs
                .momentum(0.9)
                .list(10)
                // TODO add lr and decay for bias?
                .layer(0, new ConvolutionLayer.Builder(new int[]{7, 7}, new int[]{2, 2}, new int[]{3, 3})
                        .nIn(channels)
                        .nOut(64)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                // TODO LRN? with alpha .0001 and beta .75
                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1}, new int[]{1, 1})
                        .nOut(64)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())
                .layer(, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(192)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())
                        // TODO LRN? with alpha .0001 and beta .75
                .layer(, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .build())
                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(64)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())
                // TODO connect to last Subsampling
                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(96)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())
                        // TODO connect to last Subsampling
                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(16)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                        // TODO connect to last CNN 3 x 3
                .layer(, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(128)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                // TODO connect to ?
                .layer(, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .nOut(32)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())
                .layer(, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(32)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                // TODO concat 4 previous layers (CNN & pool)

                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(128)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(128)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .nOut(192)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(32)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .nOut(96)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                .layer(, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(192)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

                        // TODO concat 4 previous layers (CNN & pool)

                .layer(, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .build())

                .layer(, new ConvolutionLayer.Builder(new int[]{1, 1})
                        .nOut(64)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.2)
                        .learningRate(1)
                        .l2(1)
                        .build())

// 644
                .layer(, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{7, 7}, new int[]{1, 1})
                        .build())
                .layer(, new DenseLayer.Builder()
                        .nOut(1000)
                        .activation("relu")
                        .dropOut(0.4)
                        .learningRate(1)
                        .l2(1)
                        .build())
                .layer(, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false);


        new ConvolutionLayerSetup(conf,height,width,channels);
        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        return model;
    }


}
