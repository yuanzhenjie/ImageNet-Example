package imagenet.Utils;

/**
 * ImageNet DataMode
 *
 * Defines which dataset between object recognition (CLS) and location identification (DET).
 * Also defines whether its train, cross validation or test phase
 */
public enum DataMode {
    CLS_TRAIN, CLS_VAL, CLS_TEST, DET_TRAIN, DET_VAL, DET_TEST
}
