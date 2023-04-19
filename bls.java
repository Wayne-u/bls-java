
/**
 * Created with vscode.
 * @Author: Liwei Zhou
 * @Date: ${2023}/${4}/${17}/${15:00:00}
 * @Description: bls model v3.0
 */
import java.io.IOException;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;

public class bls {
    private int N1;// feature numbers
    private int N2;// feature group numbers
    private int N3;// enhanced node numbers
    private int taskType;// task type(0:regression, 1:classification)
    private int trainingMode;
    private double trainAcc;
    private double trainRmse;
    private double trainCC;
    private double testAcc;
    private double testRmse;
    private double testCC;
    private double trainTime;
    private double testTime;
    private double s;
    private double c;
    private int labelClass;
    private int trainDataNum;
    private int testDataNum;
    private int featureNum;
    private int baseDataNum;
    private double parameterOfShrink;
    private double solve_time;
    private int alreadyTrained;
    private int debugMode;
    private DataLoader MatrixLoader;
    private MinMaxScaler scalerOfFeatureMapping;
    private String activeFunc;
    private DenseMatrix trainSet;
    private DenseMatrix testSet;
    private DenseMatrix weightOfInputLayer;
    private DenseMatrix weightOfOutputLayer;
    private DenseMatrix weightOfEnhancedLayer;
    private DenseMatrix inputOfOutputLayerInv;
    private DenseMatrix minOfEachWindow;
    private DenseMatrix distOfMaxAndMin;
    private DenseMatrix baseInputOfOutputLayerTrain;

    /**
     * Construct a bls network.
     * 
     * @param task_type
     *                           0: regression task
     *                           1: classification task
     * @param featureNum
     *                           feature node numbers
     * @param featureGroupNum
     *                           feature group numbers
     * @param enhancedNodeNum
     *                           enhanced node numbers
     * @param scale
     *                           scale in scaling the max value of
     *                           outputOfEnhancedLayer
     * @param lumda
     *                           lumda in Pseudo-inverse calculation through ridge
     *                           regression algorithm
     * @param ActivationFunction
     *                           activate function in enhanced layer output
     *                           calculation
     * 
     */
    public bls(int task_type, int _featureNum, int featureGroupNum, int enhancedNodeNum, double scale, double lumda,
            String ActivationFunction) {
        taskType = task_type;
        N1 = _featureNum;
        N2 = featureGroupNum;
        N3 = enhancedNodeNum;
        activeFunc = ActivationFunction;
        s = scale;
        c = lumda;
        labelClass = 0;
        trainAcc = 0;
        trainRmse = 0;
        trainCC = 0;
        testAcc = 0;
        testRmse = 0;
        testCC = 0;
        trainDataNum = 0;
        testDataNum = 0;
        featureNum = 0;
        baseDataNum = 0;
        trainingMode = 0;
        solve_time = 0;
        alreadyTrained = 0;
        debugMode = 0;
        MatrixLoader = new DataLoader();
    }

    /**
     * load weight file from disk
     * <p>
     * if you are using trainingMode 1, you can load weightOfInputLayer,
     * weightOfEnhancedLayer
     * <p>
     * if you are using trainingMode 2, you can load weightOfInputLayer,
     * weightOfEnhancedLayer, weightOfOutputLayer,otherParameters
     * 
     * @param weightFile
     *                   weight file path
     * @param weightType
     *                   weight type, include "weightOfInputLayer",
     *                   "weightOfEnhancedLayer",
     *                   "weightOfOutputLayer",
     *                   "inputOfOutputLayerInv",
     *                   "otherParameters",
     *                   "minOfEachWindow",
     *                   "distOfMaxAndMin",
     *                   "baseInputOfOutputLayerTrain"
     */
    public void loadWeight(String weightFile, String weightType) throws NotConvergedException {
        try {
            if (weightType.startsWith("weightOfInputLayer")) {
                weightOfInputLayer = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("weightOfEnhancedLayer")) {
                weightOfEnhancedLayer = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("weightOfOutputLayer")) {
                weightOfOutputLayer = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("inputOfOutputLayerInv")) {
                inputOfOutputLayerInv = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("minOfEachWindow")) {
                minOfEachWindow = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("distOfMaxAndMin")) {
                distOfMaxAndMin = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("baseInputOfOutputLayerTrain")) {
                baseInputOfOutputLayerTrain = MatrixLoader.loadMatrix(weightFile);
            } else if (weightType.startsWith("otherParameters")) {
                DenseMatrix others = MatrixLoader.loadMatrix(weightFile);
                parameterOfShrink = others.get(0, 0);
                setInputRangeOfScalerOfFeatureMapping(others.get(1, 0), others.get(2, 0));
            } else {
                System.out.println("Invalid weightType, change trainingMode to 0");
                trainingMode = 0;
                return;
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void loadDataSet(String dataFile, String dataType) throws NotConvergedException {
        try {
            if (dataType.startsWith("trainSet")) {
                trainSet = MatrixLoader.loadMatrix(dataFile);
                trainDataNum = trainSet.numRows();
                featureNum = trainSet.numColumns() - 1;
            } else if (dataType.startsWith("testSet")) {
                testSet = MatrixLoader.loadMatrix(dataFile);
                testDataNum = testSet.numRows();
                featureNum = testSet.numColumns() - 1;
            } else {
                System.out.println("Invalid dataset type");
                return;
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void loadDataSet(DenseMatrix data, String dataType) {
        if (dataType.startsWith("trainSet")) {
            trainSet = new DenseMatrix(data);
            trainDataNum = trainSet.numRows();
            featureNum = trainSet.numColumns() - 1;
        } else if (dataType.startsWith("testSet")) {
            testSet = new DenseMatrix(data);
            testDataNum = testSet.numRows();
            featureNum = testSet.numColumns() - 1;
        } else {
            System.out.println("Invalid dataset type");
            return;
        }
    }

    public void train(String TrainingDataFile) throws NotConvergedException {
        try {
            trainSet = MatrixLoader.loadMatrix(TrainingDataFile);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        train();
    }

    public void train(DenseMatrix data) throws NotConvergedException {
        trainSet = new DenseMatrix(data);
        train();
    }

    public void test(DenseMatrix data) throws NotConvergedException {
        testSet = new DenseMatrix(data);
        test();
    }

    public void test(String TestingDataFile) throws NotConvergedException {
        try {
            testSet = MatrixLoader.loadMatrix(TestingDataFile);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        test();
    }

    private void train() throws NotConvergedException {
        if (trainingMode == 2) {
            System.out.println("Model is already trined!");
            alreadyTrained = 1;
            return;
        }
        if (trainingMode == 0) {
            System.out.println("Training in mode:0");
        } else if (trainingMode == 1) {
            System.out.println("Training in mode:1");
        }

        trainDataNum = trainSet.numRows();
        baseDataNum = trainDataNum;
        featureNum = trainSet.numColumns() - 1;
        DenseMatrix inputDataWithBias = new DenseMatrix(trainDataNum, featureNum + 1);
        DenseMatrix trainY = new DenseMatrix(trainDataNum, 1);
        DenseMatrix outputOfFeatureMappingLayer = new DenseMatrix(trainDataNum, N2 * N1);
        DenseMatrix inputOfEnhancedLayerWithBias = new DenseMatrix(trainDataNum, N2 * N1 + 1);
        DenseMatrix outputOfEnhancedLayer = new DenseMatrix(trainDataNum, N3);
        DenseMatrix inputOfOutputLayer = new DenseMatrix(trainDataNum, N2 * N1 + N3);
        DenseMatrix trainPredict = new DenseMatrix(trainDataNum, 1);
        if (trainingMode == 0) {
            weightOfInputLayer = (DenseMatrix) Matrices.random(featureNum + 1, N2 * N1);
        }
        weightOfOutputLayer = new DenseMatrix(N2 * N1 + N3, 1);
        // get train feature and result from trainSet
        for (int i = 0; i < trainDataNum; i++) {
            // result
            trainY.set(i, 0, trainSet.get(i, 0));
            for (int j = 1; j < featureNum + 1; j++) {
                // features
                inputDataWithBias.set(i, j - 1, trainSet.get(i, j));
            }
        }

        long start_time_train = System.currentTimeMillis();

        // add bias of inputData(bias=0.1)
        for (int i = 0; i < trainDataNum; i++) {
            inputDataWithBias.set(i, featureNum, 0.1);
        }

        // outputOfFeatureMappingLayer=inputDataWithBias*inputLayerWeight

        // two method to calculate the outputOfFeatureMappingLayer(articles show that
        // they are not equal)
        // 1. MinMaxScale in whole groups(not recommend)
        /*
         * inputDataWithBias.mult(weightOfInputLayer, outputOfFeatureMappingLayer);
         * if (trainingMode != 2) {
         * scalerOfFeatureMapping = new MinMaxScaler();
         * outputOfFeatureMappingLayer =
         * scalerOfFeatureMapping.fit_transform(outputOfFeatureMappingLayer);
         * }
         */

        // 2. MinMaxScale in different groups(recommend)
        if (trainingMode == 0) {
            minOfEachWindow = new DenseMatrix(N2, N1);
            distOfMaxAndMin = new DenseMatrix(N2, N1);
        }
        // spare auto-encoder is not supported now
        DenseMatrix weightOfEachWindow = new DenseMatrix(featureNum + 1, N1);
        // DenseMatrix FeatureOfEachWindow = new DenseMatrix(trainDataNum, N1);
        DenseMatrix outputOfEachWindow = new DenseMatrix(trainDataNum, N1);

        for (int group = 0; group < N2; group++) {
            // get weightOfEachWindow from weightOfInputLayer
            for (int i = 0; i < featureNum + 1; i++) {
                for (int j = 0; j < N1; j++)
                    weightOfEachWindow.set(i, j, weightOfInputLayer.get(i, N1 * group + j));
            }

            // inputDataWithBias.mult(weightOfEachWindow, FeatureOfEachWindow);
            inputDataWithBias.mult(weightOfEachWindow, outputOfEachWindow);

            DenseMatrix minOfEachWindowTemp = min_y(outputOfEachWindow);
            DenseMatrix distOfMaxAndMinTemp = max_y(outputOfEachWindow);
            distOfMaxAndMinTemp.add(-1, minOfEachWindowTemp);

            if (trainingMode != 2) {
                // add data to training model
                for (int j = 0; j < N1; j++) {
                    minOfEachWindow.set(group, j, minOfEachWindowTemp.get(0, j));
                    distOfMaxAndMin.set(group, j, distOfMaxAndMinTemp.get(0, j));
                }
            } else {
                // get data from trained model
                for (int j = 0; j < N1; j++) {
                    minOfEachWindowTemp.set(0, j, minOfEachWindow.get(group, j));
                    distOfMaxAndMinTemp.set(0, j, distOfMaxAndMin.get(group, j));
                }
            }

            // outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i])
            // /distOfMaxAndMin[i] & merge outputOfEachWindow to outputOfFeatureMappingLayer
            for (int i = 0; i < trainDataNum; i++) {
                for (int j = 0; j < N1; j++) {
                    double temp = (outputOfEachWindow.get(i, j) - minOfEachWindowTemp.get(0, j))
                            / distOfMaxAndMinTemp.get(0, j);
                    outputOfFeatureMappingLayer.set(i, N1 * group + j, temp);
                }

            }
        }

        // copy data
        for (int i = 0; i < trainDataNum; i++) {
            for (int j = 0; j < N2 * N1; j++) {
                inputOfEnhancedLayerWithBias.set(i, j, outputOfFeatureMappingLayer.get(i, j));
            }
        }

        // add bias of inputOfEnhancedLayer(bias=0.1)
        for (int i = 0; i < trainDataNum; i++) {
            inputOfEnhancedLayerWithBias.set(i, N2 * N1, 0.1);
        }

        if (trainingMode == 0) {
            // imperfect code
            // each row are better to be orthogonal according to the paper
            weightOfEnhancedLayer = (DenseMatrix) Matrices.random(N2 * N1 + 1, N3);
        }
        inputOfEnhancedLayerWithBias.mult(weightOfEnhancedLayer, outputOfEnhancedLayer);

        if (trainingMode != 2) {
            // find the max value of outputOfEnhancedLayer(not activated)
            double maxOfOutputOfEnhancedLayer = max(outputOfEnhancedLayer);
            // calculate parameterOfShrink
            parameterOfShrink = s / maxOfOutputOfEnhancedLayer;
        }

        // outputOfEnhancedLayer(activate)=tanh(outputOfEnhancedLayer*parameterOfShrink)
        // tanh or other activate function
        if (activeFunc.startsWith("tanh")) {
            for (int i = 0; i < trainDataNum; i++) {
                for (int j = 0; j < N3; j++) {
                    double temp = Math.tanh(outputOfEnhancedLayer.get(i, j) * parameterOfShrink);
                    outputOfEnhancedLayer.set(i, j, temp);
                }
            }
        } else if (activeFunc.startsWith("sig")) {
            // If you need it ,you can add more func
        }

        // inputOfOutputLayer=[outputOfFeatureMappingLayer,outputOfEnhancedLayer]
        for (int i = 0; i < trainDataNum; i++) {
            for (int j = 0; j < N2 * N1; j++) {
                inputOfOutputLayer.set(i, j, outputOfFeatureMappingLayer.get(i, j));
            }
            for (int j = N2 * N1; j < N2 * N1 + N3; j++) {
                inputOfOutputLayer.set(i, j, outputOfEnhancedLayer.get(i, j - N2 * N1));
            }
        }

        if (trainingMode != 2) {
            // calculate Pseudo-inverse of inputOfOutputLayer
            // A^+ = lim[λ→0] ( λ*I + A*A.T )^−1 * A.T
            long solve_time_start = System.currentTimeMillis();

            Inverse Inverse = new Inverse(inputOfOutputLayer);

            inputOfOutputLayerInv = Inverse.getMPInverse(c);
            long solve_time_end = System.currentTimeMillis();
            solve_time = (solve_time_end - solve_time_start) * 1.0f / 1000;
            // weightOfOutputLayer= inv(inputOfOutputLayer)*trainY
            inputOfOutputLayerInv.mult(trainY, weightOfOutputLayer);
        }

        long end_time_train = System.currentTimeMillis();
        trainTime = (end_time_train - start_time_train) * 1.0f / 1000;

        inputOfOutputLayer.mult(weightOfOutputLayer, trainPredict);

        // evaluate
        if (taskType == 0) {
            // regression
            trainRmse = rmse(trainPredict, trainY);
            trainCC = corrcoef(trainPredict, trainY);

        } else {
            // classification
        }

    }

    private void test() throws NotConvergedException {
        testDataNum = testSet.numRows();
        featureNum = testSet.numColumns() - 1;
        DenseMatrix inputData = new DenseMatrix(testDataNum, featureNum);
        DenseMatrix testY = new DenseMatrix(testDataNum, 1);
        DenseMatrix testPredict = new DenseMatrix(testDataNum, 1);
        // get train feature and result from trainSet
        for (int i = 0; i < testDataNum; i++) {
            // result
            testY.set(i, 0, testSet.get(i, 0));
            for (int j = 1; j < featureNum + 1; j++) {
                // features
                inputData.set(i, j - 1, testSet.get(i, j));
            }
        }
        long start_time_test = System.currentTimeMillis();

        testPredict = forward(inputData);

        long end_time_test = System.currentTimeMillis();
        testTime = (end_time_test - start_time_test) * 1.0f / 1000;
        // evaluation
        // regression
        if (taskType == 0) {
            testCC = corrcoef(testPredict, testY);
            testRmse = rmse(testY, testPredict);
        }
    }

    /**
     * calculate forward progress
     * 
     * @param inputData
     *                  input data matrix
     * @return output of the model
     */
    private DenseMatrix forward(DenseMatrix inputData) {
        int dataNum = inputData.numRows();
        featureNum = inputData.numColumns();
        DenseMatrix inputDataWithBias = new DenseMatrix(dataNum, featureNum + 1);
        DenseMatrix outputOfFeatureMappingLayer = new DenseMatrix(dataNum, N2 * N1);
        DenseMatrix inputOfEnhancedLayerWithBias = new DenseMatrix(dataNum, N2 * N1 + 1);
        DenseMatrix outputOfEnhancedLayer = new DenseMatrix(dataNum, N3);
        DenseMatrix inputOfOutputLayer = new DenseMatrix(dataNum, N2 * N1 + N3);
        DenseMatrix predict = new DenseMatrix(dataNum, 1);

        // copy data
        for (int i = 0; i < dataNum; i++) {
            for (int j = 0; j < featureNum; j++) {
                inputDataWithBias.set(i, j, inputData.get(i, j));
            }
        }
        // add bias of inputData(bias=0.1)
        for (int i = 0; i < dataNum; i++) {
            inputDataWithBias.set(i, featureNum, 0.1);
        }

        // outputOfFeatureMappingLayer=inputDataWithBias*inputLayerWeight
        inputDataWithBias.mult(weightOfInputLayer, outputOfFeatureMappingLayer);
        outputOfFeatureMappingLayer = scalerOfFeatureMapping.transform(outputOfFeatureMappingLayer);

        // two method to calculate the outputOfFeatureMappingLayer(articles show that
        // they are not equal)
        // 1. MinMaxScale in whole groups(not recommend)

        // 2. MinMaxScale in different groups(recommend)
        // spare auto-encoder is not supported now
        DenseMatrix weightOfEachWindow = new DenseMatrix(featureNum + 1, N1);
        // DenseMatrix FeatureOfEachWindow = new DenseMatrix(trainDataNum, N1);
        DenseMatrix outputOfEachWindow = new DenseMatrix(dataNum, N1);

        for (int group = 0; group < N2; group++) {
            // get weightOfEachWindow from weightOfInputLayer
            for (int i = 0; i < featureNum + 1; i++) {
                for (int j = 0; j < N1; j++)
                    weightOfEachWindow.set(i, j, weightOfInputLayer.get(i, N1 * group + j));
            }

            // inputDataWithBias.mult(weightOfEachWindow, FeatureOfEachWindow);
            inputDataWithBias.mult(weightOfEachWindow, outputOfEachWindow);

            DenseMatrix minOfEachWindowTemp = min_y(outputOfEachWindow);
            DenseMatrix distOfMaxAndMinTemp = (DenseMatrix) (max_y(outputOfEachWindow).add(-1, minOfEachWindowTemp));

            // get data from trained model
            for (int j = 0; j < N1; j++) {
                minOfEachWindowTemp.set(0, j, minOfEachWindow.get(group, j));
                distOfMaxAndMinTemp.set(0, j, distOfMaxAndMin.get(group, j));
            }
            // outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i])
            // distOfMaxAndMin[i] & merge outputOfEachWindow to outputOfFeatureMappingLayer
            // for (int i = 0; i < dataNum; i++) {
            // for (int j = 0; j < N1; j++) {
            // double temp = (outputOfEachWindow.get(i, j) - minOfEachWindow.get(group, j))
            // / distOfMaxAndMin.get(group, j);
            // outputOfFeatureMappingLayer.set(i, N1 * group + j, temp);
            // }
            // }

            for (int i = 0; i < dataNum; i++) {
                for (int j = 0; j < N1; j++) {
                    double temp = (outputOfEachWindow.get(i, j) - minOfEachWindow.get(group, j))
                            / distOfMaxAndMin.get(group, j);
                    outputOfFeatureMappingLayer.set(i, N1 * group + j, temp);
                }
            }
        }

        // copy data
        for (int i = 0; i < dataNum; i++) {
            for (int j = 0; j < N2 * N1; j++) {
                inputOfEnhancedLayerWithBias.set(i, j, outputOfFeatureMappingLayer.get(i, j));
            }
        }

        // add bias of inputOfEnhancedLayer(bias=0.1)
        for (int i = 0; i < dataNum; i++) {
            inputOfEnhancedLayerWithBias.set(i, N2 * N1, 0.1);
        }

        // outputOfEnhancedLayer(not
        // activated)=inputOfEnhancedLayerWithBias*weightOfEnhancedLayer
        inputOfEnhancedLayerWithBias.mult(weightOfEnhancedLayer, outputOfEnhancedLayer);

        // outputOfEnhancedLayer(activated)=tanh(outputOfEnhancedLayer*parameterOfShrink)
        // tanh or other activate function
        if (activeFunc.startsWith("tanh")) {
            for (int i = 0; i < dataNum; i++) {
                for (int j = 0; j < N3; j++) {
                    double temp = Math.tanh(outputOfEnhancedLayer.get(i, j) * parameterOfShrink);
                    outputOfEnhancedLayer.set(i, j, temp);
                }
            }
        } else if (activeFunc.startsWith("sig")) {
            // If you need it ,you can add more func
        }

        // inputOfOutputLayer=[outputOfFeatureMappingLayer,outputOfEnhancedLayer]
        for (int i = 0; i < dataNum; i++) {
            for (int j = 0; j < N2 * N1; j++) {
                inputOfOutputLayer.set(i, j, outputOfFeatureMappingLayer.get(i, j));
            }
            for (int j = N2 * N1; j < N2 * N1 + N3; j++) {
                inputOfOutputLayer.set(i, j, outputOfEnhancedLayer.get(i, j - N2 * N1));
            }
        }
        alreadyTrained = 1;
        inputOfOutputLayer.mult(weightOfOutputLayer, predict);
        return predict;
    }

    /**
     * use trained model to predict
     * 
     * @param filename
     *                 input data file path
     * @return
     *         predict result of the model
     */
    public DenseMatrix predict(String filename) {
        try {
            DenseMatrix input = MatrixLoader.loadMatrix(filename);
            return forward(input);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }
    }

    /**
     * use trained model to predict
     * 
     * @param input
     *              input data matrix
     * @return
     *         predict result of the model
     */
    public DenseMatrix predict(DenseMatrix input) {
        return forward(input);
    }

    private double rmse(DenseMatrix y1, DenseMatrix y2) {
        int length = y1.numRows();
        double error = 0;
        for (int i = 0; i < length; i++) {
            error += Math.pow((y1.get(i, 0) - y2.get(i, 0)), 2);
        }
        return Math.sqrt(error / length);
    }

    private double expectation(DenseMatrix data) {
        // E(X)
        int len = data.numRows();
        double mean = 0;
        for (int i = 0; i < len; i++) {
            mean += data.get(i, 0);
        }
        return mean / len;
    }

    private double expectation_mul(DenseMatrix data1, DenseMatrix data2) {
        // E(XY)
        int len = data1.numRows();
        double mean = 0;
        for (int i = 0; i < len; i++) {
            mean += data1.get(i, 0) * data2.get(i, 0);
        }
        return mean / len;
    }

    private double corrcoef(DenseMatrix data1, DenseMatrix data2) {
        // cc(x,y)=(EX(xy)-EX(x)EX(y))/(sqrt(EX(x^2)-EX(x)^2)*sqrt(EX(y^2)-EX(y)^2))
        double a = expectation_mul(data1, data2) - expectation(data1) * expectation(data2);
        double b = Math.sqrt(expectation_mul(data1, data1) - Math.pow(expectation(data1), 2))
                * Math.sqrt(expectation_mul(data2, data2) - Math.pow(expectation(data2), 2));
        return a / b;
    }

    private double max(DenseMatrix x) {
        double result = -1e13;
        for (int i = 0; i < x.numRows(); i++) {
            for (int j = 0; j < x.numColumns(); j++) {
                if (x.get(i, j) > result)
                    result = x.get(i, j);
            }
        }
        return result;
    }

    private double min(DenseMatrix x) {
        double result = 1e13;
        for (int i = 0; i < x.numRows(); i++) {
            for (int j = 0; j < x.numColumns(); j++) {
                if (x.get(i, j) < result)
                    result = x.get(i, j);
            }
        }
        return result;
    }

    private DenseMatrix max_x(DenseMatrix x) {
        DenseMatrix result_x = new DenseMatrix(x.numRows(), 1);
        for (int i = 0; i < x.numRows(); i++) {
            double result = -1e13;
            for (int j = 0; j < x.numColumns(); j++) {
                if (x.get(i, j) > result) {
                    result = x.get(i, j);
                }
            }
            result_x.set(i, 0, result);
        }
        return result_x;
    }

    private DenseMatrix max_y(DenseMatrix x) {
        DenseMatrix result_y = new DenseMatrix(1, x.numColumns());
        for (int j = 0; j < x.numColumns(); j++) {
            double result = -1e13;
            for (int i = 0; i < x.numRows(); i++) {
                if (x.get(i, j) > result) {
                    result = x.get(i, j);
                    result_y.set(0, j, result);
                }
            }
        }
        return result_y;
    }

    private DenseMatrix min_y(DenseMatrix x) {
        DenseMatrix result_y = new DenseMatrix(1, x.numColumns());
        for (int j = 0; j < x.numColumns(); j++) {
            double result = 1e13;
            for (int i = 0; i < x.numRows(); i++) {
                if (x.get(i, j) < result) {
                    result = x.get(i, j);
                    result_y.set(0, j, result);
                }

            }
        }
        return result_y;
    }

    private DenseMatrix min_x(DenseMatrix x) {
        DenseMatrix result_x = new DenseMatrix(x.numRows(), 1);

        for (int i = 0; i < x.numRows(); i++) {
            double result = 1e13;
            for (int j = 0; j < x.numColumns(); j++) {
                if (x.get(i, j) < result) {
                    result = x.get(i, j);
                }
            }
            result_x.set(i, 0, result);
        }
        return result_x;
    }

    private DenseMatrix mapminmax_onezero(DenseMatrix data) {
        for (int i = 0; i < data.numRows(); i++) {
            double xmin = 1e13, xmax = -1e13;
            for (int m = 0; m < data.numColumns(); m++) {
                if (data.get(i, m) > xmax)
                    xmax = data.get(i, m);
                if (data.get(i, m) < xmin)
                    xmin = data.get(i, m);
            }
            for (int j = 0; j < data.numColumns(); j++) {
                if (xmax == xmin)
                    data.set(i, j, xmin);
                else {
                    double temp = (data.get(i, j) - xmin) / (xmax - xmin);
                    data.set(i, j, temp);
                }
            }
        }
        return data;
    }

    public void incrementalTrainAddData(String incrementDataFile) throws NotConvergedException {
        try {
            DenseMatrix incrementSet = MatrixLoader.loadMatrix(incrementDataFile);
            incrementalTrainAddData(incrementSet);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public void incrementalTrainAddData(DenseMatrix incrementalData) throws NotConvergedException {
        if (alreadyTrained == 0) {
            System.out.println("model must be trained if you want to use incremental training!");
            return;
        }
        int incrementDataNum = incrementalData.numRows();
        if (incrementDataNum != baseDataNum) {
            System.out.println("the incremental data number must be the same as original train data");
            return;
        }
        featureNum = incrementalData.numColumns() - 1;
        DenseMatrix inputData;
        DenseMatrix trainY = new DenseMatrix(trainDataNum + incrementDataNum, 1);
        DenseMatrix inputDataWithBias = new DenseMatrix(incrementDataNum, featureNum + 1);
        DenseMatrix weightOfEachWindow = new DenseMatrix(featureNum + 1, N1);
        DenseMatrix outputOfEachWindow = new DenseMatrix(incrementDataNum, N1);
        DenseMatrix outputOfFeatureMappingLayer = new DenseMatrix(incrementDataNum, N2 * N1);
        DenseMatrix inputOfEnhancedLayerWithBiasIncrement = new DenseMatrix(incrementDataNum, N2 * N1 + 1);
        DenseMatrix outputOfEnhancedLayerIncrement = new DenseMatrix(incrementDataNum, N3);
        DenseMatrix inputOfOutputLayerIncrement = new DenseMatrix(incrementDataNum, N2 * N1 + N3);
        DenseMatrix trainPredict = new DenseMatrix(trainDataNum + incrementDataNum, 1);
        // get train feature from trainSet
        for (int i = 0; i < incrementDataNum; i++) {
            for (int j = 1; j < featureNum + 1; j++) {
                // features
                inputDataWithBias.set(i, j - 1, incrementalData.get(i, j));
            }
        }
        // add bias of inputData(bias=0.1)
        for (int i = 0; i < incrementDataNum; i++) {
            inputDataWithBias.set(i, featureNum, 0.1);
        }

        for (int group = 0; group < N2; group++) {
            // get weightOfEachWindow from weightOfInputLayer
            for (int i = 0; i < featureNum + 1; i++) {
                for (int j = 0; j < N1; j++)
                    weightOfEachWindow.set(i, j, weightOfInputLayer.get(i, N1 * group + j));
            }

            // inputDataWithBias.mult(weightOfEachWindow, FeatureOfEachWindow);
            inputDataWithBias.mult(weightOfEachWindow, outputOfEachWindow);

            outputOfEachWindow = mapminmax_onezero(outputOfEachWindow);
            for (int i = 0; i < incrementDataNum; i++) {
                for (int j = 0; j < N1; j++) {
                    outputOfFeatureMappingLayer.set(i, N1 * group + j, outputOfEachWindow.get(i, j));
                }
            }
        }

        // copy data
        for (int i = 0; i < incrementDataNum; i++) {
            for (int j = 0; j < N1 * N2; j++) {
                inputOfEnhancedLayerWithBiasIncrement.set(i, j, outputOfFeatureMappingLayer.get(i, j));
            }
        }
        // add bias of inputOfEnhancedLayer(bias=0.1)
        for (int i = 0; i < incrementDataNum; i++) {
            inputOfEnhancedLayerWithBiasIncrement.set(i, N2 * N1, 0.1);
        }

        inputOfEnhancedLayerWithBiasIncrement.mult(weightOfEnhancedLayer, outputOfEnhancedLayerIncrement);
        // outputOfEnhancedLayer(activate)=tanh(outputOfEnhancedLayer*parameterOfShrink)
        // tanh or other activate function
        if (activeFunc.startsWith("tanh")) {
            for (int i = 0; i < incrementDataNum; i++) {
                for (int j = 0; j < N3; j++) {
                    double temp = Math.tanh(outputOfEnhancedLayerIncrement.get(i, j) * parameterOfShrink);
                    outputOfEnhancedLayerIncrement.set(i, j, temp);
                }
            }
        } else if (activeFunc.startsWith("sig")) {
            // If you need it ,you can add more func
        }

        // inputOfOutputLayer=[outputOfFeatureMappingLayer,outputOfEnhancedLayer]
        for (int i = 0; i < incrementDataNum; i++) {
            for (int j = 0; j < N2 * N1; j++) {
                inputOfOutputLayerIncrement.set(i, j, outputOfFeatureMappingLayer.get(i, j));
            }
            for (int j = N2 * N1; j < N2 * N1 + N3; j++) {
                inputOfOutputLayerIncrement.set(i, j, outputOfEnhancedLayerIncrement.get(i, j - N2 * N1));
            }
        }

        // calculate Pseudo-inverse of inputOfOutputLayer
        // A^+ = lim[λ→0] ( λ*I + A*A.T )^−1 * A.T
        Inverse Inverse = new Inverse(inputOfOutputLayerIncrement);
        DenseMatrix inputOfOutputLayerInvIncrement = Inverse.getMPInverse(c);
        // update inputOfOutputLayerInv
        DenseMatrix inputOfOutputLayerInvOld = new DenseMatrix(inputOfOutputLayerInv);
        inputOfOutputLayerInv = new DenseMatrix(N2 * N1 + N3, trainDataNum + incrementDataNum);
        for (int i = 0; i < N2 * N1 + N3; i++) {
            for (int j = 0; j < trainDataNum; j++) {
                inputOfOutputLayerInv.set(i, j, inputOfOutputLayerInvOld.get(i, j));
            }
            for (int j = trainDataNum; j < trainDataNum + incrementDataNum; j++) {
                inputOfOutputLayerInv.set(i, j, inputOfOutputLayerInvIncrement.get(i, j - trainDataNum));
            }
        }

        // update trainSet
        DenseMatrix trainSetOld = new DenseMatrix(trainSet);
        trainSet = new DenseMatrix(trainDataNum + incrementDataNum, featureNum + 1);
        for (int i = 0; i < trainDataNum; i++) {
            for (int j = 0; j < featureNum + 1; j++) {
                trainSet.set(i, j, trainSetOld.get(i, j));
            }
        }
        for (int i = trainDataNum; i < trainDataNum + incrementDataNum; i++) {
            for (int j = 0; j < featureNum + 1; j++) {
                trainSet.set(i, j, incrementalData.get(i - trainDataNum, j));
            }
        }

        trainDataNum += incrementDataNum;

        // update weightOfOutputLayer
        // get result from new trainSet
        for (int i = 0; i < trainDataNum; i++) {
            trainY.set(i, 0, trainSet.get(i, 0));
        }
        inputOfOutputLayerInv.mult(trainY, weightOfOutputLayer);
        // evaluate
        // get train feature from new trainSet
        inputData = new DenseMatrix(trainDataNum, featureNum);
        for (int i = 0; i < trainDataNum; i++) {
            for (int j = 1; j < featureNum + 1; j++) {
                // features
                inputData.set(i, j - 1, trainSet.get(i, j));
            }
        }

        DenseMatrix inputOfOutputLayerOld = new DenseMatrix(baseInputOfOutputLayerTrain);
        baseInputOfOutputLayerTrain = new DenseMatrix(
                inputOfOutputLayerOld.numRows() + inputOfOutputLayerIncrement.numRows(), N2 * N1 + N3);
        for (int i = 0; i < inputOfOutputLayerOld.numRows(); i++) {
            for (int j = 0; j < N2 * N1 + N3; j++) {
                baseInputOfOutputLayerTrain.set(i, j, inputOfOutputLayerOld.get(i, j));
            }
        }
        for (int i = inputOfOutputLayerOld.numRows(); i < inputOfOutputLayerOld.numRows()
                + inputOfOutputLayerIncrement.numRows(); i++) {
            for (int j = 0; j < N2 * N1 + N3; j++) {
                baseInputOfOutputLayerTrain.set(i, j,
                        inputOfOutputLayerIncrement.get(i - inputOfOutputLayerOld.numRows(), j));
            }
        }

        baseInputOfOutputLayerTrain.mult(weightOfOutputLayer, trainPredict);
        if (taskType == 0) {
            // regression
            trainRmse = rmse(trainPredict, trainY);
            trainCC = corrcoef(trainPredict, trainY);

        } else {
            // classification
        }
    }

    /**
     * set training mode.
     * 
     * @param mode
     *             <p>
     *             mode 0: all parameters generated by this class.(default)
     *             <p>
     *             mode 1: weightOfOutputLayer generated by this class.
     *             <p>
     *             mode 2: all parameters loaded from file.
     */
    public void setTrainMode(int mode) {
        if (mode != 0 && mode != 1 && mode != 2)
            trainingMode = 0;
        else
            trainingMode = mode;
        if (trainingMode == 2)
            alreadyTrained = 1;
        else
            alreadyTrained = 0;
    }

    private void setInputRangeOfScalerOfFeatureMapping(double min, double max) {
        if (scalerOfFeatureMapping == null)
            scalerOfFeatureMapping = new MinMaxScaler();
        scalerOfFeatureMapping.forceSetInputDataRange(min, max);
    }

    public int getTrainMode() {
        return trainingMode;
    }

    public double getTrainRmse() {
        return trainRmse;
    }

    public double getTrainCC() {
        return trainCC;
    }

    public double getTestRmse() {
        return testRmse;
    }

    public double getTestCC() {
        return testCC;
    }

    public double getTrainTime() {
        return trainTime;
    }

    public double getTestTime() {
        return testTime;
    }

    public double getSolveTime() {
        return solve_time;
    }

    public void setOriginalTrainDataNum(int num) {
        baseDataNum = num;
    }
}
