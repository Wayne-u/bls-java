import java.io.IOException;

import javax.swing.text.Segment;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;

public class main {

	/**
	 * @param args
	 * @throws NotConvergedException
	 * @throws IOException
	 */
	public static void main(String[] args) throws NotConvergedException, IOException {
		double lumda = Math.pow(2, -34);
		// double lumda = 0.000001;
		// 实例化一个bls
		bls bls1 = new bls(0, 19, 20, 7000, 0.4, lumda, "tanh");

		// 三种模式： mode0全在线训练 mode1半在线训练 mode2直接使用全部训练好的参数
		bls1.setTrainMode(2);

		// 载入训练好的参数
		loadAllWeight(bls1);

		// 使用原始直采脑电数据集训练
		// useOriginalData(bls1);

		// 不训练直接测试
		useOriginalDataToTest(bls1);
		// 使用python预处理好的数据集训练
		// useProcessedData(bls1);

		// 使用增量学习
		// useIncrementalTrain(bls1, 2000);

	}

	private static void loadAllWeight(bls model) throws NotConvergedException {
		if (model.getTrainMode() == 0)
			return;
		if (model.getTrainMode() != 0) {
			model.loadWeight("weight/inputWeight_o.txt", "weightOfInputLayer");
			model.loadWeight("weight/weightOfEnhanceLayer_o.txt", "weightOfEnhancedLayer");
			if (model.getTrainMode() == 2) {
				model.loadWeight("weight/OutputWeight_o.txt", "weightOfOutputLayer");
				// model.loadWeight("weight/pinvOfInput_o.txt", "inputOfOutputLayerInv");
				model.loadWeight("weight/parameterOfShrink_o.txt", "otherParameters");
				model.loadWeight("weight/minOfEachWindow_o.txt", "minOfEachWindow");
				model.loadWeight("weight/distOfMaxAndMin_o.txt", "distOfMaxAndMin");
				// model.loadWeight("weight/inputOfOutputLayer_o.txt",
				// "baseInputOfOutputLayerTrain");
			}
		}

	}

	private static void useOriginalData(bls model) throws NotConvergedException, IOException {
		DataLoader matrixLoader = new DataLoader();
		DenseMatrix data_train = matrixLoader.loadMatrix("dataset/eeg_train_origin_o.txt");
		DenseMatrix data_test = matrixLoader.loadMatrix("dataset/eeg_test_origin_o.txt");

		DenseMatrix feature_train = new DenseMatrix(data_train.numRows(), data_train.numColumns() - 1);
		DenseMatrix feature_test = new DenseMatrix(data_test.numRows(), data_test.numColumns() - 1);
		DenseMatrix label_train = new DenseMatrix(data_train.numRows(), 1);
		DenseMatrix label_test = new DenseMatrix(data_test.numRows(), 1);
		StandardScaler scaler1 = new StandardScaler(0);
		StandardScaler scaler2 = new StandardScaler(1);
		DownSampleTool tool1 = new DownSampleTool(4);

		for (int i = 0; i < data_train.numRows(); i++) {
			for (int j = 1; j < data_train.numColumns(); j++) {
				feature_train.set(i, j - 1, data_train.get(i, j));
			}
		}
		for (int i = 0; i < data_test.numRows(); i++) {
			for (int j = 1; j < data_test.numColumns(); j++) {
				feature_test.set(i, j - 1, data_test.get(i, j));
			}
		}
		for (int i = 0; i < data_train.numRows(); i++) {
			label_train.set(i, 0, data_train.get(i, 0));
		}
		for (int i = 0; i < data_test.numRows(); i++) {
			label_test.set(i, 0, data_test.get(i, 0));
		}
		// feature_train = tool1.averageDownSample(feature_train);
		// feature_test = tool1.averageDownSample(feature_test);
		feature_train = scaler1.fit_transform(feature_train);
		feature_test = scaler1.fit_transform(feature_test);
		label_train = scaler2.fit_transform(label_train);
		label_test = scaler2.transform(label_test);

		DenseMatrix data_train1 = new DenseMatrix(feature_train.numRows(), feature_train.numColumns() + 1);
		DenseMatrix data_test1 = new DenseMatrix(feature_test.numRows(), feature_test.numColumns() + 1);

		for (int i = 0; i < data_train1.numRows(); i++) {
			data_train1.set(i, 0, label_train.get(i, 0));
			for (int j = 1; j < data_train1.numColumns(); j++) {
				data_train1.set(i, j, feature_train.get(i, j - 1));
			}
		}
		for (int i = 0; i < data_test1.numRows(); i++) {
			data_test1.set(i, 0, label_test.get(i, 0));
			for (int j = 1; j < data_test1.numColumns(); j++) {
				data_test1.set(i, j, feature_test.get(i, j - 1));
			}
		}

		// 使用mode2时别执行bls1.train()无效
		model.train(data_train1);
		model.test(data_test1);

		System.out.println("Training time:" + model.getTrainTime());
		System.out.println("solve time:" + model.getSolveTime());
		System.out.println("Training rmse:" + model.getTrainRmse());
		System.out.println("Training CC:" + model.getTrainCC());
		System.out.println("Testing time:" + model.getTestTime());
		System.out.println("Testing Rmse:" + model.getTestRmse());
		System.out.println("Testing CC:" + model.getTestCC());

		// 使用训练好的模型推理
		/*
		 * DenseMatrix output = model.predict("dataset/test_without_label.txt");
		 * output = scaler2.inverse_transform(output);
		 * System.out.println("测试集预测结果（反标准化后）：");
		 * for (int i = 0; i < output.numRows(); i++)
		 * System.out.println(output.get(i, 0));
		 */
	}

	private static void useProcessedData(bls model) throws NotConvergedException {
		// 使用mode2时别执行bls1.train()无效
		model.train("dataset/eeg_train_o.txt");
		model.test("dataset/eeg_test_o.txt");

		System.out.println("Training time:" + model.getTrainTime());
		System.out.println("solve time:" + model.getSolveTime());
		System.out.println("Training rmse:" + model.getTrainRmse());
		System.out.println("Training CC:" + model.getTrainCC());
		System.out.println("Testing time:" + model.getTestTime());
		System.out.println("Testing Rmse:" + model.getTestRmse());
		System.out.println("Testing CC:" + model.getTestCC());

		// 使用训练好的模型推理
		/*
		 * DenseMatrix output = model.predict("dataset/test_without_label.txt");
		 * System.out.println("测试集预测结果：");
		 * for (int i = 0; i < output.numRows(); i++)
		 * System.out.println(output.get(i, 0));
		 */
	}

	private static void useIncrementalTrain(bls model, int step) throws IOException, NotConvergedException {
		DataLoader matrixLoader = new DataLoader();
		DenseMatrix data_train = matrixLoader.loadMatrix("dataset/eeg_train_origin_o.txt");
		DenseMatrix data_test = matrixLoader.loadMatrix("dataset/eeg_test_origin_o.txt");
		DenseMatrix data_val = matrixLoader.loadMatrix("dataset/eeg_val_origin_o.txt");
		DenseMatrix feature_train = new DenseMatrix(data_train.numRows(), data_train.numColumns() - 1);
		DenseMatrix feature_test = new DenseMatrix(data_test.numRows(), data_test.numColumns() - 1);
		DenseMatrix feature_val = new DenseMatrix(data_val.numRows(), data_val.numColumns() - 1);
		DenseMatrix label_train = new DenseMatrix(data_train.numRows(), 1);
		DenseMatrix label_test = new DenseMatrix(data_test.numRows(), 1);
		DenseMatrix label_val = new DenseMatrix(data_val.numRows(), 1);
		StandardScaler scaler1 = new StandardScaler(0);
		StandardScaler scaler2 = new StandardScaler(1);
		DownSampleTool tool1 = new DownSampleTool(4);

		for (int i = 0; i < data_train.numRows(); i++) {
			for (int j = 1; j < data_train.numColumns(); j++) {
				feature_train.set(i, j - 1, data_train.get(i, j));
			}
		}
		for (int i = 0; i < data_test.numRows(); i++) {
			for (int j = 1; j < data_test.numColumns(); j++) {
				feature_test.set(i, j - 1, data_test.get(i, j));
			}
		}
		for (int i = 0; i < data_val.numRows(); i++) {
			for (int j = 1; j < data_val.numColumns(); j++) {
				feature_val.set(i, j - 1, data_val.get(i, j));
			}
		}

		for (int i = 0; i < data_train.numRows(); i++) {
			label_train.set(i, 0, data_train.get(i, 0));
		}
		for (int i = 0; i < data_test.numRows(); i++) {
			label_test.set(i, 0, data_test.get(i, 0));
		}
		for (int i = 0; i < data_val.numRows(); i++) {
			label_val.set(i, 0, data_val.get(i, 0));
		}

		// feature_train = tool1.averageDownSample(feature_train);
		// feature_test = tool1.averageDownSample(feature_test);
		feature_train = scaler1.fit_transform(feature_train);
		feature_test = scaler1.fit_transform(feature_test);
		feature_val = scaler1.fit_transform(feature_val);
		label_train = scaler2.fit_transform(label_train);
		label_test = scaler2.transform(label_test);
		label_val = scaler2.transform(label_val);
		DenseMatrix data_train1 = new DenseMatrix(step, feature_train.numColumns() + 1);
		DenseMatrix data_test1 = new DenseMatrix(feature_test.numRows(), feature_test.numColumns() + 1);
		DenseMatrix data_val1 = new DenseMatrix(feature_val.numRows(), feature_val.numColumns() + 1);

		// combine test set and train set
		DenseMatrix data_train_full = new DenseMatrix(data_train.numRows() + data_test.numRows(),
				feature_train.numColumns() + 1);
		for (int i = 0; i < data_train.numRows(); i++) {
			data_train_full.set(i, 0, label_train.get(i, 0));
			for (int j = 1; j < data_train.numColumns(); j++) {
				data_train_full.set(i, j, feature_train.get(i, j - 1));
			}
		}
		for (int i = data_train.numRows(); i < data_train.numRows() + data_test.numRows(); i++) {
			data_train_full.set(i, 0, label_test.get(i - data_train.numRows(), 0));
			for (int j = 1; j < data_train.numColumns(); j++) {
				data_train_full.set(i, j, feature_test.get(i - data_train.numRows(), j - 1));
			}
		}

		// combine data_test
		for (int i = 0; i < data_test1.numRows(); i++) {
			data_test1.set(i, 0, label_test.get(i, 0));
			for (int j = 1; j < data_test1.numColumns(); j++) {
				data_test1.set(i, j, feature_test.get(i, j - 1));
			}
		}
		// combine data_val
		for (int i = 0; i < data_val1.numRows(); i++) {
			data_val1.set(i, 0, label_val.get(i, 0));
			for (int j = 1; j < data_val1.numColumns(); j++) {
				data_val1.set(i, j, feature_val.get(i, j - 1));
			}
		}
		// 原始训练数据
		for (int i = 0; i < step; i++) {
			for (int j = 0; j < data_train_full.numColumns(); j++) {
				data_train1.set(i, j, data_train_full.get(i, j));
			}
		}
		// 基准模型测试

		model.test(data_val1);
		System.out.println("base line Testing Rmse:" + model.getTestRmse());
		System.out.println("base line Testing CC:" + model.getTestCC());

		model.loadDataSet(data_val1, "testSet");
		model.loadDataSet(data_train1, "trainSet");
		model.setOriginalTrainDataNum(step);

		// 分批次进行增量学习step=500
		for (int m = 1; m < data_train_full.numRows() / step; m++) {
			for (int i = 0; i < step; i++) {
				for (int j = 0; j < data_train_full.numColumns(); j++) {
					data_train1.set(i, j, data_train_full.get(i + m * step, j));
				}
			}
			model.incrementalTrainAddData(data_train1);
			model.test(data_val1);
			System.out.println("step:2000 epoch:" + m);
			System.out.println("Training rmse:" + model.getTrainRmse());
			System.out.println("Training CC:" + model.getTrainCC());
			System.out.println("Testing time:" + model.getTestTime());
			System.out.println("Testing Rmse:" + model.getTestRmse());
			System.out.println("Testing CC:" + model.getTestCC());
		}
	}

	private static void useOriginalDataToTest(bls model) throws NotConvergedException, IOException {
		DataLoader matrixLoader = new DataLoader();
		// DenseMatrix data_test = segmentCutLoad();
		DenseMatrix data_test = matrixLoader.loadMatrix("dataset/eeg_val_origin_o.txt");
		DenseMatrix mean = matrixLoader.loadMatrix("weight/trainLabelMean.txt");
		DenseMatrix var = matrixLoader.loadMatrix("weight/trainLabelVar.txt");
		DenseMatrix feature_test = new DenseMatrix(data_test.numRows(), data_test.numColumns() - 1);
		DenseMatrix label_test = new DenseMatrix(data_test.numRows(), 1);
		StandardScaler scaler1 = new StandardScaler(0);
		StandardScaler scaler2 = new StandardScaler(1);
		// DownSampleTool tool1 = new DownSampleTool(4);

		scaler2.setBias(mean);
		scaler2.setWeight(var);
		for (int i = 0; i < data_test.numRows(); i++) {
			for (int j = 1; j < data_test.numColumns(); j++) {
				feature_test.set(i, j - 1, data_test.get(i, j));
			}
		}
		for (int i = 0; i < data_test.numRows(); i++) {
			label_test.set(i, 0, data_test.get(i, 0));
		}
		// feature_train = tool1.averageDownSample(feature_train);
		// feature_test = tool1.averageDownSample(feature_test);
		feature_test = scaler1.fit_transform(feature_test);
		label_test = scaler2.transform(label_test);

		DenseMatrix data_test1 = new DenseMatrix(feature_test.numRows(), feature_test.numColumns() + 1);

		for (int i = 0; i < data_test1.numRows(); i++) {
			data_test1.set(i, 0, label_test.get(i, 0));
			for (int j = 1; j < data_test1.numColumns(); j++) {
				data_test1.set(i, j, feature_test.get(i, j - 1));
			}
		}

		model.test(data_test1);

		System.out.println("Testing time:" + model.getTestTime());
		System.out.println("Testing Rmse:" + model.getTestRmse());
		System.out.println("Testing CC:" + model.getTestCC());

		// 使用训练好的模型推理
		/*
		 * DenseMatrix output = model.predict("dataset/test_without_label.txt");
		 * output = scaler2.inverse_transform(output);
		 * System.out.println("测试集预测结果（反标准化后）：");
		 * for (int i = 0; i < output.numRows(); i++)
		 * System.out.println(output.get(i, 0));
		 */

	}

	private static DenseMatrix segmentCutLoad() throws IOException {
		DataLoader matrixLoader = new DataLoader();
		DenseMatrix data_test = new DenseMatrix(1695, 2501);
		int g = 1695 / 5;
		for (int i = 0; i < 5; i++) {
			String path = "dataset/eeg_val_origin_o" + (i + 1) + ".txt";
			DenseMatrix data_temp = matrixLoader.loadMatrix(path);
			for (int m = 0; m < g; m++) {
				for (int n = 0; n < 2501; n++) {
					data_test.set(g * i + m, n, data_temp.get(m, n));
				}
			}
		}
		return data_test;

	}
}
