import no.uib.cipr.matrix.DenseMatrix;

public class MinMaxScaler {
    private double minScaler;
    private double maxScaler;
    private double dataMin;
    private double dataMax;

    public MinMaxScaler() {
        minScaler = 0;
        maxScaler = 1;
        dataMax = -1e13;
        dataMin = 1e13;
    }

    public MinMaxScaler(double userMin, double userMax) {
        minScaler = userMin;
        maxScaler = userMax;
        dataMax = -1e13;
        dataMin = 1e13;
    }

    public void forceSetInputDataRange(double min, double max) {
        dataMax = max;
        dataMin = min;
    }

    public void fit(DenseMatrix input) {
        for (int i = 0; i < input.numRows(); i++) {
            for (int j = 0; j < input.numColumns(); j++) {
                if (input.get(i, j) > dataMax)
                    dataMax = input.get(i, j);
                if (input.get(i, j) < dataMin)
                    dataMin = input.get(i, j);
            }
        }
    }

    public DenseMatrix transform(DenseMatrix input) {
        DenseMatrix output = new DenseMatrix(input.numRows(), input.numColumns());
        for (int i = 0; i < input.numRows(); i++) {
            for (int j = 0; j < input.numColumns(); j++) {
                double temp = (maxScaler - minScaler) / (dataMax - dataMin) * (input.get(i, j) - dataMin) + minScaler;
                output.set(i, j, temp);
            }
        }
        return output;
    }

    public DenseMatrix fit_transform(DenseMatrix input) {
        fit(input);
        return transform(input);
    }

}
