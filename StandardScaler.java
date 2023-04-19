import java.io.IOException;

import no.uib.cipr.matrix.DenseMatrix;

public class StandardScaler {
    private int method;
    private double[] e;
    private double[] d;
    private DataLoader MatrixLoader;

    public StandardScaler() {
        method = 0;
        MatrixLoader = new DataLoader();
    }

    /**
     * construct a StandardScaler
     * 
     * @param axis
     *             <p>
     *             0: normalization for elements in rows
     *             <p>
     *             1: normalization for elements in columns
     * @return output of the model
     */
    public StandardScaler(int axis) {
        method = axis;
        if (axis != 0 && axis != 1)
            method = 0;
        MatrixLoader = new DataLoader();
    }

    public void fit(String fileName) {
        try {
            DenseMatrix temp = MatrixLoader.loadMatrix(fileName);
            fit(temp);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public DenseMatrix transform(String fileName) {
        try {
            DenseMatrix temp = MatrixLoader.loadMatrix(fileName);
            return transform(temp);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }
    }

    public DenseMatrix fit_transform(String fileName) {
        try {
            DenseMatrix temp = MatrixLoader.loadMatrix(fileName);
            fit(temp);
            return transform(temp);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }
    }

    public DenseMatrix inverse_transform(DenseMatrix data) {
        DenseMatrix out = new DenseMatrix(data);
        if (method == 0) {
            for (int i = 0; i < data.numRows(); i++) {
                for (int j = 0; j < data.numColumns(); j++) {
                    double temp = data.get(i, j) * d[i] + e[i];
                    out.set(i, j, temp);
                }
            }
        } else {
            for (int j = 0; j < data.numColumns(); j++) {
                for (int i = 0; i < data.numRows(); i++) {
                    double temp = data.get(i, j) * d[j] + e[j];
                    out.set(i, j, temp);
                }
            }
        }
        return out;
    }

    public void setBias(String fileName) {
        try {
            DenseMatrix temp = MatrixLoader.loadMatrix(fileName);
            setBias(temp);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void setWeight(String fileName) {
        try {
            DenseMatrix temp = MatrixLoader.loadMatrix(fileName);
            setWeight(temp);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void fit(DenseMatrix data) {

        if (method == 0) {
            e = new double[data.numRows()];
            d = new double[data.numRows()];
        } else {
            e = new double[data.numColumns()];
            d = new double[data.numColumns()];
        }

        if (method == 0) {
            for (int i = 0; i < data.numRows(); i++) {
                DenseMatrix temp = new DenseMatrix(1, data.numColumns());
                for (int j = 0; j < data.numColumns(); j++) {
                    temp.set(0, j, data.get(i, j));
                }
                e[i] = expectation(temp);
                d[i] = Math.sqrt(variance(temp));
            }
        } else {
            for (int j = 0; j < data.numColumns(); j++) {
                DenseMatrix temp = new DenseMatrix(1, data.numRows());
                for (int i = 0; i < data.numRows(); i++) {
                    temp.set(0, i, data.get(i, j));
                }
                e[j] = expectation(temp);
                d[j] = Math.sqrt(variance(temp));
            }
        }

    }

    public DenseMatrix transform(DenseMatrix data) {
        DenseMatrix out = new DenseMatrix(data);
        if (method == 0) {
            for (int i = 0; i < data.numRows(); i++) {
                for (int j = 0; j < data.numColumns(); j++) {
                    double temp = (data.get(i, j) - e[i]) / d[i];
                    out.set(i, j, temp);
                }
            }
        } else {
            for (int j = 0; j < data.numColumns(); j++) {
                for (int i = 0; i < data.numRows(); i++) {
                    double temp = (data.get(i, j) - e[j]) / d[j];
                    out.set(i, j, temp);
                }
            }
        }
        return out;
    }

    public DenseMatrix fit_transform(DenseMatrix data) {
        fit(data);
        return transform(data);
    }

    public void setBias(DenseMatrix bias) {
        e = new double[bias.numRows()];
        for (int i = 0; i < bias.numRows(); i++)
            // var
            e[i] = bias.get(i, 0);
    }

    public void setWeight(DenseMatrix weight) {
        d = new double[weight.numRows()];
        for (int i = 0; i < weight.numRows(); i++)
            d[i] = Math.sqrt(weight.get(i, 0));
    }

    private double expectation(DenseMatrix data) {
        // E(X)
        int len = data.numColumns();

        double mean = 0;
        for (int i = 0; i < len; i++) {
            mean += data.get(0, i);
        }
        return mean / len;
    }

    private double expectation_mul(DenseMatrix data1, DenseMatrix data2) {
        // E(XY)
        int len = data1.numColumns();

        double mean = 0;
        for (int i = 0; i < len; i++) {
            mean += data1.get(0, i) * data2.get(0, i);
        }
        return mean / len;
    }

    public double variance(DenseMatrix data) {
        return expectation_mul(data, data) - Math.pow(expectation(data), 2);
    }
}
