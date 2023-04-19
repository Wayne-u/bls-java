import no.uib.cipr.matrix.DenseMatrix;

public class DownSampleTool {
    private int divTimes;
    public DownSampleTool(int div) {
        divTimes = div; 
    }
    public DownSampleTool() {
        divTimes = 4; 
    }

    public DenseMatrix averageDownSample(DenseMatrix data) {
        DenseMatrix dataNew = new DenseMatrix(data.numRows(), (int) (data.numColumns() / divTimes));
        for (int i = 0; i < data.numRows(); i++) {
            for (int j = 0; j < data.numColumns(); j += divTimes) {
                int counter = 0;
                double temp = 0;
                for (int m = 0; m < divTimes; m++) {
                    if (j + m < data.numColumns()) {
                        counter++;
                        temp += data.get(i, j + m);
                    }
                }
                if (j / divTimes < dataNew.numColumns())
                    dataNew.set(i, (int) (j / divTimes), temp / counter);
            }
        }
        return dataNew;
    }
}
