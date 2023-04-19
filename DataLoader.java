import no.uib.cipr.matrix.DenseMatrix;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class DataLoader {
    public DataLoader() {

    }

    public DenseMatrix loadMatrix(String filename) throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
        // FileInputStream
        String firstLineString = reader.readLine();
        String[] strings = firstLineString.split(" ");
        int m = Integer.parseInt(strings[0]);
        int n = Integer.parseInt(strings[1]);

        DenseMatrix matrix = new DenseMatrix(m, n);
        firstLineString = reader.readLine();
        int i = 0;
        while (i < m) {
            String[] dataStrings = firstLineString.split(" ");
            for (int j = 0; j < n; j++) {
                matrix.set(i, j, Double.parseDouble(dataStrings[j]));
            }
            i++;
            firstLineString = reader.readLine();
        }
        reader.close();
        /*
         * for(int ii = 0; ii<m; ii++)
         * matrix.add(ii, 0, -1);
         */
        return matrix;
    }

    public int getLabelClass(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
        // FileInputStream
        String firstLineString = reader.readLine();
        String[] strings = firstLineString.split(" ");

        int labelClass = 0;
        if (strings.length > 2)
            labelClass = Integer.parseInt(strings[2]);
        reader.close();
        return labelClass;
    }
}
