import applications.Decompose;
import data.LinguisticDataAbstract;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Some inputs make the analyzer emit large volumes of internal debug/log
 * output to System.out mid-computation, which would otherwise desync a
 * "one line of stdout per input word" protocol. So the real stdout is saved
 * up front, System.out is swapped to a sink for the duration of each
 * decompose call, and only our own explicitly sentinel-prefixed result line
 * is ever written to the real stdout.
 */
public class BatchDecompose {
    private static final String PREFIX = "RESULT";

    public static void main(String[] args) throws Exception {
        LinguisticDataAbstract.init();
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintStream realOut = new PrintStream(new java.io.FileOutputStream(java.io.FileDescriptor.out), true, "UTF-8");
        PrintStream sink = new PrintStream(new OutputStream() {
            public void write(int b) {}
        });

        String line;
        while ((line = br.readLine()) != null) {
            String word = line.trim();
            String result = "";
            if (!word.isEmpty()) {
                System.setOut(sink);
                try {
                    String[] decs = Decompose.decomposeToArrayOfStrings(word);
                    if (decs != null && decs.length > 0) {
                        result = decs[0];
                    }
                } catch (Exception e) {
                    result = "";
                } finally {
                    System.setOut(realOut);
                }
            }
            realOut.println(PREFIX + result);
        }
    }
}
