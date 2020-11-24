import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
/**
* Unit test for Main.
*/
public class MainTest
{
/**
* Rigorous Test :-)
*/
@Test
public void shouldAnswerWithTrue()
{
assertTrue( true );
}
@Test
public void testConstructor()
{
final int value = 13;
final Integer epochTime = -1;
final String name = "CS2334";
final Main app = new Main(value, name, epochTime);
final Integer expectedInt = app.getEpochTime();
assertEquals(expectedInt, epochTime);
final String expectedStr = app.getName();
assertEquals(expectedStr, name);
final int expectedVal = app.getValue();
assertEquals(expectedVal, value);
}
@Test
public void testConstructorNoArgs() {
final int value = 8;
final Integer epochTime = 0;
final String name = "Java 2";
final Main app = new Main();
final Integer expectedInt = app.getEpochTime();
assertEquals(expectedInt, epochTime);
final String expectedStr = app.getName();
assertEquals(expectedStr, name);
final int expectedVal = app.getValue();
assertEquals(expectedVal, value);
}
@Test
public void testFail() {
assertTrue(!false);
}
@Test
public void testGetValue() {
final Main app = new Main();
final int x = app.getValue();
final int expected = 8;
assertEquals(expected, x);
}
@Test
public void testCreateBoxEmpty() {
    
    final Main app = new Main();
final String str = "";
final String box = app.createBox(str, "*");
System.out.println(box);
final String expected = "***\n*.*\n***\n";
System.out.println(expected);
assertEquals(expected, box);
}
@Test
public void testCreateBoxNull() {
final Main app = new Main();
final String str = null;
final String box = app.createBox(str, "*");
System.out.println(box);
final String expected = "!";
System.out.println(expected);
assertEquals(expected, box);
}
@Test
public void testCreateBoxAha() {
final Main app = new Main();
final String str = "AHA";
final String box = app.createBox(str, "*");
System.out.println(box);
final String expected = "*****\n*AHA*\n*****\n";
System.out.println(expected);
assertEquals(expected, box);
}
@Test
public void testCreateLine() {
final Main app = new Main();
final String expected = "===";
final String actual = app.recursiveLine("=",
expected.length());
assertEquals(expected, actual);
}
}
