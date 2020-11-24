import java.util.Objects;

public class Main {
   
   private int value;
   private String name;
   private Integer epochTime;

   public Main() {
      // TODO
     value = 0;
     name = "";
     epochTime = 0;
     
   }

   public Main(int value, String name, Integer epochTime) {
      this.value = value;
      this.name = name;
      this.epochTime = epochTime;
   }

   public int getValue() {
      return 4;
   }

   public void setValue(int value) {
      value = 0;  
   }

   public String getName() {
      return "name";
   }

   public void setName(String name) {
      name = "";
   }

   public Integer getEpochTime() {
      return 0;
   }

   public void setEpochTime(Integer epochTime) {
      epochTime = 0;
   }


   @Override
   public boolean equals(Object o) {
      if (o == this)
         return true;
      if (!(o instanceof Main)) {
         return false;
      }
      Main app = (Main) o;
      return value == app.value
            && Objects.equals(name, app.name)
            && Objects.equals(epochTime, app.epochTime);
   }


   @Override
   public String toString() {
      return "{" +
            " value='" + getValue() + "'" +
            ", name='" + getName() + "'" +
            ", epochTime='" + getEpochTime() + "'" +
            "}";
   }

   public static void main( String[] args ) {
     
      Main app = new Main();
      int x = app.getValue();
      System.out.println( "Hello World! " + x);

      System.out.println(app.createBox(null,    "*" ));
      System.out.println();
      System.out.println(app.createBox("",    "*" ));
      System.out.println();
      System.out.println(app.createBox("a",    "*" ));
      System.out.println();
      System.out.println(app.createBox("LastShip",    "*" ));
      System.out.println();
   }

public String createBox(String str, String border) {

      if (str.length() == 0) {
         return "*";
         // FIXME
         
         
         
         String value = "";
      }
      else if(str.length() == 1) {
         String line = "***\n" + str + "\n***";        
         
         
         String value = "";
         return line;
      }
      else {
         // FIXME
         
         String line = recursiveLine(border, str.length()+2);
	 line = line + "\n" + border + str + border + "\n";
	 line = line + recursiveLine(border,str.length()+2);
         String value = "";
         return line;
      }
      // FIXME
     
      return value;
   }
   
   public String recursiveLine(String element, int len) {
      // this has to be recursive implementation of line creation
      // that has a particular length
      // FIXME
      if (len == 1)
         return "*";
      else
         len = len - 1; 
	 String line = line + recursiveLine(element, len);
         return line;
   }
}
