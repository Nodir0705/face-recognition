# Keep TFLite + ML Kit native interop classes
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.mlkit.** { *; }
-dontwarn org.tensorflow.lite.**
-dontwarn com.google.mlkit.**
