#!/usr/bin/env bash
# Build the debug APK and install it on the first connected Android device.
# No Android Studio required.

set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export ANDROID_HOME=/home/netvision/Android
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH

cd "$(dirname "$0")"

echo ">>> assembleDebug"
./gradlew assembleDebug --console=plain

APK=app/build/outputs/apk/debug/app-debug.apk
echo ">>> adb devices"
adb devices

echo ">>> adb install -r $APK"
adb install -r "$APK"

echo
echo "Done. Launch 'Attendance Spike' from the tablet's app drawer."
