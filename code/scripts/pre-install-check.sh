#!/usr/bin/env bash
# Checks the code before it is installed on the robot. Exit code 0 means OK to install.
#
#   code/scripts/pre-install-check.sh
#
# 1. builds the tests (with sanitizers) and runs them: unit tests, the robot code in a
#    simulated world, and the programs as a whole (see code/tests/CMakeLists.txt)
# 2. in the Jetson devcontainer (Linux on aarch64, where the Unitree SDK and LiteRT
#    work): builds the real app in code/build, the one that gets run
# 3. when there are photos in code/res/target: checks them with facecheck --strict
#    (needs an OpenCV that runs the face detector model, i.e. the devcontainer's)
#
# Runs on a laptop with OpenCV as well; there it skips step 2. Known issues
# (`ctest -L known`) are not part of the check.

set -euo pipefail

cd "$(dirname "$0")/.."
CODE_DIR=$(pwd)
BUILD_DIR="$CODE_DIR/build-check"
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
CXX_COMPILER=${CXX:-c++}

step() { printf '\n=== %s\n' "$*"; }
fail() { printf '\nNOT OK TO INSTALL: %s\n' "$*" >&2; exit 1; }

# runs a command for at most $1 seconds
with_timeout() {
    local seconds=$1
    shift
    perl -e 'alarm shift; exec @ARGV' "$seconds" "$@"
}

# AddressSanitizer deadlocks at startup on some macOS versions; use it where it works
sanitizers() {
    local probe
    probe=$(mktemp -d)
    printf 'int main() { return 0; }\n' > "$probe/probe.cpp"
    local result=undefined
    if "$CXX_COMPILER" -fsanitize=address,undefined "$probe/probe.cpp" -o "$probe/probe" > /dev/null 2>&1 &&
        with_timeout 20 "$probe/probe" > /dev/null 2>&1; then
        result=address,undefined
    fi
    rm -rf "$probe"
    echo "$result"
}

generator=()
if command -v ninja > /dev/null; then generator=(-G Ninja); fi

step "sanitizers"
SANITIZE=$(sanitizers)
echo "building with -fsanitize=$SANITIZE"
if [ "$SANITIZE" = undefined ]; then
    echo "(AddressSanitizer doesn't start on this machine, the devcontainer run covers it)"
fi

step "building the tests in $BUILD_DIR"
cmake -S "$CODE_DIR" -B "$BUILD_DIR" ${generator[@]+"${generator[@]}"} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DFOLLOW_TOOLS_ONLY=ON \
    -DFOLLOW_TESTS=ON \
    -DFOLLOW_SANITIZE="$SANITIZE" > /dev/null || fail "configuring the test build failed"
cmake --build "$BUILD_DIR" -j "$JOBS" || fail "building the tests failed"

step "running the tests"
ctest --test-dir "$BUILD_DIR" --output-on-failure -j "$JOBS" -LE known || fail "tests failed"

step "building the robot app"
if [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] && [ -f "$CODE_DIR/../sdk/install/lib/libunitree_sdk2.a" ]; then
    cmake -S "$CODE_DIR" -B "$CODE_DIR/build" ${generator[@]+"${generator[@]}"} > /dev/null || fail "configuring the app failed"
    cmake --build "$CODE_DIR/build" -j "$JOBS" || fail "building the app failed"
else
    echo "skipped: the Unitree SDK and LiteRT only build in the Jetson devcontainer, run this script there too"
fi

step "checking the photos in res/target"
photos=0
if [ -d "$CODE_DIR/res/target" ]; then
    photos=$(find "$CODE_DIR/res/target" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l | tr -d ' ')
fi
if [ "$photos" -eq 0 ]; then
    echo "no photos in res/target, the app will follow the first person it sees"
else
    set +e
    output=$("$BUILD_DIR/facecheck" "$CODE_DIR/res/target" --strict 2>&1)
    status=$?
    set -e
    echo "$output"
    if echo "$output" | grep -q "face detection failed"; then
        echo "skipped: this OpenCV can't run the face detector model, check the photos in the devcontainer"
    elif [ "$status" -ne 0 ]; then
        fail "the photos in res/target are not usable as they are (see above)"
    fi
fi

printf '\nOK TO INSTALL\n'
