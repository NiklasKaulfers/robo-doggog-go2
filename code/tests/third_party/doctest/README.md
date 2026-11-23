# doctest (vendored)

Test framework for `code/tests`, vendored so the tests build without network access or
installed packages (also in the Jetson devcontainer).

- Version: 2.4.12, from https://github.com/doctest/doctest/tree/v2.4.12
- License: MIT, see `LICENSE.txt`
- Files: `doctest/doctest.h`, `scripts/cmake/doctest.cmake`, `scripts/cmake/doctestAddTests.cmake`

Local change: `doctestAddTests.cmake` runs the test executable (to list its tests) with
`TIMEOUT 60`, so a test executable that hangs at startup fails the build instead of
blocking it forever (AddressSanitizer does that on current macOS). Keep it when updating.

SHA-256 of the files as downloaded (before the local change):

```
94029a7d32da24a56249658147dbd2b33ff0b9ed665295cbbaf19aafff5b0ced  doctest.h
671d9715e33e84a990a55a145a120b8e5d47583cbf84595dfe133e3b853bc4d1  doctest.cmake
68ff6641f7fed897e8579be58b96d01cd5a3cae3abe9ecb185e2d0a94ff7cce2  doctestAddTests.cmake
```
