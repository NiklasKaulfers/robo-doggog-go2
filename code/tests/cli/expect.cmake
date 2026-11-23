# Runs a program and checks how it ended, for the command line tests:
#
#   cmake -DPROGRAM=<exe> -DARGS=<a|b|c> -DEXIT_CODE=<n> [-DOUTPUT=<regex;regex>] [-DFORBID=<regex;regex>] -P expect.cmake
#
# ARGS are separated by |. Every OUTPUT regex must match and no FORBID regex may match
# stdout and stderr together.

string(REPLACE "|" ";" arguments "${ARGS}")
execute_process(
        COMMAND "${PROGRAM}" ${arguments}
        RESULT_VARIABLE exitCode
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        TIMEOUT 100)
set(output "${out}${err}")

set(failed "")
if (NOT "${exitCode}" STREQUAL "${EXIT_CODE}")
    string(APPEND failed "exit code ${exitCode}, expected ${EXIT_CODE}\n")
endif ()
foreach (regex IN LISTS OUTPUT)
    if (NOT output MATCHES "${regex}")
        string(APPEND failed "missing in the output: ${regex}\n")
    endif ()
endforeach ()
foreach (regex IN LISTS FORBID)
    if (output MATCHES "${regex}")
        string(APPEND failed "must not be in the output: ${regex}\n")
    endif ()
endforeach ()

if (failed)
    message(FATAL_ERROR "${PROGRAM} ${ARGS}\n${failed}--- output ---\n${output}")
endif ()
message(STATUS "${PROGRAM} ${ARGS}: exit code ${exitCode} as expected")
