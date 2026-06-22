if(NOT DEFINED OBJDUMP)
  message(FATAL_ERROR "OBJDUMP is not set")
endif()

if(NOT DEFINED INPUT)
  message(FATAL_ERROR "INPUT is not set")
endif()

if(NOT DEFINED OUTPUT)
  message(FATAL_ERROR "OUTPUT is not set")
endif()

execute_process(
  COMMAND "${OBJDUMP}" -d "${INPUT}"
  OUTPUT_FILE "${OUTPUT}"
  RESULT_VARIABLE dump_result
)

if(NOT dump_result EQUAL 0)
  message(FATAL_ERROR "objdump failed for ${INPUT}")
endif()
