# Set appropriate compiler and linker flags for sanitizers.
#
# Usage of this module:
#  enable_sanitizers("address;leak")

# Add flags
macro(enable_sanitizer sanitizer)
  if(${sanitizer} MATCHES "address")
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=address")

  elseif(${sanitizer} MATCHES "thread")
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=thread")

  elseif(${sanitizer} MATCHES "leak")
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=leak")

  elseif(${sanitizer} MATCHES "undefined")
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=undefined -fno-sanitize-recover=undefined")

  else()
    message(FATAL_ERROR "Sanitizer ${sanitizer} not supported.")
  endif()
endmacro()

macro(enable_sanitizers SANITIZERS)
  # Check sanitizers compatibility.
  foreach(_san ${SANITIZERS})
    string(TOLOWER ${_san} _san)
    if(_san MATCHES "thread")
      if(${_use_other_sanitizers})
        message(FATAL_ERROR "thread sanitizer is not compatible with ${_san} sanitizer.")
      endif()
      set(_use_thread_sanitizer 1)
    else()
      if(${_use_thread_sanitizer})
        message(FATAL_ERROR "${_san} sanitizer is not compatible with thread sanitizer.")
      endif()
      set(_use_other_sanitizers 1)
    endif()
  endforeach()

  message(STATUS "Sanitizers: ${SANITIZERS}")

  foreach(_san ${SANITIZERS})
    string(TOLOWER ${_san} _san)
    enable_sanitizer(${_san})
  endforeach()
  message(STATUS "Sanitizers compile flags: ${SAN_COMPILE_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_COMPILE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_COMPILE_FLAGS}")
endmacro()
