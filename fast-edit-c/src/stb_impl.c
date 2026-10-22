/* Single translation unit that instantiates the two single-file libraries.
   stb_image handles decoding, stb_image_write handles encoding. */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
