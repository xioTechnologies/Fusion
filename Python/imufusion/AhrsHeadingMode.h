#ifndef AHRS_HEADING_MODE_H
#define AHRS_HEADING_MODE_H

#include "../../Fusion/Fusion.h"
#include <Python.h>

static int ahrs_heading_mode_from(FusionAhrsHeadingMode *const heading_mode, const int heading_mode_int) {
    switch (heading_mode_int) {
        case FusionAhrsHeadingModeMagnetic:
        case FusionAhrsHeadingModeRelative:
        case FusionAhrsHeadingModeExternal:
            *heading_mode = (FusionAhrsHeadingMode) heading_mode_int;
            return 0;
    }

    PyErr_SetString(PyExc_ValueError, "'heading_mode' must be imufusion.HEADING_MODE_*");
    return -1;
}

#endif
