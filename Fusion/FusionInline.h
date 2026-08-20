/**
 * @file FusionInline.h
 * @author Seb Madgwick
 * @brief Inline macro.
 */

#ifndef FUSION_INLINE_H
#define FUSION_INLINE_H

//------------------------------------------------------------------------------
// Includes

#include "FusionConfig.h"

//------------------------------------------------------------------------------
// Definitions

#if defined FUSION_NO_INLINE
#define FUSION_INLINE inline
#elif defined _MSC_VER
#define FUSION_INLINE __forceinline
#elif defined __GNUC__
#define FUSION_INLINE inline __attribute__((always_inline))
#else
#define FUSION_INLINE inline
#endif

#endif

//------------------------------------------------------------------------------
// End of file
