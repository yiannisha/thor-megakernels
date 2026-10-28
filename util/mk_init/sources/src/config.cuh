#pragma once
#include "megakernel.cuh"

#ifdef MK_THOR
struct {{PROJECT_NAME_LOWER}}_config : megakernel::thor_config {};
#else
struct {{PROJECT_NAME_LOWER}}_config : megakernel::default_config {};
#endif
