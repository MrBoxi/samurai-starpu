// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_static.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <samurai/starpu_static/mesh.hpp>
#include <samurai/starpu_static/field.hpp>
#include <samurai/starpu_static/algorithm.hpp>
#include <samurai/starpu_static/save.hpp>
