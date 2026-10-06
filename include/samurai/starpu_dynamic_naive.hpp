// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_dynamic_naive.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <samurai/starpu_dynamic_naive/mesh.hpp>
#include <samurai/starpu_dynamic_naive/field.hpp>
#include <samurai/starpu_dynamic_naive/algorithm.hpp>
#include <samurai/starpu_dynamic_naive/adapt.hpp>
#include <samurai/starpu_dynamic_naive/save.hpp>
