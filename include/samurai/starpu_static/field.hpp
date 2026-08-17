// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_static/field.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <string>
#include <vector>
#include <type_traits>
#include <starpu.h>
#include <samurai/field/scalar_field.hpp>
#include <samurai/concepts.hpp>
#include <samurai/starpu_static/mesh.hpp>

namespace samurai
{
    namespace starpu_static
    {
        template <class StarpuMesh, class value_t = double>
        class StarpuMRScalarField
        {
        public:
            using mesh_t = StarpuMesh;
            using local_field_t = samurai::ScalarField<typename StarpuMesh::mesh_t, value_t>;
            using value_type = value_t;

            StarpuMRScalarField() = default;
            StarpuMRScalarField(const std::string& name, StarpuMesh& starpu_mesh);
            
            template <class GlobalField>
                requires samurai::field_like<std::remove_cvref_t<GlobalField>>
            StarpuMRScalarField(GlobalField& global_field, StarpuMesh& starpu_mesh);

            ~StarpuMRScalarField();

            StarpuMRScalarField(const StarpuMRScalarField&) = delete;
            StarpuMRScalarField& operator=(const StarpuMRScalarField&) = delete;

            StarpuMRScalarField(StarpuMRScalarField&&) noexcept = default;
            StarpuMRScalarField& operator=(StarpuMRScalarField&&) noexcept = delete;

            int get_nb_task() const { return m_starpu_mesh.get_nb_task(); }
            
            local_field_t& get_field(int i) { return m_fields[i]; }
            const local_field_t& get_field(int i) const { return m_fields[i]; }

            starpu_data_handle_t get_handle(int i) { return m_handles[i]; }
            starpu_data_handle_t get_handle(int i) const { return m_handles[i]; }

            StarpuMesh& starpu_mesh() { return m_starpu_mesh; }
            const StarpuMesh& starpu_mesh() const { return m_starpu_mesh; }

            const std::string& name() const { return m_name; }

            void resize()
            {
                for (auto& field : m_fields)
                {
                    field.resize();
                }
            }

            void fill(value_type value)
            {
                for (auto& field : m_fields)
                {
                    field.fill(value);
                }
            }

            void swap(StarpuMRScalarField& other) noexcept
            {
                m_fields.swap(other.m_fields);
                m_handles.swap(other.m_handles);
            }

        private:
            std::string m_name;
            StarpuMesh& m_starpu_mesh;
            std::vector<local_field_t> m_fields;
            std::vector<starpu_data_handle_t> m_handles;
        };

        template <class StarpuMesh, class value_t>
        StarpuMRScalarField<StarpuMesh, value_t>::StarpuMRScalarField(const std::string& name, StarpuMesh& starpu_mesh)
            : m_name(name)
            , m_starpu_mesh(starpu_mesh)
        {
            int nb_task = m_starpu_mesh.get_nb_task();
            m_fields.reserve(nb_task);
            m_handles.resize(nb_task);

            for (int i = 0; i < nb_task; ++i)
            {
                m_fields.emplace_back(name, m_starpu_mesh.get_mesh(i));
                
                // Register local field memory to StarPU
                starpu_vector_data_register(&m_handles[i],
                                            STARPU_MAIN_RAM,
                                            reinterpret_cast<uintptr_t>(m_fields[i].data()),
                                            m_fields[i].array().size(),
                                            sizeof(value_t));
            }
        }

        template <class StarpuMesh, class value_t>
        template <class GlobalField>
            requires samurai::field_like<std::remove_cvref_t<GlobalField>>
        StarpuMRScalarField<StarpuMesh, value_t>::StarpuMRScalarField(GlobalField& global_field, StarpuMesh& starpu_mesh)
            : m_name(global_field.name())
            , m_starpu_mesh(starpu_mesh)
        {
            int nb_task = m_starpu_mesh.get_nb_task();
            m_fields.reserve(nb_task);
            m_handles.resize(nb_task);

            using mesh_id_t = typename local_field_t::mesh_t::mesh_id_t;

            for (int i = 0; i < nb_task; ++i)
            {
                m_fields.emplace_back(m_name, m_starpu_mesh.get_mesh(i));
                
                auto& local_field = m_fields.back();
                samurai::for_each_interval(local_field.mesh()[mesh_id_t::reference], [&](std::size_t level, const auto& interval, const auto& index) {
                    local_field(level, interval, index) = global_field(level, interval, index);
                });

                // Register local field memory to StarPU
                starpu_vector_data_register(&m_handles[i],
                                            STARPU_MAIN_RAM,
                                            reinterpret_cast<uintptr_t>(local_field.data()),
                                            local_field.array().size(),
                                            sizeof(value_t));
            }
        }

        template <class StarpuMesh, class value_t>
        StarpuMRScalarField<StarpuMesh, value_t>::~StarpuMRScalarField()
        {
            for (auto handle : m_handles)
            {
                if (handle)
                {
                    starpu_data_unregister(handle);
                }
            }
        }

        template <class StarpuMesh, class value_t>
        void swap(StarpuMRScalarField<StarpuMesh, value_t>& u1, StarpuMRScalarField<StarpuMesh, value_t>& u2) noexcept
        {
            u1.swap(u2);
        }
    }
}
