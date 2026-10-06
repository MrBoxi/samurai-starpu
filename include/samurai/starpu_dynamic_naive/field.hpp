// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_dynamic_naive/field.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <string>
#include <vector>
#include <type_traits>
#include <utility>
#include <starpu.h>
#include <samurai/field/scalar_field.hpp>
#include <samurai/concepts.hpp>
#include <samurai/bc.hpp>
#include <samurai/starpu_dynamic_naive/mesh.hpp>

namespace samurai
{
    namespace starpu_dynamic_naive
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

            StarpuMRScalarField(StarpuMRScalarField&& other) noexcept
            {
                swap(other);
            }

            StarpuMRScalarField& operator=(StarpuMRScalarField&& other) noexcept
            {
                if (this != &other)
                {
                    unregister_handles();
                    swap(other);
                }
                return *this;
            }

            int get_nb_task() const { return m_starpu_mesh ? m_starpu_mesh->get_nb_task() : 0; }
            
            local_field_t& get_field(int i) { return m_fields[i]; }
            const local_field_t& get_field(int i) const { return m_fields[i]; }

            starpu_data_handle_t get_handle(int i) { return m_handles[i]; }
            starpu_data_handle_t get_handle(int i) const { return m_handles[i]; }

            StarpuMesh& starpu_mesh() { return *m_starpu_mesh; }
            const StarpuMesh& starpu_mesh() const { return *m_starpu_mesh; }

            const std::string& name() const { return m_name; }

            void register_handles()
            {
                if (m_registered || !m_starpu_mesh)
                {
                    return;
                }
                int nb_task = m_starpu_mesh->get_nb_task();
                m_handles.resize(nb_task, nullptr);
                for (int i = 0; i < nb_task; ++i)
                {
                    starpu_vector_data_register(&m_handles[i],
                                                STARPU_MAIN_RAM,
                                                reinterpret_cast<uintptr_t>(m_fields[i].data()),
                                                m_fields[i].array().size(),
                                                sizeof(value_t));
                }
                m_registered = true;
            }

            void unregister_handles()
            {
                if (!m_registered)
                {
                    return;
                }
                for (auto& handle : m_handles)
                {
                    if (handle)
                    {
                        starpu_data_unregister(handle);
                        handle = nullptr;
                    }
                }
                m_registered = false;
            }

            template <class GlobalField>
            void gather_to(GlobalField& global_field) const
            {
                global_field.fill(0.0);
                int nb_task = m_starpu_mesh->get_nb_task();
                for (int i = 0; i < nb_task; ++i)
                {
                    const auto& local_field = m_fields[i];
                    const auto& local_mesh = local_field.mesh();
                    samurai::for_each_interval(local_mesh, [&](std::size_t level, const auto& interval, const auto& index) {
                        global_field(level, interval, index) = local_field(level, interval, index);
                    });
                }
            }

            template <class GlobalField>
            void scatter_from_and_register(const GlobalField& global_field)
            {
                unregister_handles();
                int nb_task = m_starpu_mesh->get_nb_task();
                bool has_global_bc = !global_field.get_bc().empty();
                const auto* prev_field = (!has_global_bc && !m_fields.empty() && !m_fields[0].get_bc().empty()) ? &m_fields[0] : nullptr;

                std::vector<local_field_t> new_fields;
                new_fields.reserve(nb_task);

                for (int i = 0; i < nb_task; ++i)
                {
                    new_fields.emplace_back(m_name, m_starpu_mesh->get_mesh(i));
                    auto& local_field = new_fields.back();
                    if (has_global_bc)
                    {
                        local_field.copy_bc_from(global_field);
                    }
                    else if (prev_field)
                    {
                        local_field.copy_bc_from(*prev_field);
                    }
                    local_field.fill(0.0);
                    samurai::for_each_interval(local_field.mesh(), [&](std::size_t level, const auto& interval, const auto& index) {
                        local_field(level, interval, index) = global_field(level, interval, index);
                    });
                }
                m_fields = std::move(new_fields);
                register_handles();
            }

            void rebuild_and_register()
            {
                unregister_handles();
                int nb_task = m_starpu_mesh->get_nb_task();
                bool has_bc = !m_fields.empty() && !m_fields[0].get_bc().empty();
                const auto* prev_field = has_bc ? &m_fields[0] : nullptr;

                std::vector<local_field_t> new_fields;
                new_fields.reserve(nb_task);
                for (int i = 0; i < nb_task; ++i)
                {
                    new_fields.emplace_back(m_name, m_starpu_mesh->get_mesh(i));
                    if (prev_field)
                    {
                        new_fields.back().copy_bc_from(*prev_field);
                    }
                    new_fields.back().fill(0.0);
                }
                m_fields = std::move(new_fields);
                register_handles();
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
                std::swap(m_starpu_mesh, other.m_starpu_mesh);
                m_fields.swap(other.m_fields);
                m_handles.swap(other.m_handles);
                std::swap(m_registered, other.m_registered);
            }

        private:
            std::string m_name;
            StarpuMesh* m_starpu_mesh = nullptr;
            std::vector<local_field_t> m_fields;
            std::vector<starpu_data_handle_t> m_handles;
            bool m_registered = false;
        };

        template <class StarpuMesh, class value_t>
        StarpuMRScalarField<StarpuMesh, value_t>::StarpuMRScalarField(const std::string& name, StarpuMesh& starpu_mesh)
            : m_name(name)
            , m_starpu_mesh(&starpu_mesh)
        {
            int nb_task = m_starpu_mesh->get_nb_task();
            m_fields.reserve(nb_task);
            for (int i = 0; i < nb_task; ++i)
            {
                m_fields.emplace_back(name, m_starpu_mesh->get_mesh(i));
                m_fields.back().fill(0.0);
            }
            register_handles();
        }

        template <class StarpuMesh, class value_t>
        template <class GlobalField>
            requires samurai::field_like<std::remove_cvref_t<GlobalField>>
        StarpuMRScalarField<StarpuMesh, value_t>::StarpuMRScalarField(GlobalField& global_field, StarpuMesh& starpu_mesh)
            : m_name(global_field.name())
            , m_starpu_mesh(&starpu_mesh)
        {
            scatter_from_and_register(global_field);
        }

        template <class StarpuMesh, class value_t>
        StarpuMRScalarField<StarpuMesh, value_t>::~StarpuMRScalarField()
        {
            unregister_handles();
        }

        template <class StarpuMesh, class value_t>
        void swap(StarpuMRScalarField<StarpuMesh, value_t>& u1, StarpuMRScalarField<StarpuMesh, value_t>& u2) noexcept
        {
            u1.swap(u2);
        }

        template <class bc_type, class StarpuField, class... Args>
        void make_bc(StarpuField& u, Args&&... args)
        {
            for (int i = 0; i < u.get_nb_task(); ++i)
            {
                samurai::make_bc<bc_type>(u.get_field(i), std::forward<Args>(args)...);
            }
        }

        template <template <class> class bc_type, class StarpuField, class... Args>
        void make_bc(StarpuField& u, Args&&... args)
        {
            for (int i = 0; i < u.get_nb_task(); ++i)
            {
                samurai::make_bc<bc_type>(u.get_field(i), std::forward<Args>(args)...);
            }
        }
    }

    using starpu_dynamic_naive::make_bc;
}
