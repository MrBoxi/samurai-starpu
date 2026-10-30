#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/field.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <array>
#include <string>
#include <vector>
#include <starpu.h>
#include <samurai/field/scalar_field.hpp>
#include <samurai/starpu_uniform/mesh.hpp>

namespace samurai
{
    namespace starpu_uniform
    {
        template <class StarpuMesh, class value_t = double>
        class StarpuUniformScalarField
        {
        public:
            using mesh_t = StarpuMesh;
            using local_field_t = samurai::ScalarField<typename StarpuMesh::mesh_t, value_t>;
            using value_type = value_t;

            StarpuUniformScalarField() = default;
            StarpuUniformScalarField(const std::string& name, StarpuMesh& starpu_mesh);
            ~StarpuUniformScalarField();

            StarpuUniformScalarField(const StarpuUniformScalarField&) = delete;
            StarpuUniformScalarField& operator=(const StarpuUniformScalarField&) = delete;

            StarpuUniformScalarField(StarpuUniformScalarField&&) noexcept = default;
            StarpuUniformScalarField& operator=(StarpuUniformScalarField&&) noexcept = delete;

            int get_nb_task() const { return m_starpu_mesh.get_nb_task(); }
            
            local_field_t& get_field(int i) { return m_fields[i]; }
            const local_field_t& get_field(int i) const { return m_fields[i]; }

            starpu_data_handle_t get_handle(int i) { return m_handles[i]; }
            starpu_data_handle_t get_handle(int i) const { return m_handles[i]; }

            starpu_data_handle_t get_sub_handle(int i, SubdomainPart part) const
            {
                return m_sub_handles[i][static_cast<std::size_t>(part)];
            }

            starpu_data_handle_t get_sub_handle(int i, std::size_t part_idx) const
            {
                return m_sub_handles[i][part_idx];
            }

            void partitions_submit()
            {
                for (int i = 0; i < m_sub_handles.size(); ++i)
                {
                    starpu_data_partition_submit(m_handles[i], 3, m_sub_handles[i].data());
                }
            }

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

            void swap(StarpuUniformScalarField& other) noexcept
            {
                m_fields.swap(other.m_fields);
                m_handles.swap(other.m_handles);
                m_partition_sizes.swap(other.m_partition_sizes);
                m_filters.swap(other.m_filters);
                m_sub_handles.swap(other.m_sub_handles);
                for (std::size_t i = 0; i < m_filters.size(); ++i)
                {
                    m_filters[i].filter_arg_ptr = m_partition_sizes[i].data();
                }
                for (std::size_t i = 0; i < other.m_filters.size(); ++i)
                {
                    other.m_filters[i].filter_arg_ptr = other.m_partition_sizes[i].data();
                }
            }

        private:
            std::string m_name;
            StarpuMesh& m_starpu_mesh;
            std::vector<local_field_t> m_fields;
            std::vector<starpu_data_handle_t> m_handles;
            std::vector<std::array<uint32_t, 3>> m_partition_sizes;
            std::vector<starpu_data_filter> m_filters;
            std::vector<std::array<starpu_data_handle_t, 3>> m_sub_handles;
        };

        template <class StarpuMesh, class value_t>
        StarpuUniformScalarField<StarpuMesh, value_t>::StarpuUniformScalarField(const std::string& name, StarpuMesh& starpu_mesh)
            : m_name(name)
            , m_starpu_mesh(starpu_mesh)
        {
            int nb_task = m_starpu_mesh.get_nb_task();
            m_fields.reserve(nb_task);
            m_handles.resize(nb_task);
            m_partition_sizes.resize(nb_task);
            m_filters.resize(nb_task);
            m_sub_handles.resize(nb_task);

            for (int i = 0; i < nb_task; ++i)
            {
                m_fields.emplace_back(name, m_starpu_mesh.get_mesh(i));
                
                // Register local field memory to StarPU (parent handle)
                starpu_vector_data_register(&m_handles[i],
                                            STARPU_MAIN_RAM,
                                            reinterpret_cast<uintptr_t>(m_fields[i].data()),
                                            m_fields[i].array().size(),
                                            sizeof(value_t));

                // Plan partitioning into 3 parts (bottom ghost, interior, top ghost)
                m_partition_sizes[i] = m_starpu_mesh.get_partition_sizes(i);

                m_filters[i].filter_func = starpu_vector_filter_list;
                m_filters[i].nchildren = 3;
                m_filters[i].filter_arg_ptr = m_partition_sizes[i].data();
                m_filters[i].get_child_ops = nullptr;
                m_filters[i].get_nchildren = nullptr;

                starpu_data_partition_plan(m_handles[i], &m_filters[i], m_sub_handles[i].data());
            }
        }

        template <class StarpuMesh, class value_t>
        StarpuUniformScalarField<StarpuMesh, value_t>::~StarpuUniformScalarField()
        {
            for (std::size_t i = 0; i < m_handles.size(); ++i)
            {
                if (m_handles[i])
                {
                    starpu_data_partition_clean(m_handles[i], 3, m_sub_handles[i].data());
                    starpu_data_unregister(m_handles[i]);
                }
            }
        }

        template <class StarpuMesh, class value_t>
        void swap(StarpuUniformScalarField<StarpuMesh, value_t>& u1, StarpuUniformScalarField<StarpuMesh, value_t>& u2) noexcept
        {
            u1.swap(u2);
        }
    }
}
