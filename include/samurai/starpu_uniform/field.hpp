#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/field.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

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
            }

        private:
            std::string m_name;
            StarpuMesh& m_starpu_mesh;
            std::vector<local_field_t> m_fields;
            std::vector<starpu_data_handle_t> m_handles;
        };

        template <class StarpuMesh, class value_t>
        StarpuUniformScalarField<StarpuMesh, value_t>::StarpuUniformScalarField(const std::string& name, StarpuMesh& starpu_mesh)
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
        StarpuUniformScalarField<StarpuMesh, value_t>::~StarpuUniformScalarField()
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
        void swap(StarpuUniformScalarField<StarpuMesh, value_t>& u1, StarpuUniformScalarField<StarpuMesh, value_t>& u2) noexcept
        {
            u1.swap(u2);
        }
    }
}
