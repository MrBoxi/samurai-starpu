// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_static/mesh.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <vector>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>

namespace samurai
{
    namespace starpu_static
    {
        template <class Config>
        class StarpuMRMesh
        {
        public:
            using mesh_t = MRMesh<Config>;
            using config_t = Config;
            static constexpr std::size_t dim = Config::dim;
            static_assert(dim == 1 || dim == 2, "StarpuMRMesh supports only 1D and 2D domain decomposition");
            using interval_t = typename Config::interval_t;
            using cl_type = CellList<dim, interval_t>;
            using ca_type = LevelCellArray<dim, interval_t>;

            struct IntersectionEntry
            {
                int src;
                int dst;
                std::size_t level;
                ca_type intersection;
            };

            StarpuMRMesh() = default;
            StarpuMRMesh(const mesh_t& global_mesh, int nb_task);

            int get_nb_task() const { return m_nb_task; }
            std::size_t min_level() const { return m_global_mesh.min_level(); }
            std::size_t max_level() const { return m_global_mesh.max_level(); }

            mesh_t& get_mesh(int i) { return m_meshes[i]; }
            const mesh_t& get_mesh(int i) const { return m_meshes[i]; }

            mesh_t& get_global_mesh() { return m_global_mesh; }
            const mesh_t& get_global_mesh() const { return m_global_mesh; }

            const std::vector<IntersectionEntry>& get_intersections() const { return m_intersections; }

            double min_cell_length() const
            {
                return m_global_mesh.cell_length(m_global_mesh.max_level());
            }

            void swap(StarpuMRMesh& other) noexcept
            {
                std::swap(m_nb_task, other.m_nb_task);
                m_meshes.swap(other.m_meshes);
                m_global_mesh.swap(other.m_global_mesh);
                m_intersections.swap(other.m_intersections);
            }

        private:
            int m_nb_task = 0;
            std::vector<mesh_t> m_meshes;
            mesh_t m_global_mesh;
            std::vector<IntersectionEntry> m_intersections;
        };

        template <class Config>
        StarpuMRMesh<Config>::StarpuMRMesh(const mesh_t& global_mesh, int nb_task)
            : m_nb_task(nb_task)
            , m_global_mesh(global_mesh)
        {
            assert(nb_task >= 1);
            using mesh_id_t = typename mesh_t::mesh_id_t;
            auto& global_domain = m_global_mesh[mesh_id_t::cells];
            m_meshes.reserve(nb_task);

            constexpr std::size_t split_dim = (dim >= 2) ? 1 : 0;
            double spatial_min = m_global_mesh.domain().min_corner()[split_dim];
            double spatial_max = m_global_mesh.domain().max_corner()[split_dim];
            double spatial_len = spatial_max - spatial_min;

            for (int rank = 0; rank < nb_task; ++rank)
            {
                cl_type subdomain_cells(global_domain.origin_point(), global_domain.scaling_factor());
                double rank_min = spatial_min + (static_cast<double>(rank) / static_cast<double>(nb_task)) * spatial_len;
                double rank_max = spatial_min + (static_cast<double>(rank + 1) / static_cast<double>(nb_task)) * spatial_len;

                if (dim == 1)
                {
                    for_each_meshinterval(global_domain,
                                          [&](auto mi)
                                          {
                                              double cell_length = m_global_mesh.cell_length(mi.level);
                                              for (auto i = mi.i.start; i < mi.i.end; ++i)
                                              {
                                                  double pos = (static_cast<double>(i) + 0.5) * cell_length + m_global_mesh.origin_point()[0];
                                                  bool in_band = (rank == nb_task - 1) ? (pos >= rank_min && pos <= rank_max)
                                                                                       : (pos >= rank_min && pos < rank_max);
                                                  if (in_band)
                                                  {
                                                      subdomain_cells[mi.level][mi.index].add_point(i);
                                                  }
                                              }
                                          });
                }
                else
                {
                    for_each_meshinterval(global_domain,
                                          [&](auto mi)
                                          {
                                              double cell_length = m_global_mesh.cell_length(mi.level);
                                              double pos = (static_cast<double>(mi.index[0]) + 0.5) * cell_length + m_global_mesh.origin_point()[split_dim];
                                              bool in_band = (rank == nb_task - 1) ? (pos >= rank_min && pos <= rank_max)
                                                                                   : (pos >= rank_min && pos < rank_max);
                                              if (in_band)
                                              {
                                                  subdomain_cells[mi.level][mi.index].add_interval(mi.i);
                                              }
                                          });
                }

                typename mesh_t::ca_type subdomain_ca(subdomain_cells);
                m_meshes.emplace_back(subdomain_ca, m_global_mesh);
            }

            // Compute exact safe intersections: src cells (including projected cells) intersected with dst reference cells
            for (std::size_t level = m_global_mesh.min_level(); level <= m_global_mesh.max_level(); ++level)
            {
                for (int j = 0; j < nb_task; ++j) // src sender
                {
                    for (int i = 0; i < nb_task; ++i) // dst receiver
                    {
                        if (i == j) continue;

                        const auto& src_mesh = m_meshes[j];
                        const auto& dst_mesh = m_meshes[i];

                        auto intersect = intersection(src_mesh[mesh_id_t::all_cells][level],
                                                      dst_mesh[mesh_id_t::all_cells][level],
                                                      src_mesh.subdomain(level));

                        ca_type ca_intersect(intersect);
                        if (ca_intersect.nb_cells() > 0)
                        {
                            m_intersections.push_back({j, i, level, ca_intersect});
                        }
                    }
                }
            }
        }
    }
}
