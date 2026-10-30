#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/view.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <cstddef>

namespace samurai
{
    namespace starpu_uniform
    {
        /**
         * @brief Non-owning lightweight 2D Cartesian view over raw StarPU buffer.
         *
         * Wraps a raw memory pointer (e.g. from STARPU_VECTOR_GET_PTR) with
         * grid stride and ghost cell offset. Provides convenient and efficient
         * (x, y) indexing for finite volume stencils on CPU (and future GPU).
         */
        template <class T>
        class UniformFieldView
        {
        public:
            inline UniformFieldView(T* data, int row_stride, int ghost_width)
                : m_data(data)
                , m_stride(row_stride)
                , m_ghost(ghost_width)
            {
            }

            /**
             * @brief Access element at Cartesian coordinates (x, y) relative to interior origin.
             *
             * Note: x in [0, nx-1], y in [0, ny_interior-1].
             * Ghost cells are accessed with negative coordinates (e.g. x = -1, y = -1)
             * or beyond interior (e.g. x = nx, y = ny_interior).
             */
            inline T& operator()(int x, int y) const
            {
                return m_data[(y + m_ghost) * m_stride + (x + m_ghost)];
            }

            inline T* data() const { return m_data; }
            inline int row_stride() const { return m_stride; }
            inline int ghost_width() const { return m_ghost; }

        private:
            T* m_data;
            int m_stride;
            int m_ghost;
        };
    }
}
