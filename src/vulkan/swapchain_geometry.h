/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../common.h"

#include <limits.h>
#include <vulkan/vulkan_core.h>
#include <libplacebo/common.h>

struct vk_swapchain_geometry {
    VkExtent2D extent;
    int width, height;
    VkSurfaceTransformFlagBitsKHR transform;
    pl_rotation rotation;
};

// Resolve logical dimensions before converting to storage coordinates. Previous
// dimensions are logical too, so an unknown currentExtent cannot swap twice.
static inline bool vk_swapchain_geometry(const VkSurfaceCapabilitiesKHR *caps,
                                        bool pre_rotate, int width, int height,
                                        int previous_width, int previous_height,
                                        struct vk_swapchain_geometry *out)
{
    struct vk_swapchain_geometry g = {0};
    if (pre_rotate) {
        g.transform = caps->currentTransform;
        switch (g.transform) {
        case VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
            g.rotation = PL_ROTATION_0;
            break;
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
            g.rotation = PL_ROTATION_270;
            break;
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
            g.rotation = PL_ROTATION_180;
            break;
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
            g.rotation = PL_ROTATION_90;
            break;
        default:
            return false;
        }
        if (!(caps->supportedTransforms & g.transform))
            return false;
    } else if (caps->supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        g.transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else if (caps->supportedTransforms & VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR) {
        g.transform = VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
    } else {
        return false;
    }

    if (!width) {
        if (caps->currentExtent.width != UINT32_MAX) {
            if (caps->currentExtent.width > INT_MAX)
                return false;
            width = caps->currentExtent.width;
        } else {
            width = previous_width;
        }
    }
    if (!height) {
        if (caps->currentExtent.height != UINT32_MAX) {
            if (caps->currentExtent.height > INT_MAX)
                return false;
            height = caps->currentExtent.height;
        } else {
            height = previous_height;
        }
    }
    if (width <= 0 || height <= 0)
        return false;

    bool swapped = g.rotation % PL_ROTATION_180;
    uint32_t w = swapped ? height : width;
    uint32_t h = swapped ? width : height;
    if (caps->minImageExtent.width > caps->maxImageExtent.width ||
        caps->minImageExtent.height > caps->maxImageExtent.height)
        return false;
    w = PL_CLAMP(w, caps->minImageExtent.width, caps->maxImageExtent.width);
    h = PL_CLAMP(h, caps->minImageExtent.height, caps->maxImageExtent.height);
    if (!w || !h || w > INT_MAX || h > INT_MAX)
        return false;
    g.extent = (VkExtent2D) { w, h };
    g.width = swapped ? h : w;
    g.height = swapped ? w : h;
    *out = g;
    return true;
}
