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

#include <libplacebo/vulkan.h>

#define VK_PRESENT_HISTORY 256

struct vk_present_history {
    uint32_t next_id;
    struct { uint32_t id; uint64_t begin; } receipts[VK_PRESENT_HISTORY];
    struct pl_vulkan_presentation_feedback feedback;
};

static inline void vk_present_submit(struct vk_present_history *h,
                                    uint32_t id, uint64_t begin, VkResult result)
{
    h->feedback.latest_submit_id = 0;
    h->feedback.latest_submit_begin_ns = 0;
    if (!id || !begin || (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR))
        return;
    h->receipts[id % VK_PRESENT_HISTORY].id = id;
    h->receipts[id % VK_PRESENT_HISTORY].begin = begin;
    h->feedback.latest_submit_id = id;
    h->feedback.latest_submit_begin_ns = begin;
}

static inline void vk_present_observe(struct vk_present_history *h,
                                     const VkPastPresentationTimingGOOGLE *t)
{
    uint32_t slot = t->presentID % VK_PRESENT_HISTORY;
    if (!t->presentID || h->receipts[slot].id != t->presentID ||
        !t->actualPresentTime ||
        t->actualPresentTime < h->receipts[slot].begin ||
        t->actualPresentTime <= h->feedback.actual_present_ns)
        return;
    h->feedback.matched_submit_id = t->presentID;
    h->feedback.matched_submit_begin_ns = h->receipts[slot].begin;
    h->feedback.actual_present_ns = t->actualPresentTime;
}

static inline bool vk_present_update(struct vk_present_history *h, VkResult res,
                                     const VkPastPresentationTimingGOOGLE *times,
                                     uint32_t count)
{
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
        h->feedback.matched_submit_id = 0;
        h->feedback.matched_submit_begin_ns = 0;
        h->feedback.actual_present_ns = 0;
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            h->feedback.latest_submit_id = 0;
            h->feedback.latest_submit_begin_ns = 0;
        }
        return false;
    }
    for (uint32_t i = 0; i < count; i++)
        vk_present_observe(h, &times[i]);
    return h->feedback.latest_submit_id && h->feedback.matched_submit_id;
}

static inline bool vk_present_poll(struct vk_present_history *h, VkDevice device,
                                   VkSwapchainKHR swapchain,
                                   PFN_vkGetPastPresentationTimingGOOGLE get,
                                   struct pl_vulkan_presentation_feedback *out,
                                   VkResult *result)
{
    *out = (struct pl_vulkan_presentation_feedback) {0};
    *result = VK_SUCCESS;
    if (!h || !get || !h->feedback.latest_submit_id)
        return false;

    VkPastPresentationTimingGOOGLE timings[VK_PRESENT_HISTORY];
    uint32_t count = VK_PRESENT_HISTORY;
    *result = get(device, swapchain, &count, timings);
    if (count > VK_PRESENT_HISTORY)
        *result = VK_ERROR_UNKNOWN;
    if (!vk_present_update(h, *result, timings, count))
        return false;
    *out = h->feedback;
    return true;
}
