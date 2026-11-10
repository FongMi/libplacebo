#include "utils.h"
#include "vulkan/swapchain_present.h"

static void observe(struct vk_present_history *h, uint32_t id, uint64_t actual)
{
    VkPastPresentationTimingGOOGLE t = {
        .presentID = id,
        .actualPresentTime = actual,
    };
    vk_present_observe(h, &t);
}

static VkResult query_result;
static int query_calls;
static uint32_t query_id;

static VKAPI_ATTR VkResult VKAPI_CALL query(VkDevice device, VkSwapchainKHR sw,
                     uint32_t *count, VkPastPresentationTimingGOOGLE *timings)
{
    REQUIRE(*count == VK_PRESENT_HISTORY);
    query_calls++;
    *count = query_id ? 1 : 0;
    if (query_id)
        timings[0] = (VkPastPresentationTimingGOOGLE) {
            .presentID = query_id,
            .actualPresentTime = 500,
        };
    return query_result;
}

int main(void)
{
    struct vk_present_history h = {0};
    h.feedback.generation = 1;
    vk_present_submit(&h, 1, 100, VK_SUCCESS);
    observe(&h, 1, 130);
    REQUIRE(h.feedback.latest_submit_id == 1);
    REQUIRE(h.feedback.matched_submit_begin_ns == 100);
    REQUIRE(h.feedback.actual_present_ns == 130);

    vk_present_submit(&h, 2, 120, VK_SUBOPTIMAL_KHR);
    observe(&h, 2, 150);
    observe(&h, 1, 130); // The query need not return records in time order.
    REQUIRE(h.feedback.matched_submit_id == 2);
    observe(&h, 99, 200); // Unknown IDs are not observations or drop counts.
    observe(&h, 2, 110); // A timestamp cannot predate its submission.
    REQUIRE(h.feedback.actual_present_ns == 150);

    vk_present_submit(&h, 3, 140, VK_ERROR_OUT_OF_DATE_KHR);
    REQUIRE(!h.feedback.latest_submit_id);
    observe(&h, 3, 200);
    REQUIRE(h.feedback.matched_submit_id == 2);
    vk_present_submit(&h, 3, 140, VK_ERROR_DEVICE_LOST);
    REQUIRE(!h.feedback.latest_submit_id);
    vk_present_submit(&h, 0, 140, VK_SUCCESS);
    REQUIRE(!h.feedback.latest_submit_id);

    vk_present_submit(&h, 1 + VK_PRESENT_HISTORY, 200, VK_SUCCESS);
    observe(&h, 1, 300); // Evicted identity must not match the reused slot.
    REQUIRE(h.feedback.actual_present_ns == 150);
    observe(&h, 1 + VK_PRESENT_HISTORY, 250);
    REQUIRE(h.feedback.actual_present_ns == 250);

    h = (struct vk_present_history) {0};
    h.feedback.generation = 2;
    observe(&h, 1, 300); // A new native swapchain starts with no old receipts.
    REQUIRE(!h.feedback.matched_submit_id);
    vk_present_submit(&h, 1, 400, VK_SUCCESS);
    observe(&h, 1, 450);
    REQUIRE(h.feedback.generation == 2);
    REQUIRE(h.feedback.matched_submit_begin_ns == 400);
    vk_present_submit(&h, 2, 460, VK_SUCCESS);
    vk_present_submit(&h, 3, 470, VK_SUCCESS);
    VkPastPresentationTimingGOOGLE partial[] = {
        {.presentID = 3, .actualPresentTime = 490},
        {.presentID = 2, .actualPresentTime = 480},
    };
    REQUIRE(vk_present_update(&h, VK_INCOMPLETE, partial, 2));
    REQUIRE(h.feedback.actual_present_ns == 490);
    REQUIRE(h.feedback.matched_submit_id == 3);
    REQUIRE(vk_present_update(&h, VK_SUCCESS, NULL, 0));
    REQUIRE(!vk_present_update(&h, VK_ERROR_DEVICE_LOST, NULL, 0));
    REQUIRE(!vk_present_update(&h, VK_ERROR_OUT_OF_DATE_KHR, NULL, 0));
    REQUIRE(!vk_present_update(&h, VK_SUCCESS, NULL, 0));
    struct pl_vulkan_presentation_feedback out;
    VkResult result;
    out.generation = 99;
    REQUIRE(!vk_present_poll(NULL, VK_NULL_HANDLE, VK_NULL_HANDLE, query,
                             &out, &result));
    REQUIRE(!out.generation && !query_calls);
    vk_present_submit(&h, 4, 495, VK_SUCCESS);
    REQUIRE(!vk_present_poll(&h, VK_NULL_HANDLE, VK_NULL_HANDLE, NULL,
                             &out, &result));
    REQUIRE(!out.latest_submit_id && !query_calls);
    query_id = 4;
    query_result = VK_INCOMPLETE;
    REQUIRE(vk_present_poll(&h, VK_NULL_HANDLE, VK_NULL_HANDLE, query,
                            &out, &result));
    REQUIRE(out.actual_present_ns == 500 && query_calls == 1);
    query_result = VK_ERROR_DEVICE_LOST;
    REQUIRE(!vk_present_poll(&h, VK_NULL_HANDLE, VK_NULL_HANDLE, query,
                             &out, &result));
    REQUIRE(!out.actual_present_ns && !out.generation);
    query_result = VK_SUCCESS;
    query_id = 0;
    REQUIRE(!vk_present_poll(&h, VK_NULL_HANDLE, VK_NULL_HANDLE, query,
                             &out, &result));
    REQUIRE(!out.matched_submit_id); // No reuse of pre-error observations.
    vk_present_submit(&h, 5, 499, VK_SUCCESS);
    query_id = 5;
    REQUIRE(vk_present_poll(&h, VK_NULL_HANDLE, VK_NULL_HANDLE, query,
                            &out, &result));
    query_result = VK_ERROR_OUT_OF_DATE_KHR;
    REQUIRE(!vk_present_poll(&h, VK_NULL_HANDLE, VK_NULL_HANDLE, query,
                             &out, &result));
    REQUIRE(!h.feedback.latest_submit_id);
    REQUIRE(!h.feedback.latest_submit_begin_ns);
    REQUIRE(!out.generation);
    return 0;
}
